"""Aggregate saved sequential GLM all-VRAM optimization evidence; never runs inference."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'bench/results/2026-10-31-glm-hip-cache'
def read(name):return json.loads((O/name).read_text(encoding='utf-8-sig'))
def aggregate(label):
 s=read(label+'-summary.json');rows=[read(label+'-'+str(i)+'.json') for i in range(s['n'])]
 prev=read(label+'-warmup-3.json')['metrics']['tiers']['now']['expert_hip_bytes']
 final=rows[-1]['metrics']['tiers']['now']
 out={'decode_tok_s':s['decode_tok_s'],'requests':s['n'],'output_tokens':s['output_tokens'],'helper_experts_per_tok':sum(r['metrics']['tiers']['now']['helper'] for r in rows)/len(rows),'helper_ms_per_tok':sum(r['metrics']['tiers']['now']['helper_ms'] for r in rows)/len(rows),'cache_weight_readback_bytes':final['expert_hip_bytes']-prev,'runtime_expert_ssd_bytes':final['expert_ssd_bytes'],'final_tiers':final,'engine':s['engine']}
 if (O/(label+'-document.json')).exists():
  d=read(label+'-document.json');f=read(label+'-followup.json');post=[read(label+'-post-long-'+str(i)+'.json') for i in range(4)]
  out.update(document_tokens=d['prompt_tokens'],prefill_tok_s=d['response']['timings']['prompt_per_second'],document_codes_ok=d['codes_ok'],cached_followup_tokens=f['response']['usage']['prompt_tokens_details']['cached_tokens'],post_long_decode_tok_s=1000*sum(x['response']['timings']['predicted_n'] for x in post)/sum(x['response']['timings']['predicted_ms'] for x in post))
 return out
result={'date':'2026-10-31','conditions':{'model':'GLM-5.3-Flash Maya-M','context':262144,'mtp_loaded':False,'parallel':1,'temperature':0,'seed':42,'thinking':'none','warmup_requests_excluded':4,'output_tokens_each':320,'amd_budget_decimal_GB':32},'measurements':{label:aggregate(label) for label in ('control','optimized','async','backed','lfu')},'hip_call_latency':read('hip-call-latency.json'),'reference_nvidia_only':{'decode_tok_s':73.74865,'prefill_tok_s':2628.3,'expert_RAM_GiB':22.8,'source':'../2026-10-31-unified/summary.json','measured_previously':True},'limits':['The NVIDIA-only reference was measured in the previous campaign; control and optimized all-VRAM variants are fresh sequential measurements.','Long batched prefill still stages AMD expert weights and computes on NVIDIA.','CUDA cache admissions move whole expert weights AMD -> bounded host staging -> NVIDIA; cache hits require no weight copies.','All routed experts have GPU backing; host KV, embeddings, staging, Python and driver memory remain.','262K allocated and real 16K/cache transitions checked; no full 262K stress or parallel throughput test.']}
a=result['measurements']['lfu']['decode_tok_s'];b=result['measurements']['control']['decode_tok_s'];result['gain_pct']=100*(a/b-1);result['relative_to_previous_nvidia_pct']=100*(a/result['reference_nvidia_only']['decode_tok_s']-1)
for name in ('build/maya-backend.exe','build/maya-hhelper-lowlatency.dll','build/maya-hhelper.dll'):
 p=R/name
 if p.exists():result.setdefault('binaries',{})[name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
result['verification']={'async_landing_64_byte_checks_passed':'all requested asynchronous landing byte checks passed' in (O/'glm-cache-lfu.log').read_text(encoding='utf-8',errors='replace'),'expert_parity_log':(O/'zero-copy-parity.log').read_text(encoding='utf-8'),'ui_component':(O/'tier-component.json').exists(),'python':'69 of 71 passed first run; two existing locale-dependent expectations corrected and both passed targeted rerun'}
if (O/'lfu-long-code.json').exists():
 d=read('lfu-long-code.json');result['long_context_code_check']={'prompt_tokens':d['response']['usage']['prompt_tokens'],'completion_tokens':d['response']['usage']['completion_tokens'],'cached_tokens':d['response']['usage']['prompt_tokens_details']['cached_tokens'],'decode_tok_s':d['response']['timings']['predicted_per_second'],'prefill_tok_s':d['response']['timings']['prompt_per_second'],'codes_ok':all(c in d['response']['choices'][0]['message']['content'] for c in ('AMBRA-4729','CEDRO-8361','OPALE-2957')),'expert_ssd_bytes':d['metrics']['tiers']['now']['expert_ssd_bytes']}
result['production_profile']='maya-glm.json (see production-selected.json); NVIDIA+AMD, no MTP'
if (O/'production-runtime.json').exists():result['production_runtime']=read('production-runtime.json')
(O/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'control':b,'optimized_backed':a,'gain_pct':result['gain_pct'],'relative_to_previous_nvidia_pct':result['relative_to_previous_nvidia_pct'],'prefill':result['measurements']['lfu'].get('prefill_tok_s'),'post_long':result['measurements']['lfu'].get('post_long_decode_tok_s'),'byte_checks':result['verification']['async_landing_64_byte_checks_passed']},indent=2))
