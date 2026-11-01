"""Sequential real-GLM regression/throughput checks for the HIP-backed CUDA tail cache."""
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('unified_check',ROOT/'bench/unified_backend_check.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.OUT=ROOT/'bench/results/2026-10-31-glm-hip-cache';b.OUT.mkdir(parents=True,exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--warmup',type=int,default=4);p.add_argument('--runs',type=int,default=8);p.add_argument('--document',action='store_true');a=p.parse_args()
e=b.api('/metrics')['engine'];assert e['backend']=='unified' and e['architecture']=='glm' and e['max_context']==262144 and e['mtp']==0 and e['expert_vram_complete']==1,e
assert 0<float(e['ram_gb'])<1 and e['helper_slots']>0,e
b.save(a.label+'-initial.json',b.api('/metrics'))
rows=[]
def check(row):
 t=row['response']['timings'];n=row['metrics']['tiers']['now']
 assert t.get('draft_n',0)==0,t
 assert n['expert_ssd_bytes']==0 and n['expert_gpu_complete']==1 and n['helper_dead']==0 and n['helper_fb']==0,n
 assert int(n['vram_used'])+int(n['helper_held'])==e['experts'],n
for i in range(a.warmup+a.runs):
 warm=i<a.warmup;idx=i if warm else i-a.warmup
 row=b.send(b.req(b.PROMPTS[i%2],320,{'spec':1}),a.label+('-warmup-' if warm else '-')+str(idx));check(row)
 assert row['response']['usage']['completion_tokens']==320,row['response']
 if not warm:rows.append(row)
if a.document:
 b.send(b.req('Rispondi soltanto OK.',8,{'spec':1}),a.label+'-reset')
 body,count=b.document(16384,a.label);body['strata_tune']={'spec':1}
 row=b.send(body,a.label+'-document');check(row)
 assert all(c in row['response']['choices'][0]['message']['content'] for c in b.CODES)
 row.update(codes_ok=True,prompt_tokens=count,prompt_sha256=hashlib.sha256(body['messages'][0]['content'].encode()).hexdigest());b.save(a.label+'-document.json',row)
 follow=b.req('Conferma ancora i tre codici nello stesso ordine.',64,{'spec':1});follow['messages']=body['messages']+[row['response']['choices'][0]['message']]+follow['messages']
 f=b.send(follow,a.label+'-followup');check(f);assert all(c in f['response']['choices'][0]['message']['content'] for c in b.CODES)
 assert f['response']['usage']['prompt_tokens_details']['cached_tokens']>count-128
 # Lend/return is a separate transition: measure warm decode again after the long prompt.
 for i in range(4):
  row=b.send(b.req(b.PROMPTS[i%2],320,{'spec':1}),a.label+'-post-long-'+str(i));check(row)
summary={'label':a.label,'engine':e,'n':len(rows),'output_tokens':sum(r['response']['timings']['predicted_n'] for r in rows),'decode_ms':sum(r['response']['timings']['predicted_ms'] for r in rows),'decode_tok_s':0,'per_request':[{'decode':r['response']['timings']['predicted_per_second'],'helper_experts_per_tok':r['metrics']['tiers']['now']['helper'],'helper_ms_per_tok':r['metrics']['tiers']['now']['helper_ms'],'cuda_held':r['metrics']['tiers']['now']['vram_used'],'amd_only':r['metrics']['tiers']['now']['helper_held'],'hip_readback_bytes':r['metrics']['tiers']['now']['expert_hip_bytes']} for r in rows]}
summary['decode_tok_s']=1000*summary['output_tokens']/summary['decode_ms'];b.save(a.label+'-summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
