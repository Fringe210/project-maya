"""Validate recorded MiniMax compact-pool / 256K evidence without inference."""
import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'bench/results/2026-11-02-mm3-packed-256k'
def read(n):return json.loads((O/n).read_text(encoding='utf-8-sig'))
s=read('summary.json');e=s['engine'];cfg=json.loads((R/'maya-minimax.json').read_text(encoding='utf-8-sig'))
assert e['max_context']==262144 and e['kv_resident']==65536 and e['kv']=='int8' and e['dspark']
assert e['experts_vram']+e['helper_slots']==e['experts']==7296 and e['cpu_experts']==0
assert e['helper_allocated_bytes']==e['helper_payload_bytes']<=40*2**30 and e['helper_padding_bytes']==0
assert e['conversation_ram_mib']==24576 and cfg['env']['MAYA_MM3_SLOT_RAM_GB']=='24'
for key,val in [('--max-context','262144'),('--kv-resident','65536'),('--kv-cache','int8'),('--dspark','on')]:assert cfg['args'][cfg['args'].index(key)+1]==val
for n in ['hip-packed-test.log','hip-legacy-test.log']:assert 'ALL OK (0 failed)' in (O/n).read_text()
assert 'ctx=262144 ring=65536' in (O/'kv-test.log').read_text() and 'ok (0 failures)' in (O/'kv-test.log').read_text()
assert 'Ran 63 tests' in (O/'python-tests.log').read_text() and 'OK' in (O/'python-tests.log').read_text()
assert read('exports.json')['removed']==[]
log=(O/'native.log').read_text(encoding='utf-8',errors='replace')
assert f"all {e['helper_slots']} experts byte-checked" in log
snap=[(int(n),int(b)) for n,b in re.findall(r'parked (\d+) tokens, (\d+) bytes',log) if int(n)>=260000]
assert snap and all(12*2**30<b<=24*2**30 for n,b in snap)
for label in ['40k','80k','261k']:
 c=s['cases'][label];assert c['read']==1 and c['same_answer']
 assert read(label+'-cold.json')['request']==read(label+'-return.json')['request']
previous=R/'bench/results/2026-11-02-glm-mm3-context-cache'
old=json.loads((previous/'mm3-fixed-long.json').read_text());new=read('40k-cold.json')
assert old['request']==new['request']
old_engine=json.loads((previous/'mm3-fixed-initial.json').read_text())['engine']
comparison={'previous_context':131072,'previous_resident':32768,'previous_prefill_tok_s':old['response']['timings']['prompt_per_second'],'new_prefill_tok_s':new['response']['timings']['prompt_per_second'],'comparison':'single sequential cold request, not interleaved','previous_helper_slots':old_engine['helper_slots'],'new_helper_slots':e['helper_slots'],'previous_arena_bytes':old_engine['helper_allocated_bytes'],'new_arena_bytes':e['helper_allocated_bytes'],'same_count_fixed_stride_equivalent_bytes':e['helper_slots']*21676032}
comparison['prefill_delta_pct']=100*(comparison['new_prefill_tok_s']/comparison['previous_prefill_tok_s']-1)
s.update(comparison=comparison,full_snapshot_bytes=snap,binary_sha256={n:hashlib.sha256((R/'build'/n).read_bytes()).hexdigest() for n in ['maya-backend.exe','maya-hhelper.dll']})
(O/'verified-summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
print(json.dumps({'engine':e,'cases':s['cases'],'long_context_code':s['long_context_code'],'comparison':comparison,'full_snapshot_bytes':snap},indent=2))
