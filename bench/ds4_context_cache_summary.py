"""Summarize saved DS4 exact conversation-cache evidence without running inference."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'bench/results/2026-11-02-ds4-context-cache'
def read(n):return json.loads((O/n).read_text(encoding='utf-8-sig'))
before=read('control-summary.json');after=read('fixed-summary.json');b=read('control-return.json');a=read('fixed-return.json')
assert b['request']==a['request'],'The resumed requests differ'
assert before['fixture_sha256']==after['fixture_sha256']
assert after['reused']>before['reused'] and after['codes_ok'] and after['changed_prefix_ok']
assert before['drafts_total']>0 and after['drafts_total']>0
cfg=json.loads((R/'maya-ds4.json').read_text(encoding='utf-8-sig'));args=cfg['args']
for flag,value in (('--max-context','262144'),('--dspark','on'),('--mmap-experts','off')):assert args[args.index(flag)+1]==value
assert all(not any(word in k for word in ('REORDER','PROMPT_RESIDENT')) for k in cfg['env'])
r={'date':'2026-11-02','cause':'Conversation parking was disabled; unrelated short requests replaced the active state, leaving the shared system/tools prefix.','historical_log':{'prompt':111183,'reused':17022,'read':94161,'prompt_ms':33140,'original_request_token_streams_available':False},'control':before,'fixed':after,'same_resumed_request':True,'prefill_latency_ratio':before['prompt_ms']/after['prompt_ms'],'wall_time_ratio':before['wall_s']/after['wall_s'],'constraints':{'context':262144,'dspark':'on','mmap_experts':'off','amd_budget_decimal_GB':40,'splice':False,'approximate_routing':False},'cache':{'enabled':True,'ram_slots':4,'steady_state_RAM_GiB_budget':1,'directory':'E:/project-maya/data/ds4-slots','persistence':'Existing behavior: RAM first, disk spill/clean shutdown; at least one RAM slot may exceed the budget.'},'tests':{'restore_points_and_slot_environment':'12 passed','unified_launcher':'2 passed','real_document_codes':True,'changed_system_prefix':True},'limits':['Historical prompts were not recorded, so additional client-side history edits in the original incident cannot be excluded.','The controlled reproduction demonstrates state loss across unrelated requests with parking off; genuinely changed prefixes still require exact recomputation.','Latency ratios concern this one resumed-request scenario, not decode throughput or arbitrary new sessions.']}
if (O/'branch-summary.json').exists():r['branches']=read('branch-summary.json')
for name in ('build/maya-backend.exe','build/maya-backend-cache-before.exe'):
 p=R/name;r.setdefault('binaries',{})[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
r['runtime']=read('fixed-initial.json')['engine'];assert r['runtime']['conversation_cache']==1 and r['runtime']['context']==262144
(O/'summary.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'before_read':before['read'],'after_read':after['read'],'before_prefill_ms':before['prompt_ms'],'after_prefill_ms':after['prompt_ms'],'prefill_latency_ratio':r['prefill_latency_ratio'],'wall_time_ratio':r['wall_time_ratio'],'branches':r.get('branches')},indent=2))
