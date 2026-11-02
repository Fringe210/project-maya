"""Verify and summarize saved real-model evidence; never starts inference."""
import json,re,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'bench/results/2026-11-02-glm-mm3-context-cache'
sys.path[:0]=[str(R),str(R/'tools')]
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate,openai_to_messages
from serve.server import Service,MockEngine,chat_template_path
def read(n):return json.loads((O/n).read_text(encoding='utf-8-sig'))
def reuse(row):return row['response']['usage']['prompt_tokens_details']['cached_tokens']
def common(a,b):
 n=0
 for x,y in zip(a,b):
  if x!=y:break
  n+=1
 return n
result={}
for family,label,ctx in [('minimax','mm3-fixed',131072),('glm','glm-fixed',262144)]:
 cfg=json.loads((R/f'maya-{family}.json').read_text(encoding='utf-8-sig'))
 before=read('mm3-before.json' if family=='minimax' else 'glm-before.json')
 assert cfg['args']==before['args'] and cfg['args'][cfg['args'].index('--max-context')+1]==str(ctx)
 for k,v in before['env'].items():assert cfg['env'][k]==v,(family,k)
 engine=read(label+'-initial.json')['engine'];assert engine['conversation_cache']==1 and engine['conversation_slots']==4
 tok=Tokenizer.from_dir(cfg['tokenizer']);tpl=ChatTemplate(chat_template_path(cfg['tokenizer']));svc=Service(MockEngine(tok,'OK',max_context=ctx),tok,tpl)
 def tokens(q):
  m,t,k=openai_to_messages(q);return svc.render_ids(m,t,k)
 A=read(label+'-long.json');B=read(label+'-return.json');C=read(label+'-changed-prefix.json');D=read(label+'-changed-middle.json');E=read(label+'-original-branch.json')
 assert A['request']==B['request']==E['request']
 assert reuse(B)>=39999 and reuse(E)>=39999
 assert reuse(C)<=common(tokens(A['request']),tokens(C['request']))
 assert reuse(D)<=common(tokens(A['request']),tokens(D['request']))
 result[family]=read(label+'-summary.json')
 result[family]['snapshot_ram_mib']=engine['conversation_ram_mib']
 result[family]['early_edit_reuse']=reuse(C)
 if family=='minimax':
  old=read('mm3-before-return.json');assert old['request']==B['request']
  result[family]['control']=read('mm3-before-summary.json')
  result[family]['prefill_latency_ratio']=old['response']['timings']['prompt_ms']/B['response']['timings']['prompt_ms']
  assert engine['kv']=='int8' and engine['dspark'] and engine['max_context']==ctx
  log=(O/'mm3-fixed-native.log').read_text(encoding='utf-8',errors='replace')
  checks=re.findall(r'exact byte roundtrip OK \((\d+) bytes',log)
  assert len(checks)>=2 and 'FAILED' not in log
  result[family]['exact_byte_checks']=[int(n) for n in checks]
  result[family]['parity_check_enabled_in_benchmark']=True
 else:
  assert engine['context']==ctx
  result[family]['slot_files_unchanged']=read('glm-slot-files-before.json')==read('glm-slot-files-after.json')
  assert result[family]['slot_files_unchanged']
assert '0 failures' in (O/'kv-prefix-restore.log').read_text()
assert 'OK' in (O/'python-tests-final.log').read_text()
(O/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
