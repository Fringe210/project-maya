"""Exact-reuse negative control: edit the middle of a cached document, then recover the original branch."""
import copy,json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'tools')]
import unified_backend_check as b
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate,openai_to_messages
from serve.server import Service,MockEngine
b.BASE='http://127.0.0.1:8080';b.OUT=R/'bench/results/2026-11-02-ds4-context-cache'
f=json.loads((b.OUT/'fixture.json').read_text(encoding='utf-8'));original=json.loads((b.OUT/'fixed-return.json').read_text(encoding='utf-8'))['request']
tok=Tokenizer.from_dir('E:/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-S/maya-ds4/tokenizer');svc=Service(MockEngine(tok,'OK',max_context=262144),tok,ChatTemplate(R/'tools/ds4/chat_template.jinja'))
def ids(q):
 m,t,k=openai_to_messages(q,deepseek=True);return svc.render_ids(m,t,k)
def common(x,y):
 n=0
 for a,c in zip(x,y):
  if a!=c:break
  n+=1
 return n
edited=copy.deepcopy(f['request']);old=f['codes'][1];new='ZAFIRO-9186';assert old in edited['messages'][1]['content'];edited['messages'][1]['content']=edited['messages'][1]['content'].replace(old,new)
# Compare with every saved public branch that could legally satisfy this request.
candidates=[original,json.loads((b.OUT/'fixed-long.json').read_text(encoding='utf-8'))['request']]
bound=max(common(ids(q),ids(edited)) for q in candidates)
r=b.send(edited,'fixed-middle-edit');u=r['response']['usage'];reused=u['prompt_tokens_details']['cached_tokens'];text=r['response']['choices'][0]['message']['content']
assert all(c in text for c in (f['codes'][0],new,f['codes'][2])) and old not in text,text
assert reused<=bound,(reused,bound)
back=b.send(original,'fixed-original-again');u2=back['response']['usage'];restored=u2['prompt_tokens_details']['cached_tokens'];assert restored>=f['tokens']-128,(restored,f['tokens'])
assert all(c in back['response']['choices'][0]['message']['content'] for c in f['codes'])
b.save('branch-summary.json',{'middle_edit':{'first_different_token':bound,'reused':reused,'read':u['prompt_tokens']-reused,'prompt_ms':r['response']['timings']['prompt_ms'],'new_code_ok':True},'original_branch':{'reused':restored,'read':u2['prompt_tokens']-restored,'prompt_ms':back['response']['timings']['prompt_ms'],'codes_ok':True}})
print('EXACT BRANCH CHECK PASSED',flush=True)
