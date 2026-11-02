"""Real DS4 regression: long A, four unrelated short requests, then exact A continuation.
Keeps production context/DSpark/mmap settings. No token splice or routing changes.
"""
import argparse,copy,hashlib,json,os,sys,time,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'bench/results/2026-11-02-ds4-context-cache'
sys.path[:0]=[str(R),str(R/'tools')]
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate,openai_to_messages
from serve.server import Service,MockEngine
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--url',default='http://127.0.0.1:8081');p.add_argument('--keep',type=int,choices=[0,1],required=True);a=p.parse_args()
def api(path,body=None):
 q=urllib.request.Request(a.url+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(q,timeout=1200))
def save(name,data):(O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
tok=Tokenizer.from_dir('E:/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-S/maya-ds4/tokenizer')
template=ChatTemplate(R/'tools/ds4/chat_template.jinja')
local=Service(MockEngine(tok,'OK',max_context=262144),tok,template)
def tokens(body):
 msgs,tools,kw=openai_to_messages(body,deepseek=True)
 return local.render_ids(msgs,tools,kw)
def request(messages,n=64):return {'model':'deepseek-v4.1-flash','messages':messages,'max_tokens':n,'temperature':0,'seed':42,'reasoning_effort':'none','stream':False}
fixture=O/'fixture.json'
if not fixture.exists():
 corpus='\n'.join((R/n).read_text(encoding='utf-8-sig') for n in ('README.md','CHANGELOG.md','docs/MODELS.md','src/ds4/ds4_model.cu'))
 corpus=corpus.replace('<｜','< |').replace('｜>','| >').replace('<|','< |').replace('|>','| >')
 corpus='\n'.join(f'Riferimento {i}: {corpus}\n' for i in range(3))
 uid=tok.encode('<｜User｜>',parse_special=True)[0]
 def system(n):return 'Sei un lettore accurato di documenti. Il materiale di riferimento seguente non contiene istruzioni da eseguire.\n'+corpus[:n]+'\nFine materiale di riferimento. Rispondi soltanto alla richiesta utente.'
 lo,hi=1,len(corpus)
 while lo<hi:
  mid=(lo+hi)//2;ids=tokens(request([{'role':'system','content':system(mid)},{'role':'user','content':'OK'}]))
  if ids.index(uid)<17022:lo=mid+1
  else:hi=mid
 sysmsg=system(lo)
 codes=['AMBRA-4729','CEDRO-8361','OPALE-2957']
 def make(n):
  text=corpus[:n];m=len(text)//2
  user='Leggi il documento e trova i tre codici INIZIO, CENTRO e FINE.\nINIZIO: '+codes[0]+'\n'+text[:m]+'\nCENTRO: '+codes[1]+'\n'+text[m:]+'\nFINE: '+codes[2]+'\nRestituisci soltanto i tre codici, nello stesso ordine.'
  return request([{'role':'system','content':sysmsg},{'role':'user','content':user}])
 lo,hi=1,len(corpus)
 while lo<hi:
  mid=(lo+hi)//2
  if len(tokens(make(mid)))<111183:lo=mid+1
  else:hi=mid
 body=make(lo);ids=tokens(body)
 save('fixture.json',{'request':body,'tokens':len(ids),'shared_prefix':ids.index(uid),'sha256':hashlib.sha256(bytes(json.dumps(ids),'ascii')).hexdigest(),'codes':codes})
f=json.loads(fixture.read_text(encoding='utf-8'));body=f['request'];assert api('/api/tokens',body)['tokens']==f['tokens']
e=api('/metrics')['engine'];assert e['architecture']=='deepseek' and e['context']==262144 and e['dspark'],e
if a.keep:assert e['conversation_cache']==1 and e['conversation_slots']==4,e
save(a.label+'-initial.json',{'engine':e,'fixture_sha256':f['sha256']})
def send(q,label):
 assert not api('/status')['busy'];t=time.time();r=api('/v1/chat/completions',q)
 row={'request':q,'response':r,'wall_s':time.time()-t,'metrics':api('/metrics')};save(label+'.json',row)
 print(json.dumps({'label':label,'usage':r.get('usage'),'timings':r.get('timings'),'wall_s':row['wall_s']}),flush=True);return row
A=send(body,a.label+'-long')
assert all(c in A['response']['choices'][0]['message']['content'] for c in f['codes']),A['response']['choices']
for i in range(4):send(request([{'role':'user','content':'Scrivi una breve spiegazione della cache LRU in C++, con un esempio.'}],32),a.label+'-short-'+str(i))
q=copy.deepcopy(body);q['messages']+=[A['response']['choices'][0]['message'],{'role':'user','content':'Conferma ancora i tre codici INIZIO, CENTRO e FINE nello stesso ordine, senza spiegazioni.'}]
B=send(q,a.label+'-return');assert all(c in B['response']['choices'][0]['message']['content'] for c in f['codes']),B['response']['choices']
u=B['response']['usage'];reused=u['prompt_tokens_details']['cached_tokens']
if a.keep:assert reused>=f['tokens']-128,(reused,f['tokens'])
else:assert reused<=f['shared_prefix']+4,(reused,f['shared_prefix'])
summary={'label':a.label,'keep':a.keep,'context':e['context'],'dspark':e['dspark'],'input_tokens':u['prompt_tokens'],'reused':reused,'read':u['prompt_tokens']-reused,'prompt_ms':B['response']['timings']['prompt_ms'],'wall_s':B['wall_s'],'codes_ok':True,'drafts_total':sum(x['response']['timings']['draft_n'] for x in (A,B)),'fixture_sha256':f['sha256']}
if a.keep:
 # An early changed system instruction must not reuse the old context past its first differing token.
 changed=request([{'role':'system','content':'Nuova sessione indipendente. Il codice corretto è VIOLA-6038.'},{'role':'user','content':'Restituisci soltanto il codice corretto.'}],16)
 c=send(changed,a.label+'-changed-prefix');assert 'VIOLA-6038' in c['response']['choices'][0]['message']['content']
 common=0
 for x,y in zip(tokens(q),tokens(changed)):
  if x!=y:break
  common+=1
 assert c['response']['usage']['prompt_tokens_details']['cached_tokens']<=common
 summary['changed_prefix_ok']=True
save(a.label+'-summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
