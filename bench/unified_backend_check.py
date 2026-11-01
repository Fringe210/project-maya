"""Real model checks for the single backend on localhost:8080; saves requests and timings."""
import argparse,json,time,urllib.request,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'bench/results/2026-10-31-unified'
BASE=os.environ.get('MAYA_BENCH_URL','http://127.0.0.1:8080')
def api(path,body=None):
 q=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(q,timeout=1200))
def save(name,row):
 (OUT/name).write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
def req(text,n=320,tune=None):
 return {'model':api('/metrics')['engine']['model'],'messages':[{'role':'user','content':text}],'max_tokens':n,'temperature':0,'seed':42,'reasoning_effort':'none','stream':False,**({'strata_tune':tune} if tune else {})}
def send(body,label):
 assert not api('/status')['busy']
 start=time.time();result=api('/v1/chat/completions',body);elapsed=time.time()-start
 row={'label':label,'wall_s':elapsed,'request':body,'response':result,'metrics':api('/metrics')}
 save(label+'.json',row)
 print(json.dumps({'label':label,'wall_s':elapsed,'usage':result.get('usage'),'timings':result.get('timings')},ensure_ascii=False),flush=True)
 return row
PROMPTS=[
'Scrivi in italiano una guida tecnica dettagliata su come implementare una cache LRU thread safe in C++, con esempi, gestione della memoria e compromessi prestazionali. Continua fino a completare la spiegazione.',
'Implementa in Python un parser per espressioni aritmetiche con parentesi, precedenza, segni unari e messaggi di errore. Spiega il codice e fornisci numerosi esempi.'
]
CODES=['AMBRA-4729','CEDRO-8361','OPALE-2957']
def document(target,nonce=None):
 text=(ROOT/'README.md').read_text(encoding='utf-8-sig')+'\n'+(ROOT/'CHANGELOG.md').read_text(encoding='utf-8-sig')
 def make(n):
  s=text[:n];h=len(s)//2
  return req(('Identificativo prova: '+nonce+'\n' if nonce else '')+'Leggi il documento e restituisci i tre codici INIZIO, CENTRO e FINE.\nINIZIO: '+CODES[0]+'\n'+s[:h]+'\nCENTRO: '+CODES[1]+'\n'+s[h:]+'\nFINE: '+CODES[2]+'\nRispondi soltanto con i tre codici.',96)
 lo,hi=1,len(text)
 while lo<hi:
  mid=(lo+hi)//2
  if api('/api/tokens',make(mid))['tokens']<target:lo=mid+1
  else:hi=mid
 body=make(lo);count=api('/api/tokens',body)['tokens'];assert target<=count<=target+4
 return body,count
def main():
 p=argparse.ArgumentParser();p.add_argument('--family',required=True,choices=['deepseek','glm','minimax']);p.add_argument('--label',required=True)
 p.add_argument('--runs',type=int,default=4);p.add_argument('--bypass',type=int,choices=[0,1]);p.add_argument('--document',action='store_true');p.add_argument('--parity',action='store_true');p.add_argument('--failure',action='store_true');p.add_argument('--vram-only',action='store_true');p.add_argument('--spec',type=int,choices=[1,3,4])
 a=p.parse_args();m=api('/metrics');e=m['engine'];assert e.get('backend')=='unified' and e.get('architecture')==a.family,e
 assert e['max_context']==(131072 if a.family=='minimax' else 262144),e
 if a.family=='minimax':assert e['kv']=='int8' and e['prefill_chunk']==4096,e
 if a.family in ('minimax','deepseek'):assert e['dspark'],e
 if a.vram_only:
  assert a.family=='glm' and e.get('expert_vram_complete')==1,e
  assert float(e['ram_gb'])<1 and 0<e['helper_allocated_bytes']<=32e9,e
  assert len(m['gpu_map'])==2,m['gpu_map']
 save(a.label+'-initial.json',{'engine':e,'gpu_map':m['gpu_map']})
 tune={}
 if a.bypass is not None:tune['hip_helper']=a.bypass
 if a.spec is not None:tune['spec']=a.spec
 tune=tune or None
 for i in range(a.runs):
  row=send(req(PROMPTS[i%2],tune=tune),a.label+'-'+str(i))
  assert row['response']['usage']['completion_tokens']==320,row['response']
  if a.vram_only:
   n=row['metrics']['tiers']['now'];assert n['expert_gpu_complete']==1 and n['expert_ssd_bytes']==0,n
   assert int(n['vram_used'])+int(n['helper_held'])==int(e['experts']),n
 if a.document or a.parity:
  send(req('Rispondi soltanto OK.',8,tune),a.label+'-reset')
  body,count=document(4268 if a.parity else 16384, a.label if a.family=='glm' else None)
  if a.parity:body['strata_tune']={'prefill_staging':2,'kv_restore':1};body['max_tokens']=32
  elif tune:body['strata_tune']=tune
  logpath=ROOT/'maya-minimax.log'
  offset=logpath.stat().st_size if a.parity and logpath.exists() else 0
  row=send(body,a.label+'-document');text=row['response']['choices'][0]['message']['content']
  assert all(c in text for c in CODES),text
  if a.vram_only:
   n=row['metrics']['tiers']['now'];assert n['expert_ssd_bytes']==0 and n['expert_hip_bytes']>0,n
  row.update(codes_ok=True,prompt_tokens=count,prompt_sha256=hashlib.sha256(body['messages'][0]['content'].encode()).hexdigest())
  save(a.label+'-document.json',row)
  if a.parity:
   import re
   with logpath.open('rb') as f:f.seek(offset);log=f.read().decode('utf-8',errors='replace')
   matches=re.findall(r'prefill staging parity: (\d+) layers checked, max absolute error ([0-9.eE+-]+)',log)
   assert matches,'Missing numerical parity verification in engine log'
   assert all(int(n)>0 and float(error)==0 for n,error in matches),matches
   save(a.label+'-parity.json',{'checks':matches,'log':log})
  if not a.parity:
   follow=req('Conferma ancora i tre codici nello stesso ordine.',64,tune)
   follow['messages']=body['messages']+[row['response']['choices'][0]['message']]+follow['messages']
   f=send(follow,a.label+'-followup')
   assert all(c in f['response']['choices'][0]['message']['content'] for c in CODES)
   assert f['response']['usage']['prompt_tokens_details']['cached_tokens']>count-128
 if a.failure:
  text='I codici sono AMBRA-4729, CEDRO-8361 e OPALE-2957. Ripetili e scrivi una spiegazione dettagliata della cache LRU in almeno dieci punti.'
  row=send(req(text,192,{'hip_helper_fail':1}),a.label+'-fallback')
  assert all(c in row['response']['choices'][0]['message']['content'] for c in CODES),row['response']
  stat=row['metrics']['tiers']['now'];assert not a.vram_only or stat['expert_ssd_bytes']==0,stat;assert stat['helper_dead']==1 and stat['helper_fb']>0,stat
 print('CHECK PASSED '+a.label,flush=True)
if __name__=='__main__':main()
