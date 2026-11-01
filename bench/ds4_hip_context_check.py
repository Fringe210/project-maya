import json,urllib.request,time,pathlib,hashlib
r=pathlib.Path(__file__).resolve().parents[1];out=r/'bench/results/2026-10-31-ds4-hip-helper'
def api(path,body=None):
 q=urllib.request.Request('http://127.0.0.1:8080'+path,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(q,timeout=1200))
source=(r/'README.md').read_text(encoding='utf-8-sig')+'\n'+(r/'CHANGELOG.md').read_text(encoding='utf-8-sig')
codes=['AMBRA-4729','CEDRO-8361','OPALE-2957']
def body(n):
 text=source[:n];h=len(text)//2
 prompt='Leggi il documento e restituisci i tre codici di controllo INIZIO, CENTRO e FINE.\nINIZIO: '+codes[0]+'\n'+text[:h]+'\nCENTRO: '+codes[1]+'\n'+text[h:]+'\nFINE: '+codes[2]+'\nRispondi soltanto con i tre codici di controllo.'
 return {'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':prompt}],'max_tokens':96,'temperature':0,'seed':42,'reasoning_effort':'none','stream':False}
lo,hi=1,len(source)
while lo<hi:
 mid=(lo+hi)//2
 if api('/api/tokens',body(mid))['tokens']<16384:lo=mid+1
 else:hi=mid
req=body(lo);count=api('/api/tokens',req)['tokens'];assert 16384<=count<=16388
(out/'request-16k.json').write_text(json.dumps(req,ensure_ascii=False,indent=2),encoding='utf-8')
for mode in (0,1):
 reset=body(1);reset['messages']=[{'role':'user','content':'Rispondi soltanto OK.'}];reset['max_tokens']=8;reset['strata_tune']={'hip_helper':mode};api('/v1/chat/completions',reset)
 req['strata_tune']={'hip_helper':mode};start=time.time();res=api('/v1/chat/completions',req)
 text=res['choices'][0]['message']['content'];assert all(c in text for c in codes),text
 row={'mode':mode,'prompt_tokens':count,'sha256':hashlib.sha256(req['messages'][0]['content'].encode()).hexdigest(),'wall_s':time.time()-start,'response':res,'metrics':api('/metrics'),'codes_ok':True}
 (out/f'context-16k-{mode}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'mode':mode,'tokens':count,'timings':res.get('timings'),'text':text}),flush=True)
