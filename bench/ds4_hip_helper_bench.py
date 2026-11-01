import urllib.request,json,time,sys,pathlib
out=pathlib.Path(__file__).resolve().parent/'results/2026-10-31-ds4-hip-helper'
out.mkdir(parents=True,exist_ok=True)
label=sys.argv[1]
def api(path,body=None):
 r=urllib.request.Request('http://127.0.0.1:8080'+path,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(r,timeout=1200))
assert not api('/status')['busy']
prompts=[
'Scrivi in italiano una guida tecnica dettagliata su come implementare una cache LRU thread safe in C++, con esempi, gestione della memoria e compromessi prestazionali. Continua fino a completare la spiegazione.',
'Implementa in Python un parser per espressioni aritmetiche con parentesi, precedenza, segni unari e messaggi di errore. Spiega il codice e fornisci numerosi esempi.',
]
res=[]
for i in range(4):
 body={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':prompts[i%2]}],'max_tokens':320,'temperature':0,'seed':42,'reasoning_effort':'none','stream':False}
 if len(sys.argv)>2: body['strata_tune']={'hip_helper':int(sys.argv[2])}
 before=api('/metrics');start=time.time();r=api('/v1/chat/completions',body);elapsed=time.time()-start
 row={'label':label,'i':i,'wall_s':elapsed,'request':body,'response':r,'engine':before['engine'],'metrics':api('/metrics')}
 (out/f'{label}-{i}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
 res.append(row)
 print(json.dumps({'label':label,'i':i,'wall_s':elapsed,'usage':r.get('usage'),'timings':r.get('timings')},ensure_ascii=False),flush=True)
(out/f'{label}.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
