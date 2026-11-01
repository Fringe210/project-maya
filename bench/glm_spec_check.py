"""Real GLM structured-output check with MTP bypassed and enabled."""
import json,sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from unified_backend_check import api,req,send,save
args=argparse.ArgumentParser();args.add_argument('--nvidia-only',action='store_true');a=args.parse_args()
engine=api('/metrics')['engine'];assert engine['architecture']=='glm' and engine['mtp']==1,engine
if a.nvidia_only:assert not engine.get('helper_slots') and len(api('/metrics')['gpu_map'])==1,engine
else:assert engine['expert_vram_complete']==1,engine
prefix='glm-nvidia-structured' if a.nvidia_only else 'glm-mtp-structured'
prompt='Scrivi soltanto JSON valido senza markdown nella forma {"squares":[...]}, con i quadrati dei numeri interi da 1 a 20 in ordine. Non aggiungere spiegazioni.'
rows=[]
for spec in (1,3,4):
 row=send(req(prompt,256,{'spec':spec}),f'{prefix}-{spec}')
 text=row['response']['choices'][0]['message']['content'];value=json.loads(text)
 assert value=={'squares':[i*i for i in range(1,21)]},value
 if not a.nvidia_only:assert row['metrics']['tiers']['now']['expert_ssd_bytes']==0
 rows.append(row)
assert rows[0]['response']['timings']['draft_n']==0
assert all(row['response']['timings']['draft_n']>0 for row in rows[1:]),[row['response']['timings'] for row in rows]
save(prefix+'-summary.json',{'correct_both':True,'same_content':rows[0]['response']['choices'][0]['message']['content']==rows[1]['response']['choices'][0]['message']['content'],'timings':[r['response']['timings'] for r in rows]})
print('MTP STRUCTURED PARITY PASSED',flush=True)
