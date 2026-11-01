"""Sample NVIDIA PCIe separately while prefill and decode run; record expert readback bytes."""
from pathlib import Path
import sys,time,threading,argparse,statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'bench'))
from unified_backend_check import api,req,send,save,PROMPTS
from serve.telemetry import _open_nvml,_Nvml
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--no-weight-readback',action='store_true');a=p.parse_args()
lib=_open_nvml();assert lib is not None;assert (getattr(lib,'nvmlInit_v2',None) or lib.nvmlInit)()==0;card=_Nvml(lib,0);assert card.ok()
before=api('/metrics');samples=[];stop=threading.Event();start=time.monotonic()
def sample():
 while not stop.is_set():
  s=api('/status');gpu=card.read()
  if s['busy']:samples.append({'t':time.monotonic()-start,'phase':'decode' if s.get('generated',0)>0 else 'prefill','rx_mb_s':gpu['pcie_rx_mb'],'tx_mb_s':gpu['pcie_tx_mb']})
  stop.wait(.075)
t=threading.Thread(target=sample);t.start()
try:r=send(req(PROMPTS[0],320,{'spec':4}),a.label+'-request')
finally:stop.set();t.join()
after=r['metrics'];n=after['tiers']['now'];old=before['tiers']['now'];delta=n['expert_hip_bytes']-old.get('expert_hip_bytes',0)
phases={}
for phase in ('prefill','decode'):
 s=[x for x in samples if x['phase']==phase];assert s,phase
 phases[phase]={'samples':len(s),'mean_rx_mb_s':statistics.mean(x['rx_mb_s'] for x in s),'peak_rx_mb_s':max(x['rx_mb_s'] for x in s),'mean_tx_mb_s':statistics.mean(x['tx_mb_s'] for x in s)}
result={'engine':after['engine'],'phases':phases,'expert_hip_weight_bytes':delta,'expert_ssd_bytes':n['expert_ssd_bytes'],'timings':r['response']['timings'],'samples':samples}
assert n['expert_ssd_bytes']==0
if a.no_weight_readback:assert delta==0,delta
save(a.label+'.json',result);print(result['phases'], 'expert weight bytes=',delta,flush=True)
