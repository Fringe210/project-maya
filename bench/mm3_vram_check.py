from pathlib import Path
import subprocess,json,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serve import gpu_map
OUT=ROOT/'bench/results/2026-10-31-unified'
cmd=[str(ROOT/'build/maya-backend.exe'),'--family','minimax','--serve','--pack','H:/Maya-data/models/MiniMax-M3.1-Flash-Maya-S/maya-mm3','--max-context','8192','--kv-cache','int8','--kv-streaming','off','--dspark','off','--vision','off','--helper','off','--resident-cap','8','--staging','2','--chunk','32','--no-pin-pool']
with (OUT/'minimax-vram-probe-stderr.log').open('w',encoding='utf-8') as err:
 p=subprocess.run(cmd,input='QUIT\n',stdout=subprocess.PIPE,stderr=err,text=True,encoding='utf-8',errors='replace',timeout=300)
(OUT/'minimax-vram-probe-stdout.log').write_text(p.stdout,encoding='utf-8')
assert p.returncode==0,p.returncode
rows=[gpu_map.parse(line) for line in p.stdout.splitlines() if line.startswith('VRAM ')]
assert rows, p.stdout
for row in rows:
 assert row['pool']>5*(1<<30),row
 assert row['act']<3*(1<<30),row
(OUT/'minimax-vram-probe.json').write_text(json.dumps({'command':cmd,'rows':rows,'resident_experts_accounted_as_pool':True},indent=2),encoding='utf-8')
print('MiniMax VRAM categories passed:',json.dumps(rows))
