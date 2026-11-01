from pathlib import Path
import sys,json
sys.path.insert(0,'E:/project-maya/tools')
from gguf_reader import GGUFFile
rows=[]
for path in sorted(Path('E:/Maya-glm-data/models/Maya-M').glob('*.gguf')):
 g=GGUFFile(path);rows.append({'file':path.name,'metadata':{k:v for k,v in g.metadata.items() if any(s in k.lower() for s in ('nextn','trunk_block','block_count','mtp'))},'nextn_tensor_names':[t.name for t in g.tensors if t.name.startswith('blk.45.') or 'mtp' in t.name.lower() or 'nextn' in t.name.lower()]})
p=Path('E:/project-maya/bench/results/2026-10-31-unified/glm-drafter-inventory.json');p.write_text(json.dumps(rows,indent=2));print(p.read_text())
