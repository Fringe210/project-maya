"""Real-expert HIP call latency and identical-output check across transfer/wait modes."""
import ctypes as c,os,time,json,math
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'bench/results/2026-10-31-glm-hip-cache'
os.add_dll_directory('C:/Program Files/AMD/ROCm/6.4/bin')
os.environ['STRATA_GLM_GU22_TUNED']='1'
d=c.CDLL(str(R/'build/maya-hhelper-lowlatency.dll'))
P=c.c_void_p;I=c.c_int;Z=c.c_size_t;F=c.c_float
for name,args in {'open':[I,I,I,c.POINTER(P),P,Z],'reserve':[P,I,Z,P,Z],'upload':[P,I,P,Z,Z,P,Z],'compute':[P,I,I,Z,F,I,P,P,P,P,P,Z],'close':[P]}.items():
 fn=getattr(d,'glmh_'+name);fn.argtypes=args;fn.restype=None if name=='close' else I
pack=Path('E:/Maya-glm-data/models/Maya-M/pack');rows=[l.split() for l in (pack/'native_experts.txt').read_text().splitlines() if l and not l.startswith('#')]
l=next(l for l in rows if l[0]=='28');il,gu,dn=map(int,l[:3]);size=int(l[4]);off=list(map(int,l[5:8]));gbytes=4096*2048//256*82
shards=sorted(pack.parent.glob('*.gguf'));names=l[8:];blobs=[]
for e in (0,1):
 data=bytearray()
 for j in range(3):
  n=gbytes if j<2 else size-2*gbytes
  path=pack.parent/names[0 if len(names)==1 else j] if names else shards[0]
  with path.open('rb') as f:f.seek(off[j]+e*n);data+=f.read(n)
 assert len(data)==size;blobs.append(c.create_string_buffer(bytes(data)))
x=(F*4096)(*[math.sin(i*.13) for i in range(4096)]);y=(F*4096)();keys=(I*2)(il*288,il*288+1);w=(F*2)(.25,.75);err=c.create_string_buffer(512)
ref={};results=[]
for zc,spin in ((0,0),(0,1),(1,0),(1,1)):
 os.environ['STRATA_GLM_HIP_ZEROCOPY']=str(zc);os.environ['STRATA_GLM_HIP_SPIN_WAIT']=str(spin)
 h=P()
 def ok(code):
  if code:raise RuntimeError(err.value.decode())
 ok(d.glmh_open(0,4096,2048,c.byref(h),err,512));ok(d.glmh_reserve(h,45*288,size*2,err,512))
 for i,b in enumerate(blobs):ok(d.glmh_upload(h,keys[i],b,size,size,err,512))
 for i in range(10):
  x[0]=i*.01;w[0]=.25 if i%2 else .4;w[1]=1-w[0];n=1+i%2
  ok(d.glmh_compute(h,gu,dn,2*gbytes,10.,n,keys,w,x,y,err,512))
  out=bytes(y)
  if (zc,spin)==(0,0):ref[i]=out
  assert out==ref[i],(zc,spin,i,'output differs')
 # Restore fixed input and run enough calls to amortize Python overhead; discard graph warmup.
 samples=[]
 for block in range(4):
  t=time.perf_counter()
  for j in range(500):ok(d.glmh_compute(h,gu,dn,2*gbytes,10.,2,keys,w,x,y,err,512))
  samples.append((time.perf_counter()-t)*1e6/500)
 row={'zero_copy':zc,'spin_wait':spin,'host_us_per_call':sum(samples)/len(samples),'samples_us':samples,'identical_cases':10};results.append(row);print(json.dumps(row),flush=True);d.glmh_close(h)
(O/'hip-call-latency.json').write_text(json.dumps(results,indent=2))
