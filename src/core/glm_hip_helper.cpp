// HIP-owned GGUF GLM experts. Only host buffers and key IDs cross this ABI.
#include "strata/core/glm_hip_helper.h"
#include "strata/kernels/glm_fast.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
namespace gf=strata::kernels::glmf;
struct glmh {
 int device,dim,ff; cudaStream_t stream=nullptr;
 unsigned char* pool=nullptr; size_t capacity=0,used=0;
 struct Slot {size_t offset=0,bytes=0;};std::vector<Slot> slots;
 void* arena=nullptr; void* pin=nullptr;void* transfer_pin=nullptr;size_t transfer_bytes=0;
 float *x=nullptr,*out=nullptr,*hx=nullptr,*hout=nullptr;void *xq=nullptr,*hq=nullptr;
 struct Graph {int gu,down,n;size_t off;float limit;cudaGraph_t graph;cudaGraphExec_t exec;};
 std::vector<Graph> graphs;
 bool replay=true,zero_copy=false,spin_wait=false;
 unsigned long long* hp=nullptr;float* hw=nullptr;gf::MoeDev plan;
};
static int fail(char* err,size_t cap,const char* text){if(err&&cap)std::snprintf(err,cap,"%s",text);return 1;}
static bool ck(cudaError_t e,char* err,size_t cap){if(e==cudaSuccess)return true;fail(err,cap,cudaGetErrorString(e));return false;}
int glmh_version(){return 1;}
int glmh_open(int device,int dim,int ff,glmh** out,char* err,size_t cap){
 if(!out||dim<=0||dim>4096||dim%32||ff<=0||ff%32)return fail(err,cap,"GLM HIP: unsupported geometry");
 auto* h=new glmh();if(const char* v=std::getenv("STRATA_GLM_HIP_GRAPH"))h->replay=std::strcmp(v,"0")!=0;
 if(const char* v=std::getenv("STRATA_GLM_HIP_ZEROCOPY"))h->zero_copy=std::strcmp(v,"0")!=0;
 if(const char* v=std::getenv("STRATA_GLM_HIP_SPIN_WAIT"))h->spin_wait=std::strcmp(v,"0")!=0;
 h->device=device;h->dim=dim;h->ff=ff;
 if(!ck(cudaSetDevice(device),err,cap)||!ck(cudaStreamCreateWithFlags(&h->stream,cudaStreamNonBlocking),err,cap)){delete h;return 1;}
 size_t xb=(size_t)dim*4,aq=(size_t)dim/32*36,hq=(size_t)8*ff/32*36;
 if(!ck(cudaMalloc(&h->arena,2*xb+aq+hq+2048),err,cap)||!ck(cudaHostAlloc(&h->pin,2*xb+256,h->zero_copy?cudaHostAllocMapped:cudaHostAllocDefault),err,cap)){glmh_close(h);return 1;}
 auto* at=(unsigned char*)h->arena;
 auto take=[&](size_t n){void* p=at;at+=(n+63)&~size_t(63);return p;};
 h->x=(float*)take(xb);h->out=(float*)take(xb);h->xq=take(aq);h->hq=take(hq);
 h->plan.plan_ptr=(unsigned long long*)take(64);h->plan.plan_w=(float*)take(32);h->plan.cpu_flag=(int*)take(64);
 h->hx=(float*)h->pin;h->hout=h->hx+dim;h->hp=(unsigned long long*)(h->hout+dim);h->hw=(float*)(h->hp+8);
 if(h->zero_copy){
  // Runtime-owned coherent host buffers; no CUDA-owned address is passed to HIP.
  unsigned char* mapped=nullptr;
  if(!ck(cudaHostGetDevicePointer((void**)&mapped,h->pin,0),err,cap)){glmh_close(h);return 1;}
  h->x=(float*)mapped;h->out=(float*)(mapped+xb);
  h->plan.plan_ptr=(unsigned long long*)(mapped+2*xb);h->plan.plan_w=(float*)(mapped+2*xb+64);
 }
 if(!ck(cudaMemset(h->plan.cpu_flag,0,64),err,cap)){glmh_close(h);return 1;}
 *out=h;return 0;
}
int glmh_reserve(glmh* h,int keys,size_t bytes,char* err,size_t cap){
 if(!h||keys<=0||!bytes||h->pool)return fail(err,cap,"GLM HIP: invalid reserve");
 if(!ck(cudaSetDevice(h->device),err,cap)||!ck(cudaMalloc((void**)&h->pool,bytes),err,cap))return 1;
 h->capacity=bytes;h->slots.resize(keys);return 0;
}
int glmh_upload(glmh* h,int key,const void* blob,size_t bytes,size_t stride,char* err,size_t cap){
 if(!h||!h->pool||!blob||key<0||(size_t)key>=h->slots.size()||!bytes||bytes>stride||h->used>h->capacity||stride>h->capacity-h->used||h->slots[key].bytes)return fail(err,cap,"GLM HIP: invalid upload");
 if(!ck(cudaSetDevice(h->device),err,cap)||!ck(cudaMemcpy(h->pool+h->used,blob,bytes,cudaMemcpyHostToDevice),err,cap))return 1;
 if(std::getenv("STRATA_GLM_HELPER_CHECK")){
  std::vector<unsigned char> back(bytes);if(!ck(cudaMemcpy(back.data(),h->pool+h->used,bytes,cudaMemcpyDeviceToHost),err,cap)||std::memcmp(back.data(),blob,bytes))return fail(err,cap,"GLM HIP: byte readback mismatch");
 }
 h->slots[key]={h->used,bytes};h->used+=stride;return 0;
}
int glmh_compute(glmh* h,int gu,int down,size_t down_off,float limit,int n,const int* keys,const float* weights,const float* x,float* out,char* err,size_t cap){
 if(!h||!h->pool||n<1||n>8||!keys||!weights||!x||!out||!gf::moe_supported(gu)||!gf::moe_supported(down))return fail(err,cap,"GLM HIP: invalid route");
 for(int i=0;i<n;++i){if(keys[i]<0||(size_t)keys[i]>=h->slots.size()||!h->slots[keys[i]].bytes||down_off>=h->slots[keys[i]].bytes)return fail(err,cap,"GLM HIP: unowned key");h->hp[i]=(unsigned long long)(h->pool+h->slots[keys[i]].offset);h->hw[i]=weights[i];}
 size_t bytes=(size_t)h->dim*4;std::memcpy(h->hx,x,bytes);
 if(!ck(cudaSetDevice(h->device),err,cap))return 1;
 auto enqueue=[&](){
  if(!h->zero_copy && (!ck(cudaMemcpyAsync(h->x,h->hx,bytes,cudaMemcpyHostToDevice,h->stream),err,cap)||!ck(cudaMemcpyAsync(h->plan.plan_ptr,h->hp,n*8,cudaMemcpyHostToDevice,h->stream),err,cap)||!ck(cudaMemcpyAsync(h->plan.plan_w,h->hw,n*4,cudaMemcpyHostToDevice,h->stream),err,cap)))return false;
  gf::quantize_q8_1(h->x,h->xq,h->dim,h->stream);
  gf::moe_gate_up(gu,h->plan,n,h->dim,h->ff,limit,h->xq,h->hq,nullptr,0,nullptr,0,nullptr,h->stream);
  gf::moe_down(down,h->plan,n,h->dim,h->ff,down_off,h->hq,nullptr,h->out,h->stream);
  return ck(cudaGetLastError(),err,cap)&&(h->zero_copy || ck(cudaMemcpyAsync(h->hout,h->out,bytes,cudaMemcpyDeviceToHost,h->stream),err,cap));
 };
 if(h->replay){
  glmh::Graph* found=nullptr;for(auto& g:h->graphs)if(g.gu==gu&&g.down==down&&g.n==n&&g.off==down_off&&g.limit==limit){found=&g;break;}
  if(!found){
   if(!ck(cudaStreamBeginCapture(h->stream,cudaStreamCaptureModeThreadLocal),err,cap))return 1;
   bool ok=enqueue();cudaGraph_t graph=nullptr;cudaError_t end=cudaStreamEndCapture(h->stream,&graph);
   if(!ok||!ck(end,err,cap)){if(graph)cudaGraphDestroy(graph);return 1;}
   cudaGraphExec_t exec=nullptr;if(!ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0),err,cap)){cudaGraphDestroy(graph);return 1;}
   h->graphs.push_back({gu,down,n,down_off,limit,graph,exec});found=&h->graphs.back();
   if(h->graphs.size()==1)std::fprintf(stderr,"glm HIP helper: graph replay enabled (host inputs and routing updated each call)\n");
  }
  if(!ck(cudaGraphLaunch(found->exec,h->stream),err,cap))return 1;
 }else if(!enqueue())return 1;
 if(h->spin_wait){
  cudaError_t state;do{state=cudaStreamQuery(h->stream);}while(state==cudaErrorNotReady);
  if(!ck(state,err,cap))return 1;
 }else if(!ck(cudaStreamSynchronize(h->stream),err,cap))return 1;
 for(int i=0;i<h->dim;++i)if(!std::isfinite(h->hout[i]))return fail(err,cap,"GLM HIP: nonfinite result");
 std::memcpy(out,h->hout,bytes);return 0;
}
int glmh_read_bytes(glmh* h,int key,size_t off,size_t bytes,void* out,char* err,size_t cap){
 if(!h||!out||!bytes||key<0||(size_t)key>=h->slots.size()||!h->slots[key].bytes||off>h->slots[key].bytes||bytes>h->slots[key].bytes-off)return fail(err,cap,"GLM HIP: invalid expert readback");
 if(!ck(cudaSetDevice(h->device),err,cap))return 1;
 // The caller may be CUDA-pinned: HIP owns its DMA endpoint, then copies on the host.
 if(bytes>h->transfer_bytes){if(h->transfer_pin)cudaFreeHost(h->transfer_pin);h->transfer_pin=nullptr;h->transfer_bytes=0;
  size_t n=(bytes+4095)&~size_t(4095);if(!ck(cudaHostAlloc(&h->transfer_pin,n,cudaHostAllocDefault),err,cap))return 1;h->transfer_bytes=n;}
 if(!ck(cudaMemcpyAsync(h->transfer_pin,h->pool+h->slots[key].offset+off,bytes,cudaMemcpyDeviceToHost,h->stream),err,cap)||!ck(cudaStreamSynchronize(h->stream),err,cap))return 1;
 std::memcpy(out,h->transfer_pin,bytes);return 0;
}
int glmh_read_q8(glmh* h,int n,void* xq,void* hq,char* err,size_t cap){
 if(!h||n<1||n>8||!xq||!hq)return fail(err,cap,"GLM HIP: invalid diagnostic readback");
 if(!ck(cudaSetDevice(h->device),err,cap)||!ck(cudaStreamSynchronize(h->stream),err,cap)||
    !ck(cudaMemcpy(xq,h->xq,(size_t)h->dim/32*36,cudaMemcpyDeviceToHost),err,cap)||
    !ck(cudaMemcpy(hq,h->hq,(size_t)n*h->ff/32*36,cudaMemcpyDeviceToHost),err,cap))return 1;
 return 0;
}
void glmh_close(glmh* h){if(!h)return;cudaSetDevice(h->device);if(h->stream)cudaStreamSynchronize(h->stream);for(auto& g:h->graphs){cudaGraphExecDestroy(g.exec);cudaGraphDestroy(g.graph);}if(h->pool)cudaFree(h->pool);if(h->arena)cudaFree(h->arena);if(h->pin)cudaFreeHost(h->pin);if(h->transfer_pin)cudaFreeHost(h->transfer_pin);if(h->stream)cudaStreamDestroy(h->stream);delete h;}
