// Real GGUF expert parity, including the Q8 boundary between gate/up and down.
// Usage: glm_cross_helper_test <native pack> <shared HIP DLL>
#include "strata/core/glm_hip_helper.h"
#include "strata/kernels/glm_fast.hpp"
#include <cuda_runtime.h>
#define NOMINMAX
#include <windows.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
namespace gf=strata::kernels::glmf;namespace fs=std::filesystem;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Layer{int il,gu,down;size_t bytes;uint64_t off[3];fs::path shard[3];};
static double rel(const std::vector<float>& a,const std::vector<float>& b){double d=0,r=0;for(size_t i=0;i<a.size();++i){d+=double(a[i]-b[i])*(a[i]-b[i]);r+=double(b[i])*b[i];}return std::sqrt(d/std::max(r,1e-30));}
int main(int argc,char** argv)try{
 if(argc!=3)return 2;fs::path pack=argv[1],root=pack.parent_path(),first;
 for(auto& e:fs::directory_iterator(root))if(e.path().extension()==".gguf"&&(first.empty()||e.path()<first))first=e.path();
 if(first.empty())throw std::runtime_error("GGUF shards missing beside pack");
 std::ifstream manifest(pack/"native_experts.txt");std::vector<Layer> layers;std::string line;
 while(std::getline(manifest,line)){if(line.empty()||line[0]=='#')continue;Layer l;uint64_t logical;std::istringstream row(line);row>>l.il>>l.gu>>l.down>>logical>>l.bytes>>l.off[0]>>l.off[1]>>l.off[2];std::vector<std::string> names;std::string n;while(row>>n)names.push_back(n);for(int j=0;j<3;++j)l.shard[j]=names.empty()?first:root/names[names.size()==1?0:j];layers.push_back(l);}
 AddDllDirectory(L"C:/Program Files/AMD/ROCm/6.4/bin");HMODULE dll=LoadLibraryExA(argv[2],nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);if(!dll)throw std::runtime_error("HIP DLL load failed");
 #define SYM(n) auto n=(decltype(&glmh_##n))GetProcAddress(dll,"glmh_" #n);if(!n)throw std::runtime_error("missing " #n)
 SYM(open);SYM(reserve);SYM(upload);SYM(compute);SYM(close);SYM(read_q8);SYM(read_bytes);
 constexpr int E=4096,FF=2048,K=2;char err[512]{};glmh* h=nullptr;
 auto hc=[&](int code){if(code)throw std::runtime_error(err);};hc(open(0,E,FF,&h,err,sizeof err));
 glmh* invalid=nullptr;int rejects=0;
 if(open(0,33,FF,&invalid,err,sizeof err)==0)throw std::runtime_error("accepted invalid geometry");++rejects;
 const size_t maxblob=8978432;hc(reserve(h,288*45,maxblob*288,err,sizeof err));
 std::vector<float>x(E),actual(E),reference(E),downref(E);uint32_t rng=47;for(auto& v:x){rng=rng*1664525u+1013904223u;v=float(int((rng>>8)%101)-50)*.01f;}
 void *blobs=nullptr,*xq=nullptr,*hq=nullptr;float *dx=nullptr,*dout=nullptr;gf::MoeDev plan;
 ck(cudaMalloc(&blobs,K*maxblob));ck(cudaMalloc(&xq,E/32*36));ck(cudaMalloc(&hq,K*FF/32*36));ck(cudaMalloc((void**)&dx,E*4));ck(cudaMalloc((void**)&dout,E*4));
 ck(cudaMalloc((void**)&plan.plan_ptr,K*8));ck(cudaMalloc((void**)&plan.plan_w,K*4));ck(cudaMalloc((void**)&plan.cpu_flag,4));ck(cudaMemset(plan.cpu_flag,0,4));ck(cudaMemcpy(dx,x.data(),E*4,cudaMemcpyHostToDevice));gf::quantize_q8_1(dx,xq,E,nullptr);
 std::vector<unsigned char> cx(E/32*36),hx(cx.size()),ch(K*FF/32*36),hh(ch.size());ck(cudaMemcpy(cx.data(),xq,cx.size(),cudaMemcpyDeviceToHost));
 int unowned[2]={0,1};float routew[2]={.5f,.5f};
 if(compute(h,22,18,5373952,10.f,0,unowned,routew,x.data(),actual.data(),err,sizeof err)==0)throw std::runtime_error("accepted invalid route width");++rejects;
 if(compute(h,22,18,5373952,10.f,2,unowned,routew,x.data(),actual.data(),err,sizeof err)==0)throw std::runtime_error("accepted unowned expert");++rejects;
 if(read_q8(h,9,hx.data(),hh.data(),err,sizeof err)==0)throw std::runtime_error("accepted oversized readback");++rejects;
 if(read_bytes(h,0,0,16,hh.data(),err,sizeof err)==0)throw std::runtime_error("accepted unowned readback");++rejects;
 if(read_bytes(h,0,0,0,hh.data(),err,sizeof err)==0)throw std::runtime_error("accepted empty readback");++rejects;
 size_t xdiff=0,qdiff=0;int qdelta=0,scaleulp=0,cases=0;double worst=0,worstdown=0;size_t blocks=0;
 for(const auto& l:layers){if(l.il!=28&&l.il!=3&&l.il!=41)continue; // all layer 28 experts; representative other down format
  std::ifstream files[3];for(int j=0;j<3;++j)files[j].open(l.shard[j],std::ios::binary);
  size_t gu=E*FF/256*82,dn=l.bytes-2*gu;std::vector<unsigned char> blob(l.bytes);
  int count=l.il==28?288:4;
  for(int e=0;e<count;++e){for(int j=0;j<3;++j){size_t n=j==2?dn:gu;files[j].seekg(l.off[j]+uint64_t(e)*n);files[j].read((char*)blob.data()+(j==0?0:j==1?gu:2*gu),n);if(!files[j])throw std::runtime_error("short expert read");}hc(upload(h,l.il*288+e,blob.data(),l.bytes,l.bytes,err,sizeof err));
   std::vector<unsigned char> back(l.bytes);hc(read_bytes(h,l.il*288+e,0,l.bytes,back.data(),err,sizeof err));if(back!=blob)throw std::runtime_error("GPU backing expert bytes differ");
   hc(read_bytes(h,l.il*288+e,4096,4096,back.data(),err,sizeof err));if(std::memcmp(back.data(),blob.data()+4096,4096))throw std::runtime_error("partial GPU backing read differs");
   ck(cudaMemcpy((char*)blobs+(e%K)*maxblob,blob.data(),l.bytes,cudaMemcpyHostToDevice));if(e%K!=K-1)continue;
   int keys[K]={l.il*288+e-1,l.il*288+e};float w[K]={e%4==1?.25f:1.f/3,e%4==1?.75f:2.f/3};unsigned long long ptr[K]={(unsigned long long)blobs,(unsigned long long)((char*)blobs+maxblob)};
   x[0]=float(e-101)/128.f;x[1]=float(l.il)/64.f;ck(cudaMemcpy(dx,x.data(),E*4,cudaMemcpyHostToDevice));gf::quantize_q8_1(dx,xq,E,nullptr);ck(cudaMemcpy(cx.data(),xq,cx.size(),cudaMemcpyDeviceToHost));
   ck(cudaMemcpy(plan.plan_ptr,ptr,sizeof ptr,cudaMemcpyHostToDevice));ck(cudaMemcpy(plan.plan_w,w,sizeof w,cudaMemcpyHostToDevice));
   gf::moe_gate_up(l.gu,plan,K,E,FF,10.f,xq,hq,nullptr,0,nullptr,0,nullptr,nullptr);gf::moe_down(l.down,plan,K,E,FF,2*gu,hq,nullptr,dout,nullptr);
   ck(cudaMemcpy(reference.data(),dout,E*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(ch.data(),hq,ch.size(),cudaMemcpyDeviceToHost));
   hc(compute(h,l.gu,l.down,2*gu,10.f,K,keys,w,x.data(),actual.data(),err,sizeof err));hc(read_q8(h,K,hx.data(),hh.data(),err,sizeof err));
   for(size_t i=0;i<hx.size();++i)xdiff+=hx[i]!=cx[i];
   for(size_t b=0;b<hh.size();b+=36){uint16_t cd,hd;std::memcpy(&cd,ch.data()+b,2);std::memcpy(&hd,hh.data()+b,2);scaleulp=std::max(scaleulp,std::abs(int(cd)-int(hd)));for(int j=4;j<36;++j){int delta=std::abs(int((signed char)ch[b+j])-int((signed char)hh[b+j]));qdiff+=delta!=0;qdelta=std::max(qdelta,delta);}++blocks;}
   double er=rel(actual,reference);worst=std::max(worst,er);
   ck(cudaMemcpy(hq,hh.data(),hh.size(),cudaMemcpyHostToDevice));gf::moe_down(l.down,plan,K,E,FF,2*gu,hq,nullptr,dout,nullptr);ck(cudaMemcpy(downref.data(),dout,E*4,cudaMemcpyDeviceToHost));double ed=rel(actual,downref);worstdown=std::max(worstdown,ed);++cases;
   if(er>1e-5)std::printf("boundary layer=%d experts=%d,%d rel_L2=%.9g same_Hq_down=%.9g\n",l.il,e-1,e,er,ed);
  }
 }
 std::printf("RESULT cases=%d input_q8_byte_differences=%zu hq_q_differences=%zu hq_q_max_delta=%d hq_scale_max_ulp=%d blocks=%zu output_max_rel_L2=%.9g same_Hq_down_max_rel_L2=%.9g\n",cases,xdiff,qdiff,qdelta,scaleulp,blocks,worst,worstdown);
 std::printf("ABI rejected %d invalid geometry/route/key/readback cases\n",rejects);
 close(h);return rejects==6&&cases>0&&xdiff==0&&qdelta<=1&&scaleulp<=1&&worstdown<1e-5?0:1;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
