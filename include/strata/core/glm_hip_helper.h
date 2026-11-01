#pragma once
#include <stddef.h>
#ifdef _WIN32
#ifdef GLMH_BUILD
#define GLMH_API __declspec(dllexport)
#else
#define GLMH_API
#endif
#else
#define GLMH_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct glmh glmh;
GLMH_API int glmh_version(void);
GLMH_API int glmh_open(int device,int dim,int ff,glmh** out,char* err,size_t cap);
GLMH_API int glmh_reserve(glmh*,int keys,size_t bytes,char* err,size_t cap);
GLMH_API int glmh_upload(glmh*,int key,const void* blob,size_t bytes,size_t stride,char* err,size_t cap);
GLMH_API int glmh_compute(glmh*,int gu,int down,size_t down_off,float limit,int n,const int* keys,const float* weights,const float* x,float* out,char* err,size_t cap);
// Diagnostic readback after compute, used by cross-runtime numerical tests.
GLMH_API int glmh_read_bytes(glmh*,int key,size_t off,size_t bytes,void* out,char* err,size_t cap);
GLMH_API int glmh_read_q8(glmh*,int n,void* xq,void* hq,char* err,size_t cap);
GLMH_API void glmh_close(glmh*);
#ifdef __cplusplus
}
#endif
