// src/ds4/ds4_helper.h - the expert helper's C ABI.
//
// The CUDA engine (maya-ds4 on the primary card) drives a second, HIP-built DLL that owns routed experts on another
// vendor's card and computes them per decode round.  The two runtimes coexist in one process (spike-hippeer); every
// expert's blob is uploaded once at load, and a round ships the tokens' hidden states, the slots they routed to, and
// their weights through host memory - the helper answers with its part of every token's output (the caller adds its
// own device part and its CPU lane's).
#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(DS4H_BUILD)
#define DS4H_API __declspec(dllexport)
#else
#define DS4H_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4h ds4h;

typedef struct ds4h_layer {
    int layer;             // the model's layer (selects the layout given to ds4h_layout)
    int n_tok;             // the round's tokens (a DSpark verify window; at most 16)
    int k;                 // routed experts a token (6 for DeepSeek-V4.1; at most 8)
    float limit;           // the model's clamped SwiGLU limit
    const float* x;        // [n_tok][n_embd] hidden states, F32
    const int32_t* slot;   // [n_tok][k] the helper's slots; -1: not the helper's (its part is skipped)
    const float* w;        // [n_tok][k] the routing weights
    float* out;            // [n_tok][n_embd] the helper's weighted sum (the caller adds its own)
} ds4h_layer;

typedef struct ds4h_stats {
    int64_t rounds;
    int64_t experts;       // expert evaluations computed
    int64_t uploads;
    double last_round_ms;
    double total_round_ms;
} ds4h_stats;

// Opens the helper on HIP device `device`.  0 on success; otherwise `err` holds the message.
DS4H_API int ds4h_open(int device, int n_embd, int n_ff, ds4h** out, char* err, size_t cap);
// The experts' blob layout for one layer: the bytes of a strata::kernels::exl3::ExpertLayout (fmt 2, GGUF) and the
// blob's stride in bytes (each layer may use its own GGUF types).
DS4H_API int ds4h_layout(ds4h* h, int layer, const void* layout_bytes, size_t bytes, size_t stride, char* err, size_t cap);
// Allocates `slots` expert blobs of `stride` bytes each in the helper's VRAM (one arena).
DS4H_API int ds4h_reserve(ds4h* h, int slots, size_t stride, char* err, size_t cap);
// Copies one assembled expert blob (the primary's pinned-pool format) into slot `slot`, laid out as `layer`'s.
DS4H_API int ds4h_upload(ds4h* h, int slot, int layer, const void* blob, size_t bytes, char* err, size_t cap);
// Runs one decode round: every layer in order on the helper's stream, one device sync at the end.
DS4H_API int ds4h_round(ds4h* h, const ds4h_layer* layers, int n_layers, char* err, size_t cap);
// The same, split: submit enqueues the layers (nothing waited for), wait takes the device sync - the caller's own
// work runs between them
DS4H_API int ds4h_submit(ds4h* h, const ds4h_layer* layers, int n_layers, char* err, size_t cap);
DS4H_API int ds4h_wait(ds4h* h, char* err, size_t cap);
DS4H_API void ds4h_get_stats(ds4h* h, ds4h_stats* out);
DS4H_API void ds4h_close(ds4h* h);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
DS4H_API int ds4h_version(void);
#ifdef __cplusplus
}
#endif
