// src/mm3/mm3_helper.h - the expert helper's C ABI.
//
// The CUDA engine (maya-mm3 on the primary card) drives a second, HIP-built DLL that owns routed experts on another
// vendor's card and computes them per decode round.  The two runtimes coexist in one process (spike-hippeer); every
// expert's blob is uploaded once at load, and a round ships the tokens' hidden states, the slots they routed to, and
// their weights through host memory - the helper answers with its part of every token's output (the caller adds its
// own device part and its CPU lane's).
#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(MM3H_BUILD)
#define MM3H_API __declspec(dllexport)
#else
#define MM3H_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mm3h mm3h;

typedef struct mm3h_layer {
    int layer;             // the model's layer (selects the layout given to mm3h_layout)
    int n_tok;             // the round's tokens (a DSpark verify window; at most 16)
    int k;                 // routed experts a token (4 for MiniMax-M3.1; at most 8)
    float limit;           // the model's swiglu_limit (7.0 for MiniMax-M3.1)
    const float* x;        // [n_tok][n_embd] hidden states, F32
    const int32_t* slot;   // [n_tok][k] the helper's slots; -1: not the helper's (its part is skipped)
    const float* w;        // [n_tok][k] the routing weights
    float* out;            // [n_tok][n_embd] the helper's weighted sum (the caller adds its own)
} mm3h_layer;

typedef struct mm3h_stats {
    int64_t rounds;
    int64_t experts;       // expert evaluations computed
    int64_t uploads;
    double last_round_ms;
    double total_round_ms;
} mm3h_stats;

// Opens the helper on HIP device `device`.  0 on success; otherwise `err` holds the message.
MM3H_API int mm3h_open(int device, int n_embd, int n_ff, mm3h** out, char* err, size_t cap);
// The experts' blob layout for one layer: the bytes of a strata::mm3::exl3::ExpertLayout (fmt 4, GGUF) and the
// blob's stride in bytes (the layer's experts may be of different types: MiniMax's pack mixes four).
MM3H_API int mm3h_layout(mm3h* h, int layer, const void* layout_bytes, size_t bytes, size_t stride, char* err, size_t cap);
// Allocates `slots` expert blobs of `stride` bytes each in the helper's VRAM (one arena).
MM3H_API int mm3h_reserve(mm3h* h, int slots, size_t stride, char* err, size_t cap);
// Copies one assembled expert blob (the primary's pinned-pool format) into slot `slot`, laid out as `layer`'s.
MM3H_API int mm3h_upload(mm3h* h, int slot, int layer, const void* blob, size_t bytes, char* err, size_t cap);
// Runs one decode round: every layer in order on the helper's stream, one device sync at the end.
MM3H_API int mm3h_round(mm3h* h, const mm3h_layer* layers, int n_layers, char* err, size_t cap);
// The same, split: submit enqueues the layers (nothing waited for), wait takes the device sync - the caller's own
// work runs between them
MM3H_API int mm3h_submit(mm3h* h, const mm3h_layer* layers, int n_layers, char* err, size_t cap);
MM3H_API int mm3h_wait(mm3h* h, char* err, size_t cap);
MM3H_API void mm3h_get_stats(mm3h* h, mm3h_stats* out);
MM3H_API void mm3h_close(mm3h* h);

#ifdef __cplusplus
}
#endif
