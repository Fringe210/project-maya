// src/ds4/ds4_helper.cpp - the expert helper's side of ds4_helper.h: a HIP build's own kernels (x3::win_experts, the
// same code the primary's device path runs) over experts uploaded once, driven a decode round at a time.
//
// Every layer has its own pinned staging (the round's activations, plan and grouping arrays): the copies are enqueued
// on one stream and the next layer's staging never overwrites a buffer a pending copy still reads.  One device sync
// ends a round.
#include "ds4_helper.h"

#include "ds4_gguf.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace x3 = strata::kernels::exl3;
namespace gg = strata::ds4::gguf;

namespace {

constexpr int kMaxTok = 16;   // a DSpark verify window's tokens at most

int fail(char* err, size_t cap, const char* what) {
    if (err && cap) std::snprintf(err, cap, "%s", what);
    return 1;
}

bool ck(cudaError_t e, const char* what, char* err, size_t cap) {
    if (e == cudaSuccess) return true;
    if (err && cap) std::snprintf(err, cap, "%s: %s", what, cudaGetErrorString(e));
    return false;
}

}  // namespace

struct ds4h {
    int dev = -1;
    cudaStream_t s = nullptr;
    int n_embd = 0, n_ff = 0;
    uint8_t* arena = nullptr;
    int slots = 0;
    size_t stride = 0;
    std::vector<x3::ExpertLayout> lay;
    std::vector<char> have_lay;
    std::vector<float*> pin_x;
    std::vector<float*> pin_out;
    std::vector<uint8_t*> pin_meta;
    struct Output { float* dst; const float* src; size_t bytes; };
    std::vector<Output> outputs;
    bool pending = false;
    int cap_tok = 0;
    float* d_x = nullptr;
    float* d_out = nullptr;
    float* d_h = nullptr;
    float* d_part = nullptr;
    unsigned long long* d_pp = nullptr;
    float* d_pw = nullptr;
    unsigned long long* d_gp = nullptr;
    int* d_gs = nullptr;
    int* d_ge = nullptr;
    int* d_ng = nullptr;
    ds4h_stats st{};
    std::chrono::steady_clock::time_point t0;   // (a submit's start: wait's round time)
};

static size_t meta_bytes() {
    const size_t T8 = (size_t) kMaxTok * 8;
    return T8 * 8 + T8 * 4 + T8 * 8 + (T8 + 1) * 4 + T8 * 4 + 4;
}

static void free_scratch(ds4h* h) {
    if (h->d_x) cudaFree(h->d_x);
    if (h->d_out) cudaFree(h->d_out);
    if (h->d_h) cudaFree(h->d_h);
    if (h->d_part) cudaFree(h->d_part);
    if (h->d_pp) cudaFree(h->d_pp);
    if (h->d_pw) cudaFree(h->d_pw);
    if (h->d_gp) cudaFree(h->d_gp);
    if (h->d_gs) cudaFree(h->d_gs);
    if (h->d_ge) cudaFree(h->d_ge);
    if (h->d_ng) cudaFree(h->d_ng);
    h->d_x = h->d_out = h->d_h = h->d_part = nullptr;
    h->d_pp = h->d_gp = nullptr;
    h->d_pw = nullptr;
    h->d_gs = h->d_ge = h->d_ng = nullptr;
    h->cap_tok = 0;
}

static bool ensure_tok(ds4h* h, int t, char* err, size_t cap) {
    if (t <= h->cap_tok) return true;
    free_scratch(h);
    const int T8 = t * 8;
    const bool ok =
        ck(cudaMalloc((void**) &h->d_x, (size_t) t * h->n_embd * 4), "helper x", err, cap) &&
        ck(cudaMalloc((void**) &h->d_out, (size_t) t * h->n_embd * 4), "helper out", err, cap) &&
        ck(cudaMalloc((void**) &h->d_h, (size_t) T8 * h->n_ff * 4), "helper h", err, cap) &&
        ck(cudaMalloc((void**) &h->d_part, (size_t) T8 * h->n_embd * 4), "helper part", err, cap) &&
        ck(cudaMalloc((void**) &h->d_pp, (size_t) T8 * 8), "helper plan", err, cap) &&
        ck(cudaMalloc((void**) &h->d_pw, (size_t) T8 * 4), "helper plan w", err, cap) &&
        ck(cudaMalloc((void**) &h->d_gp, (size_t) T8 * 8), "helper groups", err, cap) &&
        ck(cudaMalloc((void**) &h->d_gs, (size_t) (T8 + 1) * 4), "helper group starts", err, cap) &&
        ck(cudaMalloc((void**) &h->d_ge, (size_t) T8 * 4), "helper group entries", err, cap) &&
        ck(cudaMalloc((void**) &h->d_ng, 4), "helper group count", err, cap);
    if (!ok) {
        free_scratch(h);
        return false;
    }
    h->cap_tok = t;
    return true;
}

static bool ensure_stage(ds4h* h, int layer, char* err, size_t cap) {
    if ((size_t) layer < h->pin_x.size() && h->pin_x[(size_t) layer]) return true;
    if ((size_t) layer >= h->pin_x.size()) {
        h->pin_x.resize((size_t) layer + 1, nullptr);
        h->pin_out.resize((size_t) layer + 1, nullptr);
        h->pin_meta.resize((size_t) layer + 1, nullptr);
    }
    float *px = nullptr, *po = nullptr;
    uint8_t* pm = nullptr;
    if (!ck(cudaHostAlloc((void**) &px, (size_t) kMaxTok * h->n_embd * 4, cudaHostAllocDefault), "helper pinned x",
            err, cap) ||
        !ck(cudaHostAlloc((void**) &pm, meta_bytes(), cudaHostAllocDefault), "helper pinned meta", err, cap) ||
        !ck(cudaHostAlloc((void**) &po, (size_t) kMaxTok * h->n_embd * 4, cudaHostAllocDefault), "helper pinned out",
            err, cap)) {
        if (px) cudaFreeHost(px);
        if (pm) cudaFreeHost(pm);
        return false;
    }
    h->pin_x[(size_t) layer] = px;
    h->pin_out[(size_t) layer] = po;
    h->pin_meta[(size_t) layer] = pm;
    return true;
}

int ds4h_open(int device, int n_embd, int n_ff, ds4h** out, char* err, size_t cap) {
    if (!out || n_embd <= 0 || n_ff <= 0) return fail(err, cap, "helper: bad geometry");
    auto* h = new ds4h();
    h->dev = device;
    h->n_embd = n_embd;
    h->n_ff = n_ff;
    if (!ck(cudaSetDevice(device), "helper: cudaSetDevice", err, cap) ||
        !ck(cudaStreamCreateWithFlags(&h->s, cudaStreamNonBlocking), "helper: stream", err, cap)) {
        delete h;
        return 1;
    }
    if (!gg::prepare(h->s)) {
        cudaStreamDestroy(h->s);
        delete h;
        return fail(err, cap, "helper: prepare failed");
    }
    if (!ensure_tok(h, kMaxTok, err, cap)) { ds4h_close(h); return 1; }
    *out = h;
    return 0;
}

int ds4h_layout(ds4h* h, int layer, const void* bytes, size_t nbytes, size_t stride, char* err, size_t cap) {
    if (!h || !bytes || layer < 0 || stride == 0) return fail(err, cap, "helper: bad layout");
    if (h->pending) return fail(err, cap, "helper: wait for the pending round first");
    if (nbytes != sizeof(x3::ExpertLayout))
        return fail(err, cap, "helper: the layout's size differs from this build's ExpertLayout");
    if ((int) h->lay.size() <= layer) {
        h->lay.resize((size_t) layer + 1);
        h->have_lay.resize((size_t) layer + 1, 0);
    }
    std::memcpy(&h->lay[(size_t) layer], bytes, sizeof(x3::ExpertLayout));
    h->have_lay[(size_t) layer] = 1;
    if (h->stride == 0) h->stride = stride;
    return 0;
}

int ds4h_reserve(ds4h* h, int slots, size_t stride, char* err, size_t cap) {
    if (!h || slots <= 0 || stride == 0) return fail(err, cap, "helper: bad reserve");
    if (h->pending) return fail(err, cap, "helper: wait for the pending round first");
    if (h->arena) return fail(err, cap, "helper: already reserved");
    if (!ck(cudaSetDevice(h->dev), "helper: cudaSetDevice", err, cap) ||
        !ck(cudaMalloc((void**) &h->arena, (size_t) slots * stride), "helper: expert arena", err, cap))
        return 1;
    h->slots = slots;
    h->stride = stride;
    return 0;
}

int ds4h_upload(ds4h* h, int slot, int layer, const void* blob, size_t bytes, char* err, size_t cap) {
    if (!h || !h->arena || !blob) return fail(err, cap, "helper: upload before reserve");
    if (h->pending) return fail(err, cap, "helper: wait for the pending round first");
    if (slot < 0 || slot >= h->slots) return fail(err, cap, "helper: slot out of range");
    if (bytes > h->stride) return fail(err, cap, "helper: the blob is larger than the arena's stride");
    if (layer < 0 || layer >= (int) h->have_lay.size() || !h->have_lay[(size_t) layer])
        return fail(err, cap, "helper: upload for a layer with no layout");
    if (!ck(cudaSetDevice(h->dev), "helper: cudaSetDevice", err, cap) ||
        !ck(cudaMemcpy(h->arena + (size_t) slot * h->stride, blob, bytes, cudaMemcpyHostToDevice), "helper: upload",
            err, cap))
        return 1;
    if (std::getenv("DS4_HELPER_CHECK") || std::getenv("STRATA_DS4_HELPER_CHECK")) {
        std::vector<uint8_t> back(bytes);
        if (!ck(cudaMemcpy(back.data(),h->arena+(size_t)slot*h->stride,bytes,cudaMemcpyDeviceToHost),"helper: readback",err,cap) || std::memcmp(back.data(),blob,bytes))
            return fail(err,cap,"helper: uploaded expert byte mismatch");
    }
    ++h->st.uploads;
    return 0;
}

int ds4h_submit(ds4h* h, const ds4h_layer* layers, int n, char* err, size_t cap) {
    if (!h || !layers || n <= 0) return fail(err, cap, "helper: empty round");
    if (h->pending) return fail(err, cap, "helper: wait for the pending round first");
    if (!h->arena) return fail(err, cap, "helper: submit before reserve");
    if (!ck(cudaSetDevice(h->dev), "helper: cudaSetDevice", err, cap)) return 1;
    const auto t0 = std::chrono::steady_clock::now();
    // Validate the whole round before enqueueing anything. Each layer owns one pinned staging area: repeating it
    // would overwrite buffers still read by the device. Grow scratch once, before a copy can be in flight.
    std::vector<char> seen(h->have_lay.size(), 0);
    int max_tok = 0;
    for (int li = 0; li < n; ++li) {
        const ds4h_layer& L = layers[li];
        if (L.layer < 0 || L.layer >= (int) h->have_lay.size() || !h->have_lay[(size_t) L.layer])
            return fail(err, cap, "helper: round for a layer with no layout");
        if (L.k < 1 || L.k > 8 || L.n_tok < 1 || L.n_tok > kMaxTok)
            return fail(err, cap, "helper: bad round shape");
        if (!L.x || !L.slot || !L.w || !L.out) return fail(err, cap, "helper: null round buffer");
        if (seen[(size_t) L.layer]) return fail(err, cap, "helper: duplicate layer in a round");
        seen[(size_t) L.layer] = 1;
        for (int i = 0; i < L.n_tok * L.k; ++i)
            if (L.slot[i] >= h->slots) return fail(err, cap, "helper: slot out of range in a round");
        if (L.n_tok > max_tok) max_tok = L.n_tok;
    }
    if (!ensure_tok(h, max_tok, err, cap)) return 1;
    for (int li = 0; li < n; ++li)
        if (!ensure_stage(h, layers[li].layer, err, cap)) return 1;
    h->outputs.clear();
    h->pending = true;
    // An enqueue failure must not leave DMA reading staging that a retry can overwrite.
    const auto abort_round = [&]() {
        cudaStreamSynchronize(h->s);
        h->outputs.clear();
        h->pending = false;
        return 1;
    };
    int64_t experts = 0;
    for (int li = 0; li < n; ++li) {
        const ds4h_layer& L = layers[li];
        const int T = L.n_tok, K = L.k, T8 = T * 8;
        float* const pout = h->pin_out[(size_t) L.layer];
        const size_t out_bytes = (size_t) T * h->n_embd * 4;
        h->outputs.push_back({L.out, pout, out_bytes});
        uint8_t* const pm = h->pin_meta[(size_t) L.layer];
        auto* pp = (unsigned long long*) pm;
        auto* pw = (float*) (pp + T8);
        auto* gp = (unsigned long long*) (pw + T8);
        auto* gs = (int*) (gp + T8);
        auto* ge = gs + T8 + 1;
        auto* ng = ge + T8;

        std::memset(pp, 0, (size_t) T8 * 8);
        std::memset(pw, 0, (size_t) T8 * 4);
        int picks = 0;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < K; ++i) {
                const int sl = L.slot[(size_t) t * K + i];
                if (sl < 0) continue;
                pp[(size_t) t * 8 + i] = (unsigned long long) (h->arena + (size_t) sl * h->stride);
                pw[(size_t) t * 8 + i] = L.w[(size_t) t * K + i];
                ++experts;
                ++picks;
            }
        if (T == 1) {
            if (picks == 0) {
                std::memset(pout, 0, out_bytes);
                continue;
            }
            // A single token needs no grouping metadata: use the same gate/up + down path as native decode.
            // win_experts has the same arithmetic but pays for five extra metadata transfers for its groups.
            std::memcpy(h->pin_x[(size_t) L.layer], L.x, out_bytes);
            if (!ck(cudaMemcpyAsync(h->d_x, h->pin_x[(size_t) L.layer], out_bytes, cudaMemcpyHostToDevice, h->s),
                    "helper: x", err, cap) ||
                !ck(cudaMemcpyAsync(h->d_pp, pp, 8 * 8, cudaMemcpyHostToDevice, h->s), "helper: plan", err, cap) ||
                !ck(cudaMemcpyAsync(h->d_pw, pw, 8 * 4, cudaMemcpyHostToDevice, h->s), "helper: plan w", err, cap))
                return abort_round();
            x3::MoeArgs a{};
            a.plan_ptr = h->d_pp;
            a.plan_w = h->d_pw;
            a.lay = h->lay[(size_t) L.layer];
            a.k = K;
            a.n_embd = h->n_embd;
            a.n_ff = h->n_ff;
            a.limit = L.limit;
            const int errors = gg::launch_errors();
            gg::moe_gate_up(a, h->d_x, h->d_h, h->s);
            gg::moe_down(a, h->d_h, nullptr, h->d_out, h->s);
            if (gg::launch_errors() != errors) {
                fail(err, cap, "helper: token experts failed");
                return abort_round();
            }
            if (!ck(cudaMemcpyAsync(pout, h->d_out, out_bytes, cudaMemcpyDeviceToHost, h->s), "helper: out", err, cap))
                return abort_round();
            continue;
        }
        int ngrp = 0, nent = 0;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < K; ++i) {
                const int sl = L.slot[(size_t) t * K + i];
                if (sl < 0) continue;
                const auto addr = (unsigned long long) (h->arena + (size_t) sl * h->stride);
                bool seen = false;
                for (int g = 0; g < ngrp; ++g) seen = seen || gp[g] == addr;
                if (seen) continue;
                gp[ngrp] = addr;
                gs[ngrp] = nent;
                ++ngrp;
                for (int t2 = 0; t2 < T; ++t2)
                    for (int i2 = 0; i2 < K; ++i2)
                        if (L.slot[(size_t) t2 * K + i2] == sl) ge[nent++] = t2 * 8 + i2;
            }
        gs[ngrp] = nent;
        *ng = ngrp;
        if (ngrp == 0) {   // nothing of the helper's in this layer: its part is zero
            std::memset(pout, 0, out_bytes);
            continue;
        }
        std::memcpy(h->pin_x[(size_t) L.layer], L.x, (size_t) T * h->n_embd * 4);

        if (!ck(cudaMemcpyAsync(h->d_x, h->pin_x[(size_t) L.layer], (size_t) T * h->n_embd * 4,
                                cudaMemcpyHostToDevice, h->s), "helper: x", err, cap) ||
            !ck(cudaMemcpyAsync(h->d_pp, pp, (size_t) T8 * 8, cudaMemcpyHostToDevice, h->s), "helper: plan", err, cap) ||
            !ck(cudaMemcpyAsync(h->d_pw, pw, (size_t) T8 * 4, cudaMemcpyHostToDevice, h->s), "helper: plan w", err, cap) ||
            !ck(cudaMemcpyAsync(h->d_gp, gp, (size_t) ngrp * 8, cudaMemcpyHostToDevice, h->s), "helper: groups", err, cap) ||
            !ck(cudaMemcpyAsync(h->d_gs, gs, (size_t) (ngrp + 1) * 4, cudaMemcpyHostToDevice, h->s), "helper: starts",
                err, cap) ||
            !ck(cudaMemcpyAsync(h->d_ge, ge, (size_t) nent * 4, cudaMemcpyHostToDevice, h->s), "helper: entries", err, cap) ||
            !ck(cudaMemcpyAsync(h->d_ng, ng, 4, cudaMemcpyHostToDevice, h->s), "helper: group count", err, cap))
            return abort_round();
        x3::WinArgs a;
        a.grp_ptr = h->d_gp;
        a.grp_start = h->d_gs;
        a.grp_ent = h->d_ge;
        a.n_groups = h->d_ng;
        a.plan_ptr = h->d_pp;
        a.plan_w = h->d_pw;
        a.x = h->d_x;
        a.x_stride = (size_t) h->n_embd;
        a.h = h->d_h;
        a.part = h->d_part;
        a.out = h->d_out;
        a.out_stride = (size_t) h->n_embd;
        a.lay = h->lay[(size_t) L.layer];
        a.T = T;
        a.k = K;
        a.n_embd = h->n_embd;
        a.n_ff = h->n_ff;
        a.limit = L.limit;
        if (!gg::win_experts(a, h->s)) {
            fail(err, cap, "helper: win_experts failed");
            return abort_round();
        }
        // The caller's output may be pageable or pinned by CUDA, which does not register it with HIP. Copy to
        // this runtime's own pinned buffer so submit stays asynchronous, then publish the result in wait.
        if (!ck(cudaMemcpyAsync(pout, h->d_out, out_bytes, cudaMemcpyDeviceToHost, h->s),
                "helper: out", err, cap))
            return abort_round();
    }
    h->st.experts += experts;
    h->t0 = t0;
    return 0;
}

int ds4h_wait(ds4h* h, char* err, size_t cap) {
    if (!h) return fail(err, cap, "helper: empty wait");
    if (!h->pending) return fail(err, cap, "helper: no pending round");
    if (!ck(cudaSetDevice(h->dev), "helper: cudaSetDevice", err, cap)) return 1;
    if (!ck(cudaStreamSynchronize(h->s), "helper: sync", err, cap)) return 1;
    for (const auto& out : h->outputs) std::memcpy(out.dst, out.src, out.bytes);
    h->outputs.clear();
    h->pending = false;
    const auto t1 = std::chrono::steady_clock::now();
    ++h->st.rounds;
    h->st.last_round_ms = std::chrono::duration<double, std::milli>(t1 - h->t0).count();
    h->st.total_round_ms += h->st.last_round_ms;
    return 0;
}

int ds4h_round(ds4h* h, const ds4h_layer* layers, int n, char* err, size_t cap) {
    if (ds4h_submit(h, layers, n, err, cap) != 0) return 1;
    return ds4h_wait(h, err, cap);
}

void ds4h_get_stats(ds4h* h, ds4h_stats* out) {
    if (h && out) *out = h->st;
}

void ds4h_close(ds4h* h) {
    if (!h) return;
    cudaSetDevice(h->dev);
    if (h->s) cudaStreamSynchronize(h->s);
    free_scratch(h);
    for (auto* p : h->pin_x)
        if (p) cudaFreeHost(p);
    for (auto* p : h->pin_out)
        if (p) cudaFreeHost(p);
    for (auto* p : h->pin_meta)
        if (p) cudaFreeHost(p);
    if (h->arena) cudaFree(h->arena);
    if (h->s) cudaStreamDestroy(h->s);
    delete h;
}

extern "C" int ds4h_version(void) { return 1; }
