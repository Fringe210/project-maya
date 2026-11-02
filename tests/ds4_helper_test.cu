// tests/ds4_helper_test.cu - the expert helper's C ABI (src/ds4/ds4_helper.cpp, the HIP DLL the CUDA engine loads) on
// this build's own card, against a double reference on the device's Q8_K activations (DS4 clamped SwiGLU,
// through the ABI): DS4's mixed IQ2_S/IQ3_S and IQ2_XXS/IQ3_XXS roles plus two other layouts, windows of
// one, three and sixteen tokens with a pick the helper does not own, the blobs uploaded once.
// ds4_shared_helper_test [--legacy]
#include "../src/ds4/ds4_helper.h"

#include "strata/kernels/exl3.hpp"
#include "../src/ds4/ds4_gguf.cuh"
#include "../src/ds4/ds4_iq.cuh"

#include "ggml.h"   // (the reference values: ggml's own dequantize_row_*)

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>

namespace x3 = strata::kernels::exl3;
namespace iq = strata::ds4::gguf::iq;
namespace gg = strata::ds4::gguf;

namespace {

int g_fail = 0;
uint64_t g_output_digest = 14695981039346656037ull;
std::mt19937 rng(4321);

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);      \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

float frand(float a = 1.0f) { return std::uniform_real_distribution<float>(-a, a)(rng); }

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> h(n);
    CU(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

void expect(const char* what, const float* got, const double* want, size_t n, double rtol) {
    // Packed and legacy runs must produce the same digest of every numerical output.
    const auto* bytes = reinterpret_cast<const uint8_t*>(got);
    for (size_t i = 0; i < n * sizeof(float); ++i) {
        g_output_digest ^= bytes[i]; g_output_digest *= 1099511628211ull;
    }
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            ++g_fail;
            std::printf("%-52s FAIL  not finite (at %zu: %.6g vs %.6g)\n", what, i, (double) got[i], want[i]);
            return;
        }
    double worst = 0, scale = 1e-30;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs(want[i]));
    size_t at = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs(got[i] - want[i]) / scale;
        if (!(e <= worst)) {
            worst = e;
            at = i;
        }
    }
    const bool ok = worst <= rtol;
    if (!ok) ++g_fail;
    std::printf("%-52s %s  max rel err %.2e (at %zu: %.6g vs %.6g)\n", what, ok ? "ok  " : "FAIL", worst, at,
                (double) got[at], want[at]);
}

float h2f(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, 2);
    return __half2float(h);
}
uint16_t f2h(float f) {
    const __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

// random blocks of a type (mm3_gguf_test's q_rand): random bytes, the block scales small normal F16s
std::vector<uint8_t> q_rand(int type, int n, int k) {
    const size_t rb = iq::row_bytes(type, k), bb = (size_t) iq::block_bytes(type);
    std::vector<uint8_t> w(rb * n);
    for (auto& v : w) v = (uint8_t) (rng() & 0xff);
    auto put = [&](size_t at, float f) {
        const uint16_t u = f2h(f);
        std::memcpy(&w[at], &u, 2);
    };
    for (size_t o = 0; o < w.size(); o += bb) {
        const float d = 0.001f + 0.002f * std::fabs(frand());
        put(o, d);
    }
    return w;
}
std::vector<float> deq(int type, const std::vector<uint8_t>& w, size_t n) {
    std::vector<float> v(n);
    ggml_get_type_traits((ggml_type) type)->to_float(w.data(), v.data(), (int64_t) n);
    return v;
}
// Q8_K uses 256-value blocks; this ABI repeats each scale across eight 32-value sub-blocks.
std::vector<float> act_dev(const float* dx, int rows, int k, cudaStream_t s) {
    int8_t* q = nullptr;
    float* d = nullptr;
    CU(cudaMalloc(&q, (size_t) rows * k));
    CU(cudaMalloc(&d, (size_t) rows * k / 32 * 4));
    gg::act_q8k(dx, rows, k, q, d, s);
    CU(cudaStreamSynchronize(s));
    const std::vector<int8_t> hq = host(q, (size_t) rows * k);
    const std::vector<float> hd = host(d, (size_t) rows * k / 32);
    std::vector<float> v((size_t) rows * k);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float) ((double) hq[i] * hd[i / 32]);
    cudaFree(q), cudaFree(d);
    return v;
}
double swiglu(double g, double u, double lim) {
    g = std::min(g, lim);
    u = std::min(std::max(u, -lim), lim);
    return g / (1.0 + std::exp(-g)) * u;
}
double dot(const float* w, const float* x, int k) {
    double s = 0;
    for (int i = 0; i < k; ++i) s += (double) w[i] * x[i];
    return s;
}

// an expert layer's blob (mm3_gguf_test's QE, without a device copy: the helper has its own arena)
struct QE {
    int D, F, n_exp, ty[3];
    x3::ExpertLayout lay;
    size_t blob = 0;
    std::vector<uint8_t> h;
    std::vector<std::vector<float>> w;   // [e * 3 + r]: ggml's values, [rows][k]
    const float* row(int e, int r, int i) const { return w[(size_t) e * 3 + r].data() + (size_t) i * (r < 2 ? D : F); }
};
QE make(int D, int F, int n_exp, int tg, int tu, int td) {
    QE E{D, F, n_exp, {tg, tu, td}};
    size_t off = 0;
    for (int r = 0; r < 3; ++r) {
        E.lay.trellis[r] = E.lay.suh[r] = E.lay.svh[r] = off;
        E.lay.gg_type[r] = E.ty[r];
        off += ((r < 2 ? F : D) * iq::row_bytes(E.ty[r], r < 2 ? D : F) + 255) / 256 * 256;
    }
    E.lay.fmt = gg::kFmtMXFP4;
    E.blob = off;
    E.h.assign(E.blob * n_exp, 0);
    for (int e = 0; e < n_exp; ++e)
        for (int r = 0; r < 3; ++r) {
            const int rows = r < 2 ? F : D, k = r < 2 ? D : F;
            const std::vector<uint8_t> m = q_rand(E.ty[r], rows, k);
            std::memcpy(E.h.data() + (size_t) e * E.blob + E.lay.trellis[r], m.data(), m.size());
            E.w.push_back(deq(E.ty[r], m, (size_t) rows * k));
        }
    return E;
}

// Mixed DS4 gate/up and down types, plus additional independently sized layers.
const int kTriples[][3] = {
    {iq::kIQ2_S, iq::kIQ2_S, iq::kIQ3_S},
    {iq::kIQ2_XXS, iq::kIQ2_XXS, iq::kIQ3_XXS},
    {iq::kIQ2_XS, iq::kIQ2_XS, iq::kIQ2_XS},
    {iq::kIQ3_XXS, iq::kIQ3_XXS, iq::kIQ3_XXS},
};

int check(int rc, const char* what, const char* err) {
    if (rc == 0) return 0;
    ++g_fail;
    std::printf("%-52s FAIL  %s\n", what, err);
    return 1;
}

void rejected(int rc, const char* what) {
    if (rc == 0) ++g_fail;
    std::printf("%-52s %s\n", what, rc == 0 ? "FAIL (accepted)" : "ok");
}

void unchanged(const std::vector<float>& out, float value, const char* what) {
    const bool ok = std::all_of(out.begin(), out.end(), [=](float x) { return x == value; });
    if (!ok) ++g_fail;
    std::printf("%-52s %s\n", what, ok ? "ok" : "FAIL");
}

}  // namespace

int main(int argc, char** argv) {
    const bool packed = !(argc >= 2 && std::string(argv[1]) == "--legacy");
    const int D = 512, F = 256, n_exp = 10, K = 4;
    const float lim = 7.0f;
    const int n_lay = (int) (sizeof kTriples / sizeof kTriples[0]);
    std::vector<QE> lay;
    for (const auto& tr : kTriples) lay.push_back(make(D, F, n_exp, tr[0], tr[1], tr[2]));
    size_t stride = 0;
    for (const auto& E : lay) stride = std::max(stride, E.blob);

    cudaStream_t s;
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    char err[512] = {0};
    ds4h* h = nullptr;
    if (check(ds4h_open(0, D, F, &h, err, sizeof err), "helper open", err)) return 1;
    rejected(ds4h_wait(h, err, sizeof err), "wait before first submit is rejected");
    ds4h* old_client = nullptr;
    if (check(ds4h_open(0, D, F, &old_client, err, sizeof err), "legacy-order open", err) ||
        check(ds4h_reserve(old_client, 1, stride, err, sizeof err), "legacy-order reserve first", err) ||
        check(ds4h_layout(old_client, 0, &lay[0].lay, sizeof(x3::ExpertLayout), lay[0].blob, err, sizeof err), "legacy-order layout after reserve", err) ||
        check(ds4h_upload(old_client, 0, 0, lay[0].h.data(), lay[0].blob, err, sizeof err), "legacy-order upload", err)) return 1;
    ds4h_close(old_client);
    for (int li = 0; li < n_lay; ++li)
        if (check(ds4h_layout(h, li, &lay[(size_t) li].lay, sizeof(x3::ExpertLayout), packed ? lay[(size_t) li].blob : stride, err, sizeof err),
                  "helper layout", err))
            return 1;
    std::vector<ds4h_slot> desc;
    uint64_t payload = 0, allocated = 0;
    for (int li = 0; li < n_lay; ++li) for (int e = 0; e < n_exp; ++e) {
        const size_t bytes = lay[(size_t) li].blob;
        desc.push_back({bytes, li, 0}); payload += bytes;
        allocated += packed ? (bytes + 255) & ~(size_t) 255 : stride;
    }
    if (packed) {
        rejected(ds4h_reserve_packed(h, 0, desc.data(), err, sizeof err), "empty packed reserve rejected");
        rejected(ds4h_reserve_packed(h, (int) desc.size(), nullptr, err, sizeof err), "null packed table rejected");
        auto bad = desc; bad.back().layer = n_lay;
        rejected(ds4h_reserve_packed(h, (int) bad.size(), bad.data(), err, sizeof err), "missing layer rejected");
        bad = desc; ++bad.back().bytes;
        rejected(ds4h_reserve_packed(h, (int) bad.size(), bad.data(), err, sizeof err), "wrong blob size rejected");
        bad = desc; bad.back().reserved = 1;
        rejected(ds4h_reserve_packed(h, (int) bad.size(), bad.data(), err, sizeof err), "unknown descriptor flags rejected");
    }
    if (check(packed ? ds4h_reserve_packed(h, (int) desc.size(), desc.data(), err, sizeof err)
                     : ds4h_reserve(h, (int) desc.size(), stride, err, sizeof err), "helper reserve", err)) return 1;
    ds4h_memory memory{}; ds4h_get_memory(h, &memory);
    if (memory.allocated_bytes != allocated || memory.payload_bytes != 0 || memory.packed != packed) ++g_fail;
    std::vector<float> blank_x(D), blank_out(D, -123.0f); int32_t blank_slot = 0; float one = 1;
    ds4h_layer blank{0, 1, 1, lim, blank_x.data(), &blank_slot, &one, blank_out.data()};
    rejected(ds4h_submit(h, &blank, 1, err, sizeof err), "unuploaded slot rejected before enqueue");
    unchanged(blank_out, -123.0f, "unuploaded rejection preserves output");
    for (int li = 0; li < n_lay; ++li)
        for (int e = 0; e < n_exp; ++e)
            if (check(ds4h_upload(h, li * n_exp + e, li, lay[(size_t) li].h.data() + (size_t) e * lay[(size_t) li].blob,
                                  lay[(size_t) li].blob, err, sizeof err),
                      "helper upload", err))
                return 1;

    ds4h_get_memory(h, &memory);
    if (memory.payload_bytes != payload || memory.padding_bytes != allocated - payload) ++g_fail;
    std::printf("arena %s: %llu bytes, payload %llu, padding %llu, fixed-stride equivalent %llu\n",
        packed ? "compact" : "legacy", (unsigned long long) allocated, (unsigned long long) payload,
        (unsigned long long) memory.padding_bytes, (unsigned long long) (desc.size() * stride));
    rejected(ds4h_reserve(h, 1, stride, err, sizeof err), "second arena reserve rejected");
    if (packed) rejected(ds4h_layout(h, 0, &lay[0].lay, sizeof(x3::ExpertLayout), stride, err, sizeof err), "layout mutation rejected");
    if (packed) {
        rejected(ds4h_upload(h, 0, 1, lay[0].h.data(), lay[0].blob, err, sizeof err), "cross-layer upload rejected");
        rejected(ds4h_upload(h, 0, 0, lay[0].h.data(), lay[0].blob - 1, err, sizeof err), "short packed upload rejected");
    }
    struct BatchCase {
        int T = 0;
        std::vector<int32_t> slot;
        std::vector<float> w, x, out;
        std::vector<double> want;
    };
    std::vector<BatchCase> batch((size_t) n_lay);
    const int batch_tokens[] = {1, 3, 16, 1};
    for (int T : {1, 3, 16})
      for (int li = 0; li < n_lay; ++li) {
        const QE& E = lay[(size_t) li];
        char name[64];
        std::snprintf(name, sizeof name, "%s/%s/%s", iq::name(E.ty[0]), iq::name(E.ty[1]), iq::name(E.ty[2]));
        // Include the one-token path and maximum verify window; one pick belongs to the caller (slot -1).
        std::vector<int32_t> slot((size_t) T * K, -1);
        std::vector<int> pick((size_t) T * K, 0);
        std::vector<float> w((size_t) T * K, 0.0f);
        for (int t = 0; t < T; ++t) {
            std::vector<int> all(n_exp);
            for (int i = 0; i < n_exp; ++i) all[i] = i;
            std::shuffle(all.begin(), all.end(), rng);
            for (int i = 0; i < K; ++i) {
                pick[(size_t) t * K + i] = all[(size_t) i];
                slot[(size_t) t * K + i] = t == 0 && i == 2 ? -1 : li * n_exp + all[(size_t) i];
                w[(size_t) t * K + i] = 0.2f + 0.1f * frand();
            }
        }
        std::vector<float> x((size_t) T * D), out((size_t) T * D, 0.0f);
        for (auto& v : x) v = frand();
        ds4h_layer L{};
        L.layer = li;
        L.n_tok = T;
        L.k = K;
        L.limit = lim;
        L.x = x.data();
        L.slot = slot.data();
        L.w = w.data();
        L.out = out.data();
        if (check(ds4h_round(h, &L, 1, err, sizeof err), "helper round", err)) return 1;

        // the reference: gate / up in double on the device's rounded x, the down on the device's own h rounded the same way
        float* dx = dev(x);
        const std::vector<float> xr = act_dev(dx, T, D, s);
        std::vector<double> want((size_t) T * D, 0.0);
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < K; ++i) {
                if (slot[(size_t) t * K + i] < 0) continue;
                const int e = pick[(size_t) t * K + i];
                std::vector<float> hh(F);
                for (int f = 0; f < F; ++f)
                    hh[(size_t) f] = (float) swiglu(dot(E.row(e, 0, f), xr.data() + (size_t) t * D, D),
                                                 dot(E.row(e, 1, f), xr.data() + (size_t) t * D, D), lim);
                float* dhi = dev(hh);
                const std::vector<float> hr = act_dev(dhi, 1, F, s);
                cudaFree(dhi);
                for (int d = 0; d < D; ++d)
                    want[(size_t) t * D + d] +=
                        (double) w[(size_t) t * K + i] * dot(E.row(e, 2, d), hr.data(), F);
            }
        }
        char what[128];
        std::snprintf(what, sizeof what, "%s helper round (T %d) = the double reference", name, T);
        expect(what, out.data(), want.data(), want.size(), 1e-3);
        if (T == batch_tokens[li]) batch[(size_t) li] = {T, slot, w, x, out, want};
        cudaFree(dx);
        ds4h_stats st{};
        ds4h_get_stats(h, &st);
        std::printf("%-52s      %lld experts, %.3f ms round\n", "", (long long) st.experts, st.last_round_ms);
    }
    // Submit multiple differently sized layers with ordinary pageable outputs. The helper must retain independent
    // pinned staging until wait, including after a rejected second submit; successive rounds reuse that staging.
    std::vector<ds4h_layer> ls((size_t) n_lay);
    for (int li = 0; li < n_lay; ++li) {
        auto& b = batch[(size_t) li];
        ls[(size_t) li] = {li, b.T, K, lim, b.x.data(), b.slot.data(), b.w.data(), b.out.data()};
    }
    for (int r = 0; r < 3; ++r) {
        for (auto& b : batch) std::fill(b.out.begin(), b.out.end(), -123.0f);
        if (check(ds4h_submit(h, ls.data(), n_lay, err, sizeof err), "mixed layers submit", err)) return 1;
        for (const auto& b : batch) unchanged(b.out, -123.0f, "output is published only by wait");
        rejected(ds4h_submit(h, ls.data(), n_lay, err, sizeof err), "second submit while pending is rejected");
        if (check(ds4h_wait(h, err, sizeof err), "mixed layers wait", err)) return 1;
        for (const auto& b : batch)
            expect("mixed layers = double reference", b.out.data(), b.want.data(), b.want.size(), 1e-3);
    }
    rejected(ds4h_wait(h, err, sizeof err), "wait without submit is rejected");
    auto bad = ls;
    bad.back().n_tok = 17;
    rejected(ds4h_submit(h, bad.data(), n_lay, err, sizeof err), "bad final layer rejected before enqueue");
    bad = ls;
    bad.back().layer = bad.front().layer;
    rejected(ds4h_submit(h, bad.data(), n_lay, err, sizeof err), "duplicate layer staging is rejected");
    auto& b0 = batch.front();
    const int32_t good_slot = b0.slot[0];
    b0.slot[0] = n_exp * n_lay;
    rejected(ds4h_submit(h, ls.data(), n_lay, err, sizeof err), "out-of-range slot rejected before enqueue");
    if (packed) {
        b0.slot[0] = n_exp;
        rejected(ds4h_submit(h, ls.data(), n_lay, err, sizeof err), "cross-layer round rejected before enqueue");
    }
    b0.slot[0] = good_slot;
    if (check(ds4h_round(h, ls.data(), n_lay, err, sizeof err), "valid round after rejected inputs", err)) return 1;
    for (const auto& b : batch)
        expect("retry = double reference", b.out.data(), b.want.data(), b.want.size(), 1e-3);
    std::fill(b0.slot.begin(), b0.slot.end(), -1);
    std::fill(b0.out.begin(), b0.out.end(), -123.0f);
    if (check(ds4h_round(h, ls.data(), 1, err, sizeof err), "no helper picks", err)) return 1;
    unchanged(b0.out, 0.0f, "no helper picks clear the whole output");
    ds4h_close(h);
    cudaStreamDestroy(s);
    std::printf("OUTPUT_DIGEST %016llx\n", (unsigned long long) g_output_digest);
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
