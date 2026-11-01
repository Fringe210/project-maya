// tests/mm3_helper_test.cu - the expert helper's C ABI (src/mm3/mm3_helper.cpp, the HIP DLL the CUDA engine loads) on
// this build's own card, against a double reference on the device's rounded activations (mm3_gguf_test's expert test,
// through the ABI): the four types MiniMax's Maya-S pack mixes, one layer a type (the pack's layers differ), windows of
// one, three and sixteen tokens with a pick the helper does not own, the blobs uploaded once.
// mm3_helper_test
#include "../src/mm3/mm3_helper.h"

#include "../src/mm3/mm3_exl3.hpp"
#include "../src/mm3/mm3_gguf.cuh"
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

namespace x3 = strata::mm3::exl3;
namespace iq = strata::ds4::gguf::iq;
namespace gg = mm3::gg;

namespace {

int g_fail = 0;
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
// rows rounded by the device as the experts' activations are (gg::act_quant): their values
std::vector<float> act_dev(const float* dx, int rows, int k, cudaStream_t s) {
    int8_t* q = nullptr;
    float* d = nullptr;
    CU(cudaMalloc(&q, (size_t) rows * k));
    CU(cudaMalloc(&d, (size_t) rows * k / 32 * 4));
    gg::act_quant(dx, rows, k, q, d, s);
    CU(cudaStreamSynchronize(s));
    const std::vector<int8_t> hq = host(q, (size_t) rows * k);
    const std::vector<float> hd = host(d, (size_t) rows * k / 32);
    std::vector<float> v((size_t) rows * k);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float) ((double) hq[i] * hd[i / 32]);
    cudaFree(q), cudaFree(d);
    return v;
}
double oai(double g, double u, double lim) {
    g = std::min(g, lim);
    u = std::min(std::max(u, -lim), lim);
    return g / (1.0 + std::exp(-1.702 * g)) * (u + 1.0);
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
    E.lay.fmt = gg::kFmtGG;
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

// MiniMax-M3.1's Maya-S layers: the pack's four expert types (IQ2_S, IQ2_XS, IQ2_XXS, IQ3_XXS), one a layer
const int kTriples[][3] = {
    {iq::kIQ2_XXS, iq::kIQ2_XXS, iq::kIQ2_XXS},
    {iq::kIQ2_XS, iq::kIQ2_XS, iq::kIQ2_XS},
    {iq::kIQ2_S, iq::kIQ2_S, iq::kIQ2_S},
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

// the blobs alone (no ggml values): a round of the real shape - 60 layers, a verify window's 8 tokens, top 4
std::vector<uint8_t> blobs(int D, int F, int n_exp, int ty, size_t& stride) {
    size_t off = 0;
    for (int r = 0; r < 3; ++r) {
        const int rows = r < 2 ? F : D, k = r < 2 ? D : F;
        off += ((size_t) rows * iq::row_bytes(ty, k) + 255) / 256 * 256;
    }
    stride = off;
    std::vector<uint8_t> h(off * (size_t) n_exp);
    for (auto& v : h) v = (uint8_t) (rng() & 0xff);
    return h;
}

int bench(bool token = false, const std::string& schedule = "auto", int picks = 1) {
#if defined(STRATA_USE_HIP)
    CU(hipSetDeviceFlags(schedule == "spin" ? hipDeviceScheduleSpin :
                        schedule == "yield" ? hipDeviceScheduleYield : hipDeviceScheduleAuto));
#endif
    const int D = 6144, F = 3072, n_exp = 10, K = 4, T = token ? 1 : 8, n_lay = 60, ty = iq::kIQ2_XXS;
    size_t stride = 0;
    const std::vector<uint8_t> blob = blobs(D, F, n_exp, ty, stride);
    char err[512] = {0};
    mm3h* h = nullptr;
    if (check(mm3h_open(0, D, F, &h, err, sizeof err), "bench open", err)) return 1;
    x3::ExpertLayout lay{};
    lay.fmt = gg::kFmtGG;
    {
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const int rows = r < 2 ? F : D, k = r < 2 ? D : F;
            lay.gg_type[r] = ty;
            lay.trellis[r] = lay.suh[r] = lay.svh[r] = off;
            off += ((size_t) rows * iq::row_bytes(ty, k) + 255) / 256 * 256;
        }
    }
    for (int li = 0; li < n_lay; ++li)
        if (check(mm3h_layout(h, li, &lay, sizeof lay, stride, err, sizeof err), "bench layout", err)) return 1;
    if (check(mm3h_reserve(h, n_exp, stride, err, sizeof err), "bench reserve", err)) return 1;
    for (int e = 0; e < n_exp; ++e)
        if (check(mm3h_upload(h, e, 0, blob.data() + (size_t) e * stride, stride, err, sizeof err), "bench upload", err))
            return 1;

    std::vector<int32_t> slot((size_t) T * K, 0);
    std::vector<float> w((size_t) T * K, 0.25f);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i) slot[(size_t) t * K + i] = !token || i < picks ? (t * K + i) % n_exp : -1;
    std::vector<float> x((size_t) n_lay * T * D), out((size_t) n_lay * T * D, 0.0f);
    for (auto& v : x) v = frand();
    std::vector<mm3h_layer> ls((size_t) n_lay);
    for (int li = 0; li < n_lay; ++li) {
        mm3h_layer& L = ls[(size_t) li];
        L.layer = li;
        L.n_tok = T;
        L.k = K;
        L.limit = 7.0f;
        L.x = x.data() + (size_t) li * T * D;
        L.slot = slot.data();
        L.w = w.data();
        L.out = out.data() + (size_t) li * T * D;
    }
    double best = 1e9, sum = 0.0;
    double submit_ms = 0.0, wait_ms = 0.0;
    const auto elapsed = [](auto t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    const int rounds = 20;
    for (int r = 0; r < rounds; ++r) {
        mm3h_stats st{};
        const auto t0 = std::chrono::steady_clock::now();
        if (token) {
            for (const auto& L : ls) {
                const auto tsub = std::chrono::steady_clock::now();
                if (check(mm3h_submit(h, &L, 1, err, sizeof err), "bench submit", err)) return 1;
                if (r >= 2) submit_ms += elapsed(tsub);
                const auto twait = std::chrono::steady_clock::now();
                if (check(mm3h_wait(h, err, sizeof err), "bench wait", err)) return 1;
                if (r >= 2) wait_ms += elapsed(twait);
            }
        } else if (check(mm3h_round(h, ls.data(), n_lay, err, sizeof err), "bench round", err)) return 1;
        const double ms = elapsed(t0);
        if (r >= 2) {
            best = std::min(best, ms);
            sum += ms;
        }
        mm3h_get_stats(h, &st);
    }
    mm3h_stats st{};
    mm3h_get_stats(h, &st);
    std::printf("%d layers, T %d, k %d, %d pooled experts (%zu B): %.1f ms a round, best %.1f (%.3f ms an expert)\n",
                n_lay, T, K, n_exp, stride, sum / (rounds - 2), best, best / (n_lay * T * (token ? picks : K)));
    if (token) std::printf("schedule=%s picks=%d submit=%.3f ms wait=%.3f ms per60layers\n", schedule.c_str(), picks,
                           submit_ms / (rounds - 2), wait_ms / (rounds - 2));
    mm3h_close(h);
    return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--bench") return bench();
    if (argc >= 2 && std::string(argv[1]) == "--bench-token")
        return bench(true, argc >= 3 ? argv[2] : "auto", argc >= 4 ? std::max(1, std::min(4, std::atoi(argv[3]))) : 1);
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
    mm3h* h = nullptr;
    if (check(mm3h_open(0, D, F, &h, err, sizeof err), "helper open", err)) return 1;
    rejected(mm3h_wait(h, err, sizeof err), "wait before first submit is rejected");
    for (int li = 0; li < n_lay; ++li)
        if (check(mm3h_layout(h, li, &lay[(size_t) li].lay, sizeof(x3::ExpertLayout), stride, err, sizeof err),
                  "helper layout", err))
            return 1;
    if (check(mm3h_reserve(h, n_exp * n_lay, stride, err, sizeof err), "helper reserve", err)) return 1;
    for (int li = 0; li < n_lay; ++li)
        for (int e = 0; e < n_exp; ++e)
            if (check(mm3h_upload(h, li * n_exp + e, li, lay[(size_t) li].h.data() + (size_t) e * lay[(size_t) li].blob,
                                  lay[(size_t) li].blob, err, sizeof err),
                      "helper upload", err))
                return 1;

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
        mm3h_layer L{};
        L.layer = li;
        L.n_tok = T;
        L.k = K;
        L.limit = lim;
        L.x = x.data();
        L.slot = slot.data();
        L.w = w.data();
        L.out = out.data();
        if (check(mm3h_round(h, &L, 1, err, sizeof err), "helper round", err)) return 1;

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
                    hh[(size_t) f] = (float) oai(dot(E.row(e, 0, f), xr.data() + (size_t) t * D, D),
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
        expect(what, out.data(), want.data(), want.size(), 1e-4);
        if (T == batch_tokens[li]) batch[(size_t) li] = {T, slot, w, x, out, want};
        cudaFree(dx);
        mm3h_stats st{};
        mm3h_get_stats(h, &st);
        std::printf("%-52s      %lld experts, %.3f ms round\n", "", (long long) st.experts, st.last_round_ms);
    }
    // Submit multiple differently sized layers with ordinary pageable outputs. The helper must retain independent
    // pinned staging until wait, including after a rejected second submit; successive rounds reuse that staging.
    std::vector<mm3h_layer> ls((size_t) n_lay);
    for (int li = 0; li < n_lay; ++li) {
        auto& b = batch[(size_t) li];
        ls[(size_t) li] = {li, b.T, K, lim, b.x.data(), b.slot.data(), b.w.data(), b.out.data()};
    }
    for (int r = 0; r < 3; ++r) {
        for (auto& b : batch) std::fill(b.out.begin(), b.out.end(), -123.0f);
        if (check(mm3h_submit(h, ls.data(), n_lay, err, sizeof err), "mixed layers submit", err)) return 1;
        for (const auto& b : batch) unchanged(b.out, -123.0f, "output is published only by wait");
        rejected(mm3h_submit(h, ls.data(), n_lay, err, sizeof err), "second submit while pending is rejected");
        if (check(mm3h_wait(h, err, sizeof err), "mixed layers wait", err)) return 1;
        for (const auto& b : batch)
            expect("mixed layers = double reference", b.out.data(), b.want.data(), b.want.size(), 1e-4);
    }
    rejected(mm3h_wait(h, err, sizeof err), "wait without submit is rejected");
    auto bad = ls;
    bad.back().n_tok = 17;
    rejected(mm3h_submit(h, bad.data(), n_lay, err, sizeof err), "bad final layer rejected before enqueue");
    bad = ls;
    bad.back().layer = bad.front().layer;
    rejected(mm3h_submit(h, bad.data(), n_lay, err, sizeof err), "duplicate layer staging is rejected");
    auto& b0 = batch.front();
    const int32_t good_slot = b0.slot[0];
    b0.slot[0] = n_exp * n_lay;
    rejected(mm3h_submit(h, ls.data(), n_lay, err, sizeof err), "out-of-range slot rejected before enqueue");
    b0.slot[0] = good_slot;
    if (check(mm3h_round(h, ls.data(), n_lay, err, sizeof err), "valid round after rejected inputs", err)) return 1;
    for (const auto& b : batch)
        expect("retry = double reference", b.out.data(), b.want.data(), b.want.size(), 1e-4);
    std::fill(b0.slot.begin(), b0.slot.end(), -1);
    std::fill(b0.out.begin(), b0.out.end(), -123.0f);
    if (check(mm3h_round(h, ls.data(), 1, err, sizeof err), "no helper picks", err)) return 1;
    unchanged(b0.out, 0.0f, "no helper picks clear the whole output");
    mm3h_close(h);
    cudaStreamDestroy(s);
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
