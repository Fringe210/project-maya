// Exercise the production bank fences on a real GPU, without loading a model.
// Delayed readers make overwriting an in-flight bank observable in the results.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "mm3_prefill_staging.hpp"

#define CUDA(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return false; } } while (0)

__global__ void consume(const int* weights, int* out, int n, unsigned long long delay) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const unsigned long long start = clock64();
    while (clock64() - start < delay) {}
    const volatile int* w = weights;
    int sum = 0;
    for (int j = 0; j < 32; ++j) sum += w[i];
    out[i] = sum;
}

bool trial(int slots, int passes, int rounds, bool fences) {
    constexpr int W = 128;
    const int max_n = ((slots + 1) / 2) * W, total = rounds * passes;
    int *arena = nullptr, *out = nullptr, *host = nullptr;
    cudaStream_t compute = nullptr, copy = nullptr;
    mm3::PrefillStaging pipe;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    CUDA(pipe.init());
    CUDA(cudaMalloc(&arena, (size_t) slots * W * sizeof(int)));
    CUDA(cudaMalloc(&out, (size_t) total * max_n * sizeof(int)));
    CUDA(cudaMallocHost(&host, (size_t) total * max_n * sizeof(int)));
    for (int p = 0; p < total; ++p)
        for (int i = 0; i < max_n; ++i) host[(size_t) p * max_n + i] = 10000 * p + i + 7;
    CUDA(cudaMemsetAsync(out, 0, (size_t) total * max_n * sizeof(int), compute));
    auto prefetch = [&](int p, int local) -> bool {
        const int bank = local % 2;
        if (fences) CUDA(pipe.copy_begin(bank, copy));
        CUDA(cudaMemcpyAsync(arena + pipe.first(slots, bank) * W, host + (size_t) p * max_n,
                             (size_t) pipe.count(slots, bank) * W * sizeof(int), cudaMemcpyHostToDevice, copy));
        CUDA(pipe.copy_end(bank, copy));
        return true;
    };
    for (int r = 0; r < rounds; ++r) {
        // No host wait between rounds: begin must also fence readers left by the
        // preceding layer, even when it ended on a different bank.
        CUDA(pipe.begin(compute));
        if (!prefetch(r * passes, 0)) return false;
        for (int p = 0; p < passes; ++p) {
            const int bank = p % 2, index = r * passes + p, n = pipe.count(slots, bank) * W;
            CUDA(pipe.compute_begin(bank, compute));
            consume<<<(n + 127) / 128, 128, 0, compute>>>(arena + pipe.first(slots, bank) * W,
                                                        out + (size_t) index * max_n, n, 300000);
            CUDA(cudaGetLastError());
            CUDA(pipe.compute_end(bank, compute));
            if (p + 1 < passes && !prefetch(index + 1, p + 1)) return false;
        }
    }
    CUDA(cudaStreamSynchronize(compute));
    CUDA(cudaStreamSynchronize(copy));
    std::vector<int> got((size_t) total * max_n);
    CUDA(cudaMemcpy(got.data(), out, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int r = 0; r < rounds; ++r)
        for (int p = 0; p < passes; ++p) {
            const int index = r * passes + p, n = pipe.count(slots, p % 2) * W;
            for (int i = 0; i < n; ++i)
                errors += got[(size_t) index * max_n + i] != 32 * host[(size_t) index * max_n + i];
        }
    CUDA(cudaFree(arena)); CUDA(cudaFree(out)); CUDA(cudaFreeHost(host));
    pipe.release();
    CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(copy));
    std::printf("slots=%d passes=%d rounds=%d fences=%d mismatches=%d\n", slots, passes, rounds, fences, errors);
    return fences ? errors == 0 : errors > 0;   // negative control must expose the missing read fence
}

int main() {
    for (int slots : {2, 7, 48})
        for (int passes : {1, 2, 3, 9})
            if (!trial(slots, passes, 5, true)) return 1;
    if (!trial(7, 9, 2, false)) return 1;
    std::puts("mm3_prefill_staging_test: ALL OK (including missing-fence negative control)");
    return 0;
}
