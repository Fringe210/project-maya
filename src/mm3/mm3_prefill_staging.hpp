// Two banks in the existing expert staging arena. The copy stream may fill the
// next bank while the compute stream reads the current one; neither may reuse a
// bank before its previous reader/copy has finished. No extra weight arena.
#pragma once
#include <cuda_runtime.h>

namespace mm3 {
struct PrefillStaging {
    cudaEvent_t ready[2] = {}, consumed[2] = {};

    static int first(int slots, int bank) { return bank ? slots / 2 : 0; }
    static int count(int slots, int bank) { return bank ? slots - slots / 2 : slots / 2; }

    cudaError_t init() {
        for (int b = 0; b < 2; ++b) {
            cudaError_t e = cudaEventCreateWithFlags(&ready[b], cudaEventDisableTiming);
            if (e != cudaSuccess) return e;
            e = cudaEventCreateWithFlags(&consumed[b], cudaEventDisableTiming);
            if (e != cudaSuccess) return e;
        }
        return cudaSuccess;
    }
    void release() {
        for (int b = 0; b < 2; ++b) {
            if (ready[b]) cudaEventDestroy(ready[b]);
            if (consumed[b]) cudaEventDestroy(consumed[b]);
            ready[b] = consumed[b] = nullptr;
        }
    }
    cudaError_t begin(cudaStream_t compute) {
        // Also protect any decode/vision work which previously used these slots.
        for (int b = 0; b < 2; ++b) {
            const cudaError_t e = cudaEventRecord(consumed[b], compute);
            if (e != cudaSuccess) return e;
        }
        return cudaSuccess;
    }
    cudaError_t copy_begin(int bank, cudaStream_t copy) {
        return cudaStreamWaitEvent(copy, consumed[bank], 0);
    }
    cudaError_t copy_end(int bank, cudaStream_t copy) { return cudaEventRecord(ready[bank], copy); }
    cudaError_t compute_begin(int bank, cudaStream_t compute) {
        return cudaStreamWaitEvent(compute, ready[bank], 0);
    }
    cudaError_t compute_end(int bank, cudaStream_t compute) { return cudaEventRecord(consumed[bank], compute); }
};
}  // namespace mm3
