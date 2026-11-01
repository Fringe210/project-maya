# GLM: NVIDIA and optional AMD expert compute

The canonical checkout uses the same `build/maya-backend.exe --family glm` for both configurations. Weights remain in `E:/Maya-glm-data/models/Maya-M/pack`; the old GLM checkout is not launched. Tests below retain **262144 context, one stream, KV streaming**, the original expert profile and model quantization.

The HIP helper computes its resident experts during decode. **Large batched prefill still runs on NVIDIA and stages AMD-resident expert weights through host memory. AMD batched expert compute is not implemented.** Short prompts can use the token path. The optional adaptive CUDA cache admits GPU-backed experts through bounded host staging; cache hits then use the CUDA copy directly. This distinction matters when assessing PCIe traffic.

## Updated all-VRAM decode — 2026-10-31

The local GLM profile now selects **NVIDIA + W7800, no MTP, 262144 context**, with a 32 GB decimal AMD budget, mapped HIP buffers and an adaptive AMD-backed CUDA cache. This is the same common backend and BAT; it does not launch another installation.

Fresh sequential comparisons used the same two code prompts, four excluded warmups and eight 320-token responses per variant, temperature 0, seed 42, thinking off, one stream:

| All-VRAM, MTP not loaded | Decode tok/s |
|---|---:|
| Original helper/cache policy | 55.63 |
| Synchronous adaptive tail + mapped HIP buffers | 58.69 |
| Asynchronous tail cache, recency admission | 63.41 |
| Additional cold-main AMD backups, recency admission | 62.88 |
| **AMD-backed adaptive CUDA cache, reuse-based admission** | **69.91** |

The final policy improves decode **25.7%** over the fresh all-VRAM control. The earlier NVIDIA-only/no-MTP reference was 73.75 tok/s: the updated all-VRAM deployment reaches about **94.8%** of that rate, with a 268 MiB expert RAM staging tier instead of 22.8 GiB of host expert cache. The NVIDIA-only reference is from the preceding campaign, not a new interleaved comparison.

AMD stores **3700** experts/copies using **31,993,364,480 bytes**, including 1114 additional cold-main backups. The CUDA table uses its 10392 physical slots adaptively; AMD-only counts vary with admissions. Each of the 12096 routed experts is counted once in the residency bar, while physical duplicate copies remain visible in the note. Host KV, embeddings, driver memory and bounded transfer buffers remain.

The real 16384-token document measured **2587.9 prefill tok/s**, recovered all three codes and reused 16383 tokens in its immediate continuation. Four short code requests immediately after the long prompt measured 64.95 tok/s including cache readaptation. A separate single 16451-token document-plus-code request also recovered all codes and produced 320 tokens correctly at 55.5 decode tok/s; it reread the prompt after intervening conversations (cached tokens 0). That single long-context case is a different workload from the eight short-prompt measurements above.

The kernel check passed 148 real expert pairs and six invalid ABI cases; the full load sampled two experts in every served MoE layer (42 layers, maximum relative error 0). The first 64 asynchronous CUDA admissions were byte-identical to their pinned sources. Runtime expert SSD reads remained **zero** through short decode, long prefill and prefix reuse. The actual Preact tier component checks cover duplicate ownership counts; Python checks passed after correcting two existing locale-dependent formatting expectations.

This recovery deliberately trades some PCIe weight traffic for CUDA cache reuse. During the eight measured responses, cache admissions read **6.68 GB** of expert bytes from AMD, versus 24.10 GB with recency-only asynchronous tail admission. These are bounded GPU-to-host-to-GPU copies, not retained host model copies or SSD reads. The final mode is **not** a results-only PCIe path. Long batched prefill still stages AMD weights to NVIDIA.

Raw requests, metrics, logs, configuration snapshots and hashes are in `bench/results/2026-10-31-glm-hip-cache/summary.json`; regenerate with `bench/glm_hip_cache_summary.py`.

## Initial deployment measurements — 2026-10-31

RTX PRO 6000 Blackwell (96 GiB), W7800 (48 GiB), Ryzen 5950X, CUDA 13.2, HIP SDK 6.4. Two fixed Italian code prompts, temperature 0, seed 42, thinking off, 320 output tokens each. Warmups excluded; eight measurements per setting. MTP 2/3 used ABBA blocks in the same process. Rates are total output tokens / total decode time.

| Configuration | Decode tok/s | Real 16K prefill tok/s | Pinned expert-cache GiB |
|---|---:|---:|---:|
| NVIDIA, MTP not loaded | **73.75** | **2628.3** | 22.8 |
| NVIDIA, MTP 2 predictions | 50.94 | 2554.6 | 28.3 |
| NVIDIA, MTP 3 predictions | 47.30 | 2571.9 | 28.3 |
| NVIDIA + AMD, all routed experts in VRAM, no MTP | 55.26 | 2527.0 | 0.26 |
| NVIDIA + AMD, all routed experts in VRAM, MTP 2 | 57.49 | not separately measured | about 0.44 |
| NVIDIA + AMD, all routed experts in VRAM, MTP 3 | 52.14 | not separately measured | about 0.44 |

With the MTP block loaded on NVIDIA but drafting bypassed, decode measured **62.38 tok/s**. That is a separate control: bypassing drafting retains the weights/buffers, whereas not loading MTP gives the main expert cache more room. Cross-deployment comparisons have different memory plans and are not isolated kernel speedups.

All 16K documents retrieved the three codes placed at the start, middle and end; cached continuations reused more than 16K tokens. A short functional JSON case (20 squares, 48 output tokens) was correct in all modes: NVIDIA no-MTP 56.2 tok/s, MTP 2 72.8, MTP 3 69.4. Those are individual short checks, not the eight-sample code benchmark; MTP benefit depends on output predictability.

These measurements favor NVIDIA without MTP for the tested code workloads. AMD all-VRAM mode substantially reduces duplicated host expert storage, but is not the fastest measured deployment. Native `ram_gb`/`vram_gb` are GiB; they do not include all system RAM. KV, embeddings, transfer buffers, Python and the drivers remain separate.

## MTP controls

The three local GGUF shards already contain NextN block 45: no separate drafter download is needed. Single-CUDA-GPU MTP is off by default in the upstream driver. `MAYA_GLM_MTP_DECODE=1` loads/enables it.

- `--spec 3`, or request `strata_tune: {"spec": 3}`: at most **two** predictions plus the current token.
- `--spec 4`, or request `spec: 4`: at most **three** predictions plus the current token.
- Request `spec: 1`: no drafting for that request; the loaded MTP weights remain allocated.
- `MAYA_GLM_NO_MTP=1`: do not load the MTP block at all.
- `MAYA_GLM_MTP_ADAPT=0`: fixed prediction count for controlled comparisons. Otherwise the existing adaptive policy may choose fewer predictions. The global `MAYA_GLM_MTP_DRAFT` cap still bounds request-level choices.

The driver now applies the verify-window setting to MTP as well as DFlash; previously a per-request MTP window could not select two versus three predictions.

## Optional all-VRAM expert policy

The shared `maya-hhelper.dll` uses GLM's own quantized kernels and clamped activation. Enable HIP device 0 with `MAYA_GLM_HIP_HELPER=0` (the index zero enables it), budget `MAYA_GLM_HIP_HELPER_GB=32`, and point `MAYA_GLM_HIP_HELPER_DLL` at this checkout's `build/maya-hhelper.dll`. `off` disables the device. See [the unified build](UNIFIED-BACKEND.md).

`MAYA_GLM_HIP_VRAM_ONLY=1` requires NVIDIA main slots plus AMD storage to cover every routed expert, including the MTP layer when loaded. Loading fails if they do not fit. AMD also stores the weights that would occupy NVIDIA's borrowable prompt/vision tail, so lending the tail cannot lose the only GPU copy. After loading, an expert with no VRAM owner is an error rather than a silent SSD fallback.

Before adaptive caching, without MTP: 9510 NVIDIA-main experts + 2586 AMD experts = 12096, AMD allocation 22,346,072,064 bytes. With MTP: 9165 + 3219 = 12384, AMD allocation 28,060,680,192 bytes. The expert RAM tier keeps only 16 staging slots per size class (268 MiB without MTP; about 0.44 GiB with it). This covers **routed experts**, not every host allocation: embeddings, KV and bounded transfer buffers still use RAM.

The older complementary-pool policy (`MAYA_GLM_HIP_VRAM_ONLY=0`) retains protected RAM copies of helper-owned experts. In all-VRAM mode a compute failure can instead read weights from usable AMD storage into the existing bounded CPU staging blobs. This is not an independent CPU copy for recovery from complete AMD-device loss. The injected compute-failure test returned a correct answer with CPU fallback and zero expert SSD reads.

`expert_hip_bytes` and `expert_cuda_bytes` count weight readbacks into host staging, not activation/result traffic. `expert_ssd_bytes` counts actual pack reads; all-VRAM mode resets it at READY after loading, and runtime guards forbid expert SSD reads. In other modes it also includes loading, so compare before/after deltas.

## PCIe behavior

There is no CUDA-to-AMD peer-memory path in this implementation. Helper compute exchanges activations, routing and results through runtime-owned host buffers. With the adaptive CUDA cache enabled, selected whole experts also move AMD → bounded host staging → NVIDIA when admitted. Cache hits require no expert weight copies. The source counters distinguish these copies from activation/result traffic.

For short prompts, `MAYA_GLM_PREFILL_MIN=128` avoids the NVIDIA batched-prefill weight staging. On the same 63-token test, NVIDIA receive traffic during prefill fell from **4329.8 to 184.6 MB/s** (peak 5757.0 to 195.4); expert weight readbacks fell from **6.92 GB to zero**, and prefill latency from **2169.7 to 1236.3 ms**. Decode receive traffic was about 200–218 MB/s. Requests/results and NVML samples are saved in `glm-pcie-before.json` and `glm-pcie-after.json`.

This does not eliminate weight traffic for **long batched prompts**. Their NVIDIA path still downloads the AMD-owned experts into bounded staging. Executing the AMD share of batched MoE on AMD, then merging its output, is the outstanding architectural change.

## Adaptive CUDA cache with permanent AMD backing

`MAYA_GLM_HIP_TAIL_CACHE=1` keeps NVIDIA's disposable prompt/vision tail useful during decode. Every cached expert retains permanent AMD backing, so lending or replacing the CUDA copy cannot create an expert that exists only in RAM or on SSD. The primary router uses its CUDA table first, then the helper.

`MAYA_GLM_HIP_CACHE_BACKUPS=1` uses spare capacity **within** `MAYA_GLM_HIP_HELPER_GB` to back up cold primary-main experts on AMD, distributed across layers. Those main slots can then participate in the adaptive CUDA cache too. A primary expert with no AMD backing is never a cache victim. The initial complementary coverage must still fit before optional backups are uploaded.

Admissions compare aged routing frequency, require a reuse advantage over the victim, and consider recent use. This avoids copying an expert at every token merely because a later layer just used it. At most one admission starts per idle boundary. Two stable pinned landing buffers bound host storage; a copy-stream event completes before the CUDA table publishes the new entry. The old entry is removed before its bytes are overwritten. Prompt/vision lending drains pending copies before reusing the memory. `MAYA_GLM_HIP_CACHE_CHECK=64` checks the first 64 completed admissions byte for byte against their pinned source; use `0` to disable this diagnostic.

`MAYA_GLM_HIP_ZEROCOPY=1` lets the HIP kernels access the helper's own mapped host input, routing and output buffers. GPU pointers remain private to HIP. The isolated two-expert call measured about 334 µs with copies versus 241 µs with mapped buffers, with identical outputs in the transfer-mode comparison. The full 148-pair numerical check also passed. The optional polling wait (`MAYA_GLM_HIP_SPIN_WAIT=1`) gave no improvement and remains off.

The Monitor counts each routed expert once: NVIDIA-resident versus AMD-only. Its note separately reports the AMD experts also cached on NVIDIA. AMD's VRAM bytes include those physical backup copies. This policy does not implement AMD batched-prefill compute or eliminate weight traffic while the cache adapts.

## Reproduction and scope

`bench/unified_backend_check.py` saves real requests, outputs, timings and GPU maps. Use `MAYA_BENCH_URL=http://127.0.0.1:8081` for an isolated test instance; `--spec 1|3|4` selects the control/two/three predictions. `bench/glm_spec_check.py --nvidia-only` checks real structured output and active drafting. `bench/glm_pcie_check.py` separates prefill/decode bus samples. `bench/glm_hip_cache_bench.py` checks all-VRAM/no-MTP operation, eight measured responses, a real 16K document and decode after lending; `bench/glm_hip_latency_check.py` compares helper transfer/wait modes on real expert weights. `bench/glm_hip_cache_summary.py` aggregates `bench/results/2026-10-31-glm-hip-cache/`. `bench/unified_result_summary.py` aggregates the saved evidence.

The component test `glm_cross_helper_test` covers 148 real expert pairs with changing input/routing weights, full/partial byte-exact GPU weight readbacks, and six rejected invalid ABI calls. It isolates rare Q8/FP16 rounding ties and compares down projections with common hidden activations. Full-model helper checks sample two experts from every served MoE layer. The segmented Windows RAM verifier uses `RamClass::slot()`.

Raw evidence is in `bench/results/2026-10-31-unified/summary.json`. Full 262K stress prompts and parallel 2/4-stream CUDA+HIP tests were not run; parallelism was explicitly deferred.
