# Unified CUDA backend and AMD expert helper

The Windows CUDA build now produces **one `build/maya-backend.exe`**. It links the DeepSeek, GLM and MiniMax architecture implementations into the same process, sharing kernel libraries, build, native protocol and Python server/dashboard. Architecture-specific attention, activation and KV layouts remain distinct. There is no subprocess dispatch to the old model engines.

## Build and launch

1. Run `build-unified.bat` (Visual Studio 2022 BuildTools, CUDA and the existing Python venv/local llama.cpp dependency). `MAYA_CUDA_ARCHS` defaults to `120`; set it for your GPU.
2. For an AMD helper, run `tools\hip\build_helper_windows.bat`. Defaults: HIP SDK 6.4, `gfx1100`. Override with `DS4_HIP_SDK` and `DS4_HIP_ARCH`. While all model processes are stopped, copy `build-hip\maya-hhelper.dll` to `build\maya-hhelper.dll`.
3. Run `START-MAYA.bat`. Its menu chooses installed weight profiles in **this checkout**. `START-MAYA.bat --list` lists them; `START-MAYA.bat --model deepseek`, `--model glm` or `--model minimax` starts one directly. Add `--dry-run` to inspect the resolved configuration without changing it. `START-MAYA.bat --setup` still enters setup.

The native command is `build\maya-backend.exe --family deepseek|glm|minimax` followed by that family's model options. `--list-families` lists the supported families. The default CUDA build excludes the old standalone executables; they remain optional developer targets. HIP/Metal standalone builds retain their existing entry points.

Profiles are local `maya-*.json` files, ignored by Git because they contain machine paths. Each uses the same `exe` (`<checkout>/build/maya-backend.exe`) and `cwd` (`<checkout>`), with `--family` first in `args`; `tokenizer`, model path and architecture options select the weights. The launcher migrates legacy CUDA profile executable paths to this backend. Weights stay in their existing folders. Multiple installed quant profiles appear as separate menu choices.

On the validated PC:

| Profile | Weights | Context and preserved options |
|---|---|---|
| `maya-ds4.json` | `E:/Maya-data/models/DeepSeek-V4.1-Flash-1017-GGUF/Maya-S/maya-ds4` | 262144, TP1, DSpark on, mmap-experts off |
| `maya-glm.json` | `E:/Maya-glm-data/models/Maya-M/pack` | 262144, expert-cache/prefill auto, KV streaming on, existing expert profile |
| `maya-minimax.json` | `H:/Maya-data/models/MiniMax-M3.1-Flash-Maya-S/maya-mm3` | 131072, KV int8, DSpark on, KV streaming on, existing expert profile |

The former GLM checkout's source matched upstream v1.1.27 after newline normalization: its config, expert profile and vision assets were preserved while using the newer common source, rather than replacing the newer engine with that older checkout.

## Cross-vendor helper

One HIP DLL exports **`ds4h_*`, `glmh_*`, `mm3h_*`**. Separate C ABIs preserve each architecture's geometry, quantization and activation. All runtime-owned GPU pointers stay inside their runtime; activations/results cross host memory. NVIDIA keeps the trunk, attention/KV and model-specific speculation. The helper offloads complementary routed experts during decode; large batched prefill stays on the architecture's primary path.

Set the relevant entries in the profile's `env` (paths below are examples):

| Family | Device and budget | DLL and diagnostics |
|---|---|---|
| DeepSeek | `DS4_HIP_HELPER=0`, `DS4_HIP_HELPER_GB=32` | `DS4_HIP_HELPER_DLL=E:/project-maya/build/maya-hhelper.dll`, `DS4_HIP_HELPER_BIN=C:/Program Files/AMD/ROCm/6.4/bin`, `DS4_HELPER_CHECK=1` |
| GLM | `MAYA_GLM_HIP_HELPER=0`, `MAYA_GLM_HIP_HELPER_GB=32` | `MAYA_GLM_HIP_HELPER_DLL=E:/project-maya/build/maya-hhelper.dll`, `MAYA_GLM_HELPER_CHECK=2` |
| MiniMax | Native args `--helper on --helper-gb 40` | `MAYA_MM3_HELPER_DLL=E:/project-maya/build/maya-hhelper.dll` |

These are JSON string values in `env`. Device `0` **enables** HIP device 0; use `off` to disable DeepSeek/GLM, or `--helper off` for MiniMax. HIP index 0 is independent of CUDA index 0. DeepSeek/GLM GB are decimal; MiniMax retains its original GiB budget. Budgets cap allocation: actual helper capacity depends on complementary experts and memory available.

GLM scope is Windows CUDA, one trunk GPU, native GGUF, dimension up to 4096, at most eight experts per route. It uses GLM's own quantized expert kernels and clamped activation. Same-backend helper/layer split/EXL3 combinations are rejected for this mode. DeepSeek scope and restrictions are in [DS4-CUDA-HIP.md](DS4-CUDA-HIP.md).

The optional all-VRAM GLM profile enables `MAYA_GLM_HIP_VRAM_ONLY=1`. The current local GLM default uses NVIDIA + AMD without MTP, mapped HIP buffers and an adaptive AMD-backed CUDA cache within a 32 GB budget. The previous NVIDIA-only profile is preserved in `bench/results/2026-10-31-glm-hip-cache/production-before.json`. NVIDIA slots and permanent AMD backing must cover **every** routed expert or loading fails. The adaptive CUDA cache can replace only entries with AMD backing, and waits for copy completion before publishing them. AMD may retain backups for disposable CUDA tail copies and selected cold-main slots; Monitor counts these duplicates once. AMD also owns the experts that would occupy NVIDIA's borrowable prompt/vision tail; lending that tail therefore cannot lose the only GPU copy. Prompt reads come from GPU storage through bounded host staging. The RAM expert tier keeps its small staging floor rather than duplicating the model. The 262K KV streaming cache, embeddings and transfer buffers still use RAM. After loading, a missing VRAM owner is an error: the mode does not silently read an expert from SSD. `expert_ssd_bytes`, `expert_hip_bytes` and `expert_cuda_bytes` expose runtime source counters (reset at READY).

With `MAYA_GLM_HIP_VRAM_ONLY=0`, the earlier complementary-pool mode retains protected RAM backing for its AMD experts. In either mode a compute failure retires the HIP worker for the process. CPU fallback uses retained backing in that older mode; in all-VRAM mode it reads expert bytes from AMD into at most eight existing staging blobs, so it still needs usable AMD storage. Reload to re-enable HIP compute. `MAYA_GLM_HIP_GRAPH=1` replays cached transfer/kernel graphs with updated inputs, routes and weights; `0` uses ordinary launches. `MAYA_GLM_GU22_TUNED=1` selects the existing RDNA expert kernel.

For controlled diagnostic requests, `strata_tune: {"hip_helper": 0}` bypasses HIP and `1` enables it. This keeps placement constant in a loaded process; all-VRAM CPU bypass includes AMD-to-host weight readback, so it does not reproduce a fresh helper-disabled memory plan. GLM's `hip_helper_fail: 1` injects a worker failure for fallback validation and requires reload afterward. These accept integer 0/1, not booleans. Use only on an idle sequential server.

For all-VRAM GLM, the expert bar separately counts NVIDIA and AMD ownership and reports no SSD-only experts. The Monitor adds the AMD card only when the loaded engine reports actual helper slots. CUDA 0 and HIP 0 have separate identities. AMD memory uses the actual adapter and allocated arena; unavailable power/temperature sensors are omitted. A helper-disabled run keeps the single-GPU layout.

## Verification artifacts

`bench/results/2026-10-31-unified/` contains real request/response/timing files and model logs. `bench/unified_backend_check.py` verifies architecture, configured context, real 16K document retrieval at start/middle/end, and reuse of the long prefix. `MAYA_BENCH_URL` selects a test port. `bench/mm3_vram_check.py` verifies native resident-expert allocation accounting with an isolated small-context load; production MiniMax remains 128K/int8.

MiniMax numerical GPU checks cover prefix restore through all KV formats, actual 128K/int8 transitions, overlapped staging and a missing-fence negative control. The shared-DLL expert checks compare arithmetic and invalid ABI calls. On the full model the staging diagnostic checked 57 MoE layers with maximum absolute error 0; a real 16K prompt measured approximately 1075 new prompt tok/s and retrieved all three codes, followed by a cached continuation. DeepSeek passed the same real-document/cache checks at 262K. These verify allocation and 16K operation; a full 262K document stress test is a separate workload and has not been run.

Controlled throughput measurements and remaining validation results are recorded in the local result summary. Routing, cache warmth and speculative acceptance affect rates; cold starts and helper bypass are labeled separately.

### GLM cross-runtime numerical boundary

The real-model load exposed a pre-existing verifier bug on segmented Windows pinned RAM: it used `base + slot * stride` rather than `RamClass::slot()`. The verifier now follows the actual segmented layout.

A separate 148-case test over real IQ2_S experts with IQ3_XXS/IQ3_S down projections isolated a legitimate FP16 scale rounding boundary: CUDA and HIP input Q8 bytes and hidden Q8 integers were identical, one FP16 scale differed by one ULP, and both down projections were bit-identical when given the same hidden Q8. Six invalid geometry/route/key/readback ABI cases were rejected; complete and partial GPU expert readbacks were byte-identical to the pack.

The load-time verifier retains the strict `1e-5` check for ordinary output comparisons. A HIP result outside it is accepted only after verifying byte-identical input Q8, hidden Q8 integers differing by at most one bin at no more than one position per 4096 hidden values, all FP16 scale/sum fields within one ULP, and down projection parity with a common hidden Q8 at the original strict tolerance. It logs each rounding case and the raw versus common-input error separately. This does not broadly relax the output tolerance. Reproduce with `glm_cross_helper_test <pack> <shared-dll>` (Windows CUDA test target); raw output is `bench/results/2026-10-31-unified/glm-q8-boundary.log`.

## GLM deployment measurements

See [GLM CUDA/HIP and MTP](GLM-CUDA-HIP.md) for the NVIDIA-only versus CUDA+HIP comparison, two/three-prediction controls, short-prompt PCIe reduction and the outstanding AMD batched-prefill compute path. All-VRAM refers to routed experts; host embeddings, KV and buffers remain.
