# Integrating upstream v1.1.57 into the Windows CUDA/HIP fork

Date: 2026-11-09. Previous local revision: `1cb3575` (v1.1.52).
Upstream release: `00655481ce2b0689ab1c2e8d67034dc5701d6511` (`v1.1.57`).
This is a merge of the published tag, not PR #879.

## Configuration and backup

The verified ZIP in `bench/results/merge-v1.1.57-2026-11-09-144612/configuration-before-v1.1.57.zip`
contains the original three model JSON files, launch/build BAT files, referenced GLM/MiniMax expert profiles,
and SHA-256 manifests. Old CUDA/helper binaries are retained beside it for local recovery; they are not added to Git.

- DeepSeek: 524288 context, DSpark on, mmap-experts off, compact AMD pool, exact conversation parking.
- GLM: 262144 context, W7800 enabled, MTP disabled, all-VRAM policy, original helper graph/cache settings and RAM slots.
- MiniMax: 262144 context, int8 KV, 65536 resident KV positions, DSpark, packed AMD pool and 24 GiB conversation budget.
- DeepSeek/MiniMax JSON files are unchanged. GLM adds only `--kv fp16`: its previous implicit FP16 format is made
  explicit because v1.1.55 changes the upstream default to INT8. No graph, speculative, routing or pool policy is enabled by this merge.
- Existing paths, sampling settings, ports, expert profiles and the common `build/maya-backend.exe` are retained.

## Conflict decisions and runtime effects

33 blocks in 13 files were read with surrounding context and resolved statement by statement.

| File / blocks | Decision and expected runtime effect |
| --- | --- |
| CHANGELOG.md / 1 | Retain all local improvements above all upstream release notes; no history is discarded. |
| CMakeLists.txt / 1 | Retain the unified executable, three-ABI HIP DLL, packed helpers and existing optional checks; also include upstream GLM-DSA registration. The new model's libraries remain available to the common backend. |
| README.md / 16 | Adopt current upstream logo, catalog, attachments, platform notes, support text and benchmark updates. Retain the CUDA/HIP feature, local DS4 benchmark, shared-backend wording, Strata credit, licensing notices and complete local configuration/measurement section. No duplicate benchmark rows. |
| include/strata/core/remote_experts.hpp / 1 | Retain HIP client state and add upstream pending-call state and start/finish declarations. Remove a duplicated atomic include. |
| include/strata/hip_compat/cublas_v2.h / 2 | Retain our ROCm 6.4 hipBLAS v2 opt-in and workspace compatibility; retain the upstream strided-batched GEMM alias outside these blocks. |
| maya.py / 1 | Keep upstream stop-on-rebuild-failure behavior/documentation and remap all three CUDA model targets to maya-backend exactly once. |
| serve/gpu_map.py / 1 | Return upstream tensor-parallel room accounting and keep our separately numbered HIP helper adapter/report function. |
| serve/server.py / 1 | Retain helper-device detection and also upstream HIP library-directory discovery. Initialization remains before telemetry consumption. |
| serve/web/monitor.js / 1 | Keep unique GPU expert ownership and physical helper-copy accounting; use upstream tierTip for ordinary tiers, suppress inappropriate RAM advice for all-GPU ownership. Avoid stale references to locals moved into tierTip. |
| src/core/glm_fast_path.cu / 4 | Keep separate HIP/CUDA ordinals, reject EXL3 on our HIP DLL, retain native EXL3/async CUDA support. Keep HIP residency logging once, preserve HIP serial dispatch and bounded readback/CPU fallback. Add an early error for native HELPER_OVERLAP with the HIP DLL so no CUDA stream/device state is touched. Keep upstream NUMA split reads, counting actual file fallback only; all-VRAM SSD prohibition remains. |
| src/core/remote_experts.cpp / 1 | Dispatch HIP compute through the DLL before native CUDA start/finish. Native enqueue/EXL3 requests explicitly reject a HIP client. Retain upstream native pending/error/stream synchronization behavior. |
| src/ds4/ds4_main.cpp / 1 | Apply the request's HIP-helper selection and record upstream request parsing time before begin_request; both independent statements occur once. The existing single READY branch remains after effective cache initialization. |
| src/mm3/mm3_model.cu / 2 | Keep helper wait/upload/add ordering and upstream opt-in ablation. Do not repeat moe_down: the local code already issues it in mutually exclusive CPU/no-CPU paths. Apply the optional linear ablation to the helper result separately before its residual add too. With ablation unset these calls are no-ops. |

## Automatically merged executed code

The automatically merged runtime paths include `START-MAYA.bat`, `cmake/hip_backend.cmake`,
`include/strata/core/glm_model.hpp`, `maya.py` outside its conflict, `serve/server.py` outside its conflict,
`serve/telemetry.py`, `src/core/glm_fast_state.hpp`, `src/core/glm_mtp.cu`, `src/core/glm_prefill.cu`,
`src/ds4/ds4_model.cu`, `src/ds4/ds4_model.hpp`, `src/mm3/mm3_kernels.cu`, `src/mm3/mm3_kernels.cuh`,
`src/mm3/mm3_main.cpp`, `src/mm3/mm3_model.hpp`, and `src/program/generate.cpp`.
The GLM slot-load fallback fix and exact incremental prompt preparation are included. Local DS4/MM3 cache,
packed-pool and memory diagnostics remain. The 27 local-only source/header/tool/check files are unchanged against
our pre-merge revision (line endings normalized for comparison).

## Validation boundary

- ZIP integrity and archived hashes checked; profile differences inspected against the original JSONs.
- Python syntax parsed for 121 changed Python files without importing/executing application or test modules;
  JavaScript syntax checked for monitor.js. No test suite was run.
- No conflict markers remain; request selection, timing and READY statements inspected for duplication/order.
- Upstream's tokenizer whitespace fixtures and CRLF JSON fixture generate existing whitespace-check warnings;
  their contents are retained to avoid changing test inputs.
- CUDA build: **passed**, CUDA 13.2 / sm_120, Release `maya-backend.exe` (exit 0).
- HIP helper build: **passed**, HIP 6.4 / gfx1100, Release `maya-hhelper.dll` (exit 0); copied into build/ for the existing profiles. All previous DLL export names, including the three family ABIs and packed DS4 exports, are retained (static PE inspection).
- **Runtime/numerical/throughput and regression tests are deliberately deferred at the user's request.** Compilation
  and source inspection do not establish behavioral parity or speed for this merge. No model/server has been started.
- Existing CUDA/HIP helper upload diagnostics remain enabled; the previously documented intermittent GLM mismatch
  is not declared root-caused or fixed by importing upstream's separate native-helper readback change.
- GLM-5.3 744B is now included upstream, but our GGUF HIP helper's existing geometry restrictions remain;
  this merge does not qualify that new model on the mixed-vendor helper.
