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
- **At the original build handoff, runtime tests were deferred at the user's request.** Compilation
  and source inspection alone did not establish behavioral parity or speed. Subsequently authorized DS4/GLM live checks
  are recorded below; the full regression suite and MiniMax runtime checks remain deferred.
- Existing CUDA/HIP helper upload diagnostics remain enabled; the previously documented intermittent GLM mismatch
  is not declared root-caused or fixed by importing upstream's separate native-helper readback change.
- GLM-5.3 744B is now included upstream, but our GGUF HIP helper's existing geometry restrictions remain;
  this merge does not qualify that new model on the mixed-vendor helper.

## Subsequently authorized live checks (2026-11-09)

The user authorized short checks after the merge. Production profiles were preserved; each model was left running
at the end of its check (DeepSeek was subsequently closed before GLM was started). No full suite or 262K/512K stress
run was performed. These are indicative live measurements, not frozen-placement old/new binary A/B results.

| Check | v1.1.57 result | Historical matching reference |
| --- | --- | --- |
| DS4, 512K, DSpark, mmap-experts off, W7800 packed pool; six 320-token code answers | 73.97 tok/s overall | 75.37 tok/s (-1.85%) |
| DS4, two code answers after a 16K reference | 83.0 and 78.3 tok/s | No matched post-document pair |
| DS4, fresh reference | 16324 tokens at 936.9 tok/s | Different corpus; not a direct prefill A/B |
| DS4, exact return after an unrelated request | 16323 reused, one read; full response 0.369 s | Exact-prefix behavior retained |
| GLM, 262K, FP16 KV, W7800, no MTP; four measured 320-token code/prose answers after four warmups | 66.16 tok/s (code 64.51, prose 67.90) | 65.82 tok/s (+0.51%) |
| GLM, two post-document code/prose answers | 60.56 tok/s | 60.51 tok/s |
| GLM, fresh reference | 16338 tokens at 2631.9 tok/s | Historical 16K rate 2608.0; different corpus |
| GLM, exact return after four unrelated requests | 16337 reused, one read; native prefill 19.9 ms, full response 0.636 s | Exact-prefix behavior retained |

Both document returns retrieved all three codes correctly. GLM also returned the correct JSON squares of 1..20.
GLM's configured startup check covered 84 sampled experts across all 42 MoE layers: maximum relative L2 error zero;
the asynchronous landing byte checks passed too. No helper failure or fallback was observed. GLM reported zero
runtime expert SSD bytes and zero speculative drafts; DS4 retained zero SSD-only expert residency. Engram I/O is
separate and is not claimed to be zero.

GLM's primary CUDA graphs remain **off**, matching its unchanged production profile and the v1.1.52 OFF reference;
HIP helper graphs remain on. The historical 71.07 tok/s primary-graph-ON result is a different configuration.

Maya reformatted GLM's JSON indentation at startup, causing the initial harness's final byte-equality assertion to
fail after inference had completed. Parsed configurations are identical. The artifact harness now checks JSON values;
no product/configuration fix or inference rerun was needed. This bookkeeping failure is distinct from model checks.

Local, untracked evidence: `bench/results/ds4-v1157-quick-2026-11-09-152117/` and
`bench/results/glm-v1157-quick-2026-11-09-152648/`. Machine-specific configurations, ZIP backups, logs and full outputs
remain local rather than being added with this documentation update.
