# GLM CUDA graphs with a HIP expert helper — 2026-11-03

## Result and configuration

On RTX PRO 6000 Blackwell 96 GB + Radeon PRO W7800 48 GB, GLM Maya-M at **262144 context**, **MTP off**, all routed experts GPU-owned:

| Workload | Primary graphs OFF | Primary graphs ON | Change |
| --- | ---: | ---: | ---: |
| Decode, aggregate | 65.82 tok/s | 71.07 tok/s | +7.97% |
| Code | 64.15 tok/s | 69.26 tok/s | +7.96% |
| Prose | 67.59 tok/s | 72.98 tok/s | +7.97% |
| Decode after the 16K document | 60.51 tok/s | 64.71 tok/s | +6.95% |
| Cold 16K prefill | 2608.0 tok/s | 2608.2 tok/s | unchanged |

OFF sets `MAYA_GLM_KDA_GRAPH=0` and `MAYA_GLM_DSA_FFN_GRAPH=0`; ON sets both to `1`. The helper's own `MAYA_GLM_HIP_GRAPH=1` stays enabled in both. This compares the full primary graph path, including KDA and the newly captured DSA FFN; it does not isolate the DSA addition from KDA capture. `MAYA_GLM_DEFER_CPU=0`: no approximate expert deferral. No MTP/DFlash drafts, expert skipping, splice or approximate routing.

Production profiles remain byte-for-byte unchanged: GLM 262K; DeepSeek 512K, DSpark on, mmap-experts off; MiniMax 256K, int8 KV with 64K resident. Primary graph flags were changed only in benchmark child environments. These results do not automatically enable primary graphs in `maya-glm.json`.

## Method and limits

Four successful reloads in OFF/ON/ON/OFF order, with the same frozen expert profile and prompt-token fixtures. Each block has four 320-token warmups and eight measured 320-token requests: Python and C++ code, plus two prose tasks. Thus each mode has 16 measured requests / 5120 output tokens. Throughput is total tokens divided by total native decode time. Model loading, warmup, JSON checks, context checks and post-long samples are excluded from the main decode average. The production server was initially stopped and remains stopped.

Each block checks valid JSON squares, the exact three codes in a real 16384-token document, return after four independent requests, and post-long decode. Every return reads one token. OFF native prefill times: 19.6 / 19.4 ms; ON: 17.9 / 18.2 ms. These are native prompt-read clocks, not end-to-end latency: restoration plus generation makes the return about 0.4 seconds in this fixture.

Automatic VRAM sizing allocated 10391 primary slots in three blocks and 10392 in one ON block; the AMD arena held 4624 experts / 39995310080 bytes, including backup copies. All **12096 unique routed experts** remained covered by the two GPUs. Responses from the equal-10391-slot OFF and ON blocks match token-for-token on all eight measured requests. The 10392-slot block's free-form responses differ, so cross-allocation bitwise answer parity is **not** claimed. Comparing the equal-capacity ON block with the two OFF blocks still gives **+7.25%** decode. A controlled single-instance logits test below separates graph arithmetic from these ownership/rounding changes.

The startup upload checker was hardened between the first and second successful pair. GPU compute kernels and the native CUDA executable did not change across the four successful blocks; the changes concern initial host transfer/check buffers. Consequently this is not four reloads of a byte-identical helper DLL. Aborted starts are preserved separately and excluded, rather than scored as throughput samples. These measurements support an observed **7–8% decode gain**, not a guarantee for every prompt or machine. Prefill does not improve, and this patch does not eliminate adaptive-cache or KV PCIe transfers.

## Correctness checks

- Graph registry: 96 repeated direct/captured steps, six retained graphs, 90 replays, no invalidations; dynamic tables, changed CPU plans, request accounting and abandoned capture pass.
- CUDA/HIP component check: 148 pairs of real GGUF experts, full and partial weight readback, portable CUDA-pinned round trip and six invalid-ABI rejections pass. Input Q8 bytes match. One hidden-activation Q8 rounding tie differs by one quantization code; the down stage with common hidden Q8 input matches exactly. No rounding tolerance was relaxed.
- GLM FFN, DSA and KV streaming self-tests pass.
- Full model, same loaded instance at 262K: **192 full-vocabulary bitwise direct/graph comparisons**, snapshot restore, reset and 48 mixed-plan finite/liveness steps pass. HIP computed 24817 experts, with no CPU fallback. Adaptive tail admission is disabled only for this numerical control, and the expert pool is capped at 84 GiB to freeze placement; this control is not included in the performance table.
- Both ON performance blocks show **45 captures / 246645 replays**; OFF blocks capture none. Context INFO is 262144, MTP INFO is 0, and every request offers zero drafts. JSON/code retrieval/cache assertions pass. No helper fallback or dead-helper state, no runtime expert SSD reads; these counters do not assert zero unrelated system disk I/O.

## Upload verification incident

Several initial native starts stopped during full expert-byte checking before READY. A diagnostic run reported a mismatch at key 4477, and a later run at key 6540. Small and full-40GB synthetic transfer probes passed, including a nonuniform probe with 90 GiB allocated on NVIDIA. Thus the probes did **not** establish the cause of the native-start failures.

`glmh_upload` now copies through a HIP-owned pinned endpoint and explicitly waits on the helper stream, matching the readback ownership rule. With byte checking enabled it keeps an immutable CPU reference, separately counts changed source bytes and GPU differences, repeats a failing GPU read and preserves the underlying HIP API error. Optional `MAYA_GLM_HIP_DIAGDIR` saves failing input/source/readback payloads locally. A mismatch still fails startup: no check was bypassed, no partial upload was admitted, and no automatic retry hides it.

The last two performance loads and the full-model logits load passed with this final checker; no diagnostic payloads were produced. This is three successful full loads, **not proof that the original intermittent cause has been identified or eliminated**. Keep the integrity check enabled and investigate further if another start reports a difference.

## Reproduction and artifacts

With Maya stopped and a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -X utf8 bench\glm_graph_check.py --out bench\results\YOUR-FRESH-LABEL --order off,on,on,off --warmup 4 --runs 8
.\.venv\Scripts\python.exe -X utf8 bench\glm_graph_summary.py bench\results\YOUR-FRESH-LABEL\results.json
```

The runner uses the existing `maya-glm.json`, creates isolated snapshots, freezes the expert profile, verifies its original config bytes and stops its own model after every block. Full-model numerical control: `glm_pack_graph_test PACK 262144 EXPERT_PROFILE`, with the same HIP/no-MTP environment, `STRATA_GLM_KV_RESIDENT=32768`, `STRATA_GLM_HIP_TAIL_CACHE=0`, and `STRATA_GLM_POOL_GB=84.0`.

Local raw evidence:

- `bench/results/2026-11-03-glm-graphs-summary/summary.json`: aggregates of the four successful blocks; `results.json` links source artifacts and checker versions.
- `bench/results/2026-11-03-glm-graphs-verified/`: first successful OFF/ON pair and the later aborted start.
- `bench/results/2026-11-03-glm-graphs-immutable/`: final ON/OFF pair with immutable checks.
- `bench/results/2026-11-03-main-1152-merge/`: builds, component tests, actual-pack logits, initial failures, original config hashes and conflict review.

## Merge review

The upstream v1.1.52 integration resolves four conflicts:

- `README.md`: retain upstream documentation/translations and the local CUDA/HIP introduction, compact-pool links, reference-PC benchmark, full unified-backend section and Strata attribution. Documentation only.
- `VERSION`: use upstream 1.1.52 so the reported version matches the integrated code.
- `include/strata/core/glm_model.hpp`: keep local `decode_mtp(..., draft_limit=0)` and add upstream `decode_spec_pipe` once; the local definition/callers retain their draft limit while split decode gains its declaration.
- `src/ds4/ds4_model.cu`: retain separate local physical/commit allocation limits and logging, replacing both removed `kRamMargin` uses with upstream `ram_margin()`. One budget decision/log; the default remains 8 GiB and the upstream override is respected.

CUDA and HIP builds pass. Runtime checks in this resumed round cover GLM. The earlier broad Python run was interrupted at the user's request before a suite result (it contained failure markers); it is not recorded as a pass. DS4, MiniMax and the complete server/installer suites were not rerun after that instruction. The GPU tests here do not certify those other paths.
