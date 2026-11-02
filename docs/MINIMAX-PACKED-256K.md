# MiniMax: compact AMD pool, 256K context, 64K resident KV

Verified 2026-11-02 on RTX PRO 6000 Blackwell + W7800, using the common CUDA backend and HIP helper DLL.

## Current local profile

`maya-minimax.json` retains its existing Maya-S weights, expert profile, DSpark, vision and helper settings. It now
uses `--max-context 262144 --kv-cache int8 --kv-streaming on --kv-resident 65536 --helper on --helper-gb 40`.
Environment: `MAYA_MM3_HELPER_PACKED=1`, `MAYA_MM3_HELPER_CHECK=1`, `MAYA_MM3_SLOTS=4`,
`MAYA_MM3_SLOT_RAM_GB=24`. The last setting budgets ordinary RAM for exact parked prompt states, separately from
the existing pinned KV allocation. A full 256K snapshot is about 19.3 GiB, so the earlier 12 GiB budget would skip it.
Parked states are LRU-evicted before allocation to enforce the payload cap; one full state fits within 24 GiB.

The model actually loaded at 262144 context, with 65536 KV positions per layer resident on NVIDIA and DSpark on.
The full host KV allocation is 16,609,443,840 bytes; GPU KV/index storage is 8,194,621,440 bytes. Positions outside
the GPU ring remain available from pinned RAM. All routed experts have a GPU owner:

| Location | Experts |
|---|---:|
| NVIDIA | 4791 |
| AMD | 2505 |
| Outside both GPUs | 0 |
| Total | 7296 |

The CPU thread pool and host expert copies remain available for fallback and CUDA batched-prefill staging.
This change does not release those host copies or make every PCIe transaction disappear. Batched prefill still
computes routed experts on NVIDIA, staging nonresident weights from their retained host blobs. The larger ring
reduces attention KV misses for prompts within its range. AMD compute handles its decode/verify experts.

## Compact arena and compatibility

Previously every AMD slot used the largest expert stride, despite the pack mixing six blob sizes. The extension
`mm3h_reserve_packed` describes each slot's exact size and layer. The helper aligns offsets to 256 bytes and routes
both single-token and grouped-window kernels through those offsets. Layouts become immutable after reservation;
uploads and rounds validate slot capacity, initialization and layer ownership before any work is enqueued.
`mm3h_get_memory` reports allocation, uploaded payload and padding without changing the old stats structure.
Legacy reservation remains supported; an older DLL falls back to the fixed-stride path. The new production
profile is checked to have `helper_packed=1`. No previous shared-DLL export was removed (DS4 and GLM retain theirs).

The full-model AMD allocation is **37,891,325,952 bytes (35.29 GiB)**,
equal to the uploaded payload, with **zero padding** for this pack. It holds 524
more experts than the previous 1981-slot arena while using
4.70 GiB less physical allocation. Allocating the same
new expert count with the old largest stride would require 50.57 GiB.
All 2505 real expert uploads were read back from AMD and compared byte for byte.

No token splice, routing approximation, weight requantization or speculative-policy change was introduced.
CUDA/HIP ownership is selected by the existing routing profile; the packed plan selects candidates within the byte
budget. Host expert backing is not counted as a separate compute owner in the engine's `cpu_experts` startup field.

## Real-model checks and timings

The saved prompts contain source/documentation with three unique retrieval codes at start, middle and end.
Each prompt is followed by four short requests and then the identical prompt. Sampling is temperature 0,
seed 42, thinking off; DSpark stays active. Every retrieval and repeated-answer check passed.

| Prompt tokens | Fresh tokens in first request | First-read tok/s | Reused on return | Read on return | Return prefill including restore |
|---|---:|---:|---:|---:|---:|
| 40000 | 40000 | 1128.8 | 39999 | 1 | 0.833 s |
| 80000 | 79780 | 944.7 | 79999 | 1 | 1.064 s |
| 261001 | 220333 | 404.0 | 261000 | 1 | 4.174 s |

The near-full prompt reaches about 99.6% of the configured context. Its immutable snapshot occupied
20,588,518,496 bytes, above the old 12 GiB budget. It survived four short requests and a changed
early system prefix; returning to the original branch reused 261000 tokens. The changed prefix
returned its new code and did not reuse past the first differing token.

A subsequent code request at the same long context generated 320 tokens at
**32.30 decode tok/s**, with 189 of 238 DSpark proposals
accepted. This is one workload, not a general decode rate or a controlled comparison with the old deployment.

On the identical 40K document, the preceding 128K/32K deployment measured 981.0 first-read
tok/s; this 256K/64K deployment measured 1128.8 (+15.1%). These are single
sequential measurements, not repeated/interleaved A/B trials. Cached-return timings should not be directly compared
with the earlier diagnostic run, which additionally saved/compared the restored snapshot bytes.

## Verification and artifacts

- HIP compact and legacy helper tests passed against the numerical reference for 1/3/16-token windows, including
  mixed quantization types and invalid descriptor/upload/round negative controls.
- GPU KV tests passed for all supported format combinations and exact int8 ring restoration at 262144/65536.
- 63 Python protocol/image/setup/launcher tests passed; the unified CUDA backend and HIP DLL built successfully.
- `bench/mm3_packed_256k_check.py` runs the API checks. `bench/mm3_packed_256k_summary.py` validates saved evidence
  without inference. Config backups, raw requests/responses, logs, export comparison and hashes are in
  `bench/results/2026-11-02-mm3-packed-256k/`.

The large synthetic conversation states were evicted through the normal LRU policy after validation; only about 0.65 GiB of small test states remain. MiniMax is left idle and serving on port 8080. DeepSeek's 262K/DSpark/mmap-experts-off profile and GLM's tuned profile are
unchanged. Machine-local profile files remain ignored by Git; copy the settings above when reproducing the setup.
