# DS4 host expert memory budget on Windows

Measured on 2026-11-02, after upstream v1.1.50 integration. The profile remains DS4 Maya-S, 524288 context, DSpark on, mmap-experts off, and a 40 GB HIP helper arena. No allocation policy, expert routing, context or Windows paging setting was changed.

The startup diagnostics now print the physical-RAM budget separately from the Windows commit budget and split host expert backing between CPU-only experts and helper-owned fallback copies. The budget calculation remains min(physical budget, commit budget); unknown/unlimited diagnostic amounts print as -1.

## Measured decision

All sizes below are decimal GB, as in the native log.

| Item | GB |
|---|---:|
| Expert copies required, including cache spare slots | 89.837 |
| Available physical RAM at the decision | 114.060 |
| Physical RAM reserve | 8.590 |
| Physical copy budget | 105.470 |
| Available Windows commit | 99.769 |
| Commit reserve (10% of physical RAM) | 13.735 |
| Commit copy budget | 86.034 |
| Copy-budget shortfall | 3.803 |

The limiting resource is Windows commit, not physical RAM. Before the host expert copies, the process already had about 146.4 GB of private committed memory while only about 5.7 GB was in its CPU working set. This is consistent with the substantial commit reservation associated with this Windows GPU runtime configuration; these counters do not mean 146 GB was physically paged to SSD. The machine currently has a manually configured C: page file, 130000 MiB initial / 135000 MiB maximum, 130000 MiB allocated. No page-file setting was changed for this measurement.

The earlier run left about 5 GB of file backing; this run left 3.795 GB. Available commit and automatically sized GPU buffers vary between starts. This is a diagnostic run, not an optimization or a controlled speed comparison.

## Ownership of host copies

| Ownership | Experts | Payload GB |
|---|---:|---:|
| CPU-only, copied into RAM | 4618 | 50.482 |
| CPU-only, file-backed | 356 | 3.795 |
| AMD-owned, copied into host RAM for fallback/prefill | 3168 | 35.109 |
| AMD-owned, file-backed | 0 | 0.000 |

The residual file-backed experts are genuine CPU-only experts, not just AMD fallback copies. File backing alone does not quantify physical SSD traffic, since Windows can cache source-file pages. All AMD-owned experts have host RAM copies in this run.

## Ways to eliminate the shortfall

The subsequent [packed AMD pool implementation](DS4-PACKED-HELPER.md) recovered 4.894 GB and successfully copied all host experts into RAM on this PC. The diagnostic measurements below remain the earlier fixed-stride baseline.

1. Add 16 GiB of actually allocated page-file capacity, while keeping the existing memory reserves. This increases the commit ceiling; the physical RAM budget already covers the required copies. Raising only the configured maximum may not be counted by the engine's conservative page-file growth estimate: C: currently has less than the 10% free-space reserve used by commit_growth. After any Windows change, verify all experts copied, cpu_file=0, sufficient physical headroom and no paging pressure under the real workload.
2. Compact the DS4 AMD arena. It currently allocates 39.999 GB for about 35.109 GB of actual expert payload, leaving about 4.89 GB of fixed-stride padding. The existing MiniMax packed-pool pattern could remove that padding without reducing the number of AMD experts. Whether the full allocation saving translates into commit headroom must be measured. This requires a DS4 helper/client ABI extension and numerical tests; it was not implemented by this diagnostic change.
3. DS4_REQUIRE_ALL_RAM=1 makes missing host allocations fatal instead of silently retaining file backing. It validates the guarantee but does not supply memory or prevent Windows paging of ordinary RAM.

## Verification

The common CUDA sm_120 backend compiled successfully. A real native smoke test loaded all 524288 KV positions in VRAM, with DSpark and the AMD helper active. Helper numerical/fallback checks passed for both expert formats. After four unrelated requests, a 4607-token document reused 4606 tokens and processed one in 56.7 ms; all three codes were correct. An early changed prefix reused zero and returned its new code. Exactly one READY followed effective cache INFO. The model exited successfully. All three production profile hashes match their pre-test values. This does not validate a full 512K-token prompt or constitute a prefill/decode speed benchmark.

Raw logs, half-second Windows memory samples and summary: bench/results/2026-11-02-ds4-memory-budget/. Smoke snapshots are isolated from production conversation slots.
