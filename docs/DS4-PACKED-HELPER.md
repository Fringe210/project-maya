# DS4 packed AMD expert pool

The Windows CUDA client now uses a compact HIP arena by default when the DLL exports `ds4h_reserve_packed`. Each selected expert has a 256-byte-aligned offset and an exact layer/size binding. The selected expert count and ranking are identical to the legacy fixed-stride plan; saved padding is left free instead of filled with more experts. DS4 expert kernels, clamped SwiGLU, routing, KV and DSpark are unchanged.

## Compatibility and controls

- Every existing version-1 C ABI export remains. The two added exports are `ds4h_reserve_packed` and `ds4h_get_memory`. An older DLL automatically uses fixed-stride reserve.
- `DS4_HIP_HELPER_PACKED=0` forces legacy allocation; unset or `1` enables compact allocation when supported. `DS4_HIP_HELPER_GB` still determines the legacy-equivalent selection budget, not a requirement to allocate that many physical bytes.
- Old clients may still reserve first and provide layouts afterward. Packed reserve requires layouts first and makes them immutable. Packed uploads/rounds reject size, layer and initialization errors before enqueue.
- Payload and padding are reported through `helper_payload_bytes`, `helper_padding_bytes`, `helper_packed` and the selection fingerprint `helper_selection`.
- Host expert copies remain for batched prefill and CPU fallback. The change frees AMD VRAM and Windows commit headroom; it does not remove those RAM copies.

## Verified on this PC, 2026-11-02

RTX PRO 6000 Blackwell + Radeon PRO W7800, DS4 Maya-S, context 524288, DSpark on, mmap-experts off, 40 GB helper selection budget:

| Item | Legacy | Packed |
|---|---:|---:|
| AMD experts | 3168 | 3168 |
| Expert selection fingerprint | 75f7d0a2cda7d8ac | 75f7d0a2cda7d8ac |
| AMD arena bytes | 39998914560 | 35104481280 |
| Actual payload bytes | 35104481280 | 35104481280 |
| Padding bytes | 4894433280 | 0 |
| Host experts copied into RAM | 7646 / 8169 | 8169 / 8169 |
| CPU-only file-backed experts | 523 (5.616 GB) | 0 |

The arena saves **4.894 GB (4.558 GiB)** without reducing AMD ownership. In this run the smaller allocation allowed every host expert copy to fit under Windows' commit guard, without altering the page file or memory reserves. Available commit also varies with other programs, so this is not a guarantee under arbitrary memory pressure. Engram tables still use their separate SSD/cache path; zero expert file backing does not mean zero system disk activity or zero Windows paging.

Validation completed:

- Common CUDA sm_120 backend and HIP gfx1100 shared DLL builds. All 29 previous exports retained; two additive exports.
- Dedicated DS4 compact and legacy tests: mixed IQ2/IQ3 roles, one/three/sixteen-token rounds, multi-layer asynchronous staging, invalid input controls, old client call ordering, byte-checked uploads. Both modes match the double reference (observed relative errors around 1e-7) and produce the same output digest, `06291b2455e76e84`.
- MiniMax compact/legacy numerical tests pass against the same shared DLL. Python setup/dashboard suite: 67 tests pass.
- Real model: both layouts retrieve all three codes from the identical 16384-token prompt. After four unrelated requests, both reuse 16383 tokens and read one; the changed-prefix control returns its new code. One READY follows effective cache INFO. Runtime numerical/fallback checks pass, normal helper_fb=0 and helper_dead=0.
- Original DS4, GLM and MiniMax profile hashes are unchanged. No splice, routing approximation or new model weights were introduced.

One sequential comparison observed prefill 980.3 -> 1005.0 tok/s, and aggregate decode 72.61 -> 75.37 tok/s across four 320-token code requests per layout. This is not a repeated/interleaved performance estimate: cache warmth, CPU-lane timing and DSpark acceptance vary. Standalone helper outputs are bit-identical; full generated answers are not asserted bit-identical across different host residency/execution plans.

## Reproduce

With the server/model stopped, run `build-unified.bat` and `tools\hip\build_helper_windows.bat`, then install `build-hip/maya-hhelper.dll` beside the CUDA backend. Build the `ds4_shared_helper_test` and `mm3_shared_helper_test` targets in build-hip and run each with default compact and `--legacy` allocation.

For full-model checks, use fresh output labels:

```powershell
.\.venv\Scripts\python.exe -X utf8 bench/ds4_packed_helper_check.py --mode legacy --label legacy-new
.\.venv\Scripts\python.exe -X utf8 bench/ds4_packed_helper_check.py --mode packed --label packed-new
```

The harness isolates conversation snapshots, shares a frozen expert-profile seed between runs, fixes the staging-ring budget, saves requests/results and shuts down its native process. It does not rewrite the production profiles. Raw evidence from this campaign is under `bench/results/2026-11-02-ds4-packed-helper/`.
