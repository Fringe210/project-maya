# DS4 CUDA + HIP helper on Windows

The CUDA `maya-backend.exe --family deepseek` can load `maya-hhelper.dll`, built separately with HIP, to compute routed GGUF experts on an AMD GPU. The primary keeps attention, shared experts, KV and DSpark. Activations, slot IDs, weights and output sums cross host memory; CUDA pointers never cross into HIP. The HIP worker runs beside the primary and CPU lane. Large batched prefill keeps the primary DS4 path.

Current scope: Windows CUDA primary, GGUF experts, TP1, static helper pool. EXL3, shared drafter placement, recent slots and `DS4_FLOW_HELPER` are excluded from this first cross-vendor integration. The existing same-backend `DS4_HELPER` remains separate; do not combine it with this mode.

## Build and enable

Run `tools\hip\build_helper_windows.bat` from a checkout with the Python venv and local llama.cpp dependency. It defaults to HIP SDK 6.4 and gfx1100; `DS4_HIP_SDK` and `DS4_HIP_ARCH` override these. The vendored rocWMMA 2.2 headers support the Windows SDK. Copy `build-hip\maya-hhelper.dll` to `build\maya-hhelper.dll` while the server is stopped, then rebuild the common CUDA engine with `build-unified.bat`. The same DLL also exports the GLM and MiniMax helper ABIs; see [the unified backend](UNIFIED-BACKEND.md).

Set these entries in the configuration's `env`:

```json
{
  "DS4_HIP_HELPER": "0",
  "DS4_HIP_HELPER_GB": "32",
  "DS4_HIP_HELPER_BIN": "C:/Program Files/AMD/ROCm/6.4/bin",
  "DS4_HELPER_CHECK": "1",
  "DS4_HIP_HELPER_DLL": "E:/project-maya/build/maya-hhelper.dll"
}
```

`0` selects HIP device 0 and enables it; `off` disables loading the helper. This index is independent of the CUDA device index. The DLL dependency directory must match the HIP SDK used to build it. Pool GB are decimal, as for the existing DS4 helper. Fixed-stride slots use the largest selected expert blob, including padding.

Keep `--max-context 262144 --tp 1 --dspark on --mmap-experts off`. The helper retains host copies for CPU fallback. It does not promise to free host RAM. The loader rejects incompatible combinations and ABI versions. A worker error disables the helper and routes its work back to retained CPU weights. Worker threads finish before buffers and the DLL are released.

`DS4_HELPER_CHECK=1` enables uploaded-byte readback, real-format CPU comparisons, grouped versus per-token equality and fallback checks at load. Check the log for `helper check passed` and telemetry `helper_dead=0`, `helper_fb=0` during normal runs. Some legacy check log labels say `CUDA-1` for the separate HIP backend; the preceding `HIP helper device 0` line identifies the actual backend.

For measurement only, `strata_tune: {"hip_helper": 0}` bypasses the HIP worker for one request; `1` enables it. Only integer 0/1 are accepted and the default is enabled. Request bypass keeps the static pool and its ownership placement, so it isolates execution within one loaded process rather than reproducing an unloaded-helper memory plan. The sequential server supports this diagnostic switch.

## Verified on this PC, 2026-10-31

RTX PRO 6000 Blackwell + Radeon PRO W7800, DeepSeek V4.1 Flash Maya-S GGUF, 262144 context, DSpark on, mmap-experts off:

- AMD arena: 31,994,081,280 bytes, 2,534 static slots, stride 12,625,920 bytes.
- Both IQ2/IQ3 format combinations passed; maximum absolute difference versus CPU 1.91e-6. Grouped and per-token AMD outputs were bit-identical. CPU fallback passed with retained weights.
- Two fixed prompts, 320 generated tokens each, four excluded warmups, ABBA blocks with eight measurements per condition: **58.19 tok/s bypassed, 69.39 tok/s enabled (+19.24%)**.
- Mean helper work: 23.72 expert evaluations per decoded token, residual helper wait 4.45 ms/token, zero fallback/dead events. CPU expert evaluations dropped from 26.34 to 2.48 per decoded token; these are work counts, not CPU utilization percentages.
- The original running process measured 47.57 tok/s on the same short workloads. Its cache warmth and placement differ, so that number is a reference, not an isolated helper gain.
- A real 16,385-token document returned all three distributed control codes with helper bypassed and enabled. The two passes are functional checks, not a controlled prefill benchmark. Runtime reports context 262144 and DSpark enabled. No full 262K prompt stress test was run.

Raw requests, timings, numerical-check logs and hashes are in `bench/results/2026-10-31-ds4-hip-helper/summary.json`. Repeat short measurements with `python bench/ds4_hip_helper_bench.py LABEL 0` or `LABEL 1` against an idle server on port 8080. Throughput depends on routing, warm state and DSpark acceptance.
