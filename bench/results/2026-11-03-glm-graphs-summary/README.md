# GLM CUDA/HIP graphs at 262K — 2026-11-03

`summary.json` records 32 measured 320-token requests (16 per mode) in four successful OFF/ON/ON/OFF blocks. `validation.json` records the component and same-instance numerical controls. Cold model loads, warmups and context checks are excluded from the decode aggregate.

The primary graph path measures 65.82 → 71.07 tok/s (+7.97%); the equal-primary-slot-capacity comparison is +7.25%. Cold 16K prefill is unchanged at about 2608 tok/s. Production profiles are unchanged.

The startup-only upload verifier was hardened between the two successful pairs; generated GPU compute kernels and the native CUDA executable did not change. Initial aborted starts and the unproven cause of their byte mismatches must not be hidden: see the full report. Runtime fallback/SSD counters stayed zero in the successful runs.

[Full report, reproduction, ownership/rounding control and upload incident](../../../docs/GLM-CUDA-GRAPHS.md).
