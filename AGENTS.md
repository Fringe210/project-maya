# Project Maya — local notes (this PC)

## Settings to keep

- Engine: DeepSeek-V4.1-Flash-1017 **Maya-S**, native pack in
  `E:\Maya-data\models\DeepSeek-V4.1-Flash-1017-GGUF\Maya-S\maya-ds4`.
- `E:\project-maya\maya-ds4.json` (ignored by git) must contain:
  - `--dspark on`
  - `--mmap-experts off`
- On this PC (128 GB RAM, Windows) `--mmap-experts auto` keeps the CPU lane's experts in the files: decode ~13.9 tok/s.
  With `off` they are copied into RAM (~92.2 GB, 8383/9393 experts) and decode rises to ~33.7 tok/s.
- Do not drop `--mmap-experts off` during update/setup. `maya.py` (`keep_args`) preserves it, but check it is
  still there after every setup.
- Configuration backup: `E:\Maya-data\maya-ds4-config.backup.json`.
  If a setup rewrites `maya-ds4.json`, restore from here (or add `--mmap-experts off` back into `args`).
- Build: `cmd /c build-maya.bat` (CUDA 13.2, arch 120, local `third_party/llama.cpp`).
  `build/MAYA-BUILD.json` can be stale: `maya.py` refreshes it at startup.
- Last verified: v1.1.43 (`acdab34bd58e`), 2026-10-31.
  320 tokens generated in 9.5 s of decode (33.7 tok/s), 92.2 GB of experts copied into RAM.

## Branch merges and conflicts

1. Read each conflict block **in full**, without truncated previews, plus 15 lines of context before and after in
   the already merged file. If a preview we generated is cut short, print it again in full: do not conclude on
   something we have not read entirely.
2. Resolve **statement by statement**, not block by block. An empty side of ours does not mean "take theirs": a
   block with N lines added on one side can hold independent statements, and the merge also keeps ours that sit
   right after it.
3. For every statement kept, say what it does at runtime (effect, order, output emitted, who consumes it) and
   check it is not duplicated in the merged file: `printf`/`READY`, protocol messages, initialisations, repeated
   side effects.
   Real case (`src/ds4/ds4_main.cpp`, the v1.1.50 merge): in block 742-749 keep only `INFO logit_bias=1` and the
   `--parallel` branch, while the `READY` pair stays in the following position our code already uses; taking both
   sides emits `READY` twice, the first time before the cache is initialised.
4. "Cleanly auto-merged" does not mean verified. List the files merged automatically that touch executed code
   (`maya.py`, `serve/server.py`, the kernels in `src/core/*`, `src/ds4/*`, `src/mm3/*`) and, for those, compile
   and run the tests that exercise them. When a check is not possible, say so instead of assuming it passed.
5. In the final report, for every conflict: which side was kept for each statement, why, and the runtime effect
   expected.
