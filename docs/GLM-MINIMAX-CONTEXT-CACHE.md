# Exact conversation reuse: GLM and MiniMax

The initial cache validation below used MiniMax 128K/32K on 2026-11-02. The current [256K/64K profile](MINIMAX-PACKED-256K.md) raises its snapshot budget to 24 GiB and adds a compact AMD pool. Historical measurements below retain their original settings.
No token splice, expert substitution or approximate routing was added.

## Findings and changes

GLM already parks four conversations during a running session. `MAYA_GLM_SLOT_KEEP` controls persistence across
restarts, not whether runtime parking works. Its default parking uses files in the pack's `slots` directory.
The tuned local profile now selects the existing RAM path (`MAYA_GLM_SLOTS=4`, `MAYA_GLM_SLOT_RAM_GB=4`).
This avoids conversation-cache SSD writes; GLM's attention, recurrent state and routing code are unchanged.
Startup now exposes effective cache settings before READY. Existing slot files are left untouched in RAM mode.

MiniMax's single-sequence server previously retained only the live conversation. A short unrelated request
overwrote that state, so returning to a document reread nearly all of it. It now captures an immutable prompt
anchor before generation, retaining exact K/V codes and scales, BF16 index keys and DSpark's BF16 context rings.
Each anchor covers the prompt except its last token, which is always evaluated again to regenerate logits.
The draft ring's 64 extra rows preserve its attention window despite this one-token lookahead.
Restoring a different conversation invalidates the NVIDIA ring tags and repopulates the ring from the restored
host KV bytes. There is no dequantization/requantization and no change to CUDA/HIP expert placement.

The local MiniMax profile uses `MAYA_MM3_SLOTS=4`, `MAYA_MM3_SLOT_RAM_GB=12` and the default
`MAYA_MM3_SLOT_MIN=1024`. The RAM payload budget is strict: LRU anchors are removed before a new allocation,
and an anchor larger than the budget is skipped. A full 128K int8 anchor needs about 9.7 GiB including index keys
and the draft rings. The 40K test anchor occupied 3,191,319,776 bytes. Copies use ordinary RAM, without snapshot files;
the model's existing pinned KV allocation is separate. Both profiles' parked RAM states die with the native process.
GLM's existing RAM policy can retain one oversized state; its byte budget is a target in that case.

Both paths require the entire saved token prefix and picture identities at the same positions to match.
A changed line inside that prefix legitimately prevents restoring that anchor. Earlier live/checkpoint prefixes
may still be reused when valid. LRU eviction, a disabled/insufficient budget or a process restart can also require
rereading. These mechanisms do not hide changing system text, omitted reasoning or rewritten tool histories.

The new MiniMax parking path supports the single-sequence layer-split backend used by this GGUF profile.
Tensor parallel and `--parallel` scheduling are unchanged. Four parked states do not mean four simultaneous streams
and do not divide the active context limit. Set `MAYA_MM3_SLOTS=0` or `MAYA_MM3_SLOT_RAM_GB=0` to disable it.

## Real-model verification

The saved fixtures contain 40K tokens of source/documentation with codes at the start, middle and end. Each run reads
the document, runs four short requests and submits the identical document request again. Both control/fixed runs used temperature 0, seed 42 and thinking off. Native speculation,
weights and GPU/helper settings were preserved. These are single-request latency measurements, not decode speedups.

| Model / policy | Active context | Reused on return | Newly read | Engine prefill | Whole HTTP response |
|---|---:|---:|---:|---:|---:|
| MiniMax before | 131072, int8 | 210 | 39790 | 39.159 s | 39.680 s |
| MiniMax exact RAM anchors | 131072, int8 | 39999 | 1 | 2.062 s | 2.585 s |
| GLM existing parking, RAM selected | 262144 | 40001 | 1 | 0.020 s | 0.532 s |

MiniMax's resumed prefill latency fell about 19.0x in this reproduction. Its test profile
enabled `MAYA_MM3_SLOT_CHECK=1`: after each RAM restore it saved the state again and compared **every byte** of the
KV codes/scales, index cache and DSpark rings with the original snapshot. The 2 successful
comparisons each covered 3,191,319,776 bytes. Their overhead is included in MiniMax's reported time;
the canonical production profile does not enable this diagnostic. A cold first read is still necessary and saving
an anchor adds a RAM copy (about 0.6–1.4 s in this run). MiniMax includes restore/capture in its prompt time;
GLM starts its prompt timer after slot restoration, so use the HTTP time to include its restore cost.

All three codes were retrieved correctly. The identical request returned the same answer before/after restoration.
Changing the middle code returned the new code and reused no tokens beyond the first difference:
MiniMax reused 161 before difference 21196;
GLM reused 0 before difference 21509.
Returning to the original branch then reused 39999 / 40001 tokens
(MiniMax / GLM). Continuations also passed. MiniMax kept DSpark on (100 proposals in the checked requests).
GLM kept its tuned no-MTP setting. GLM's pre-existing slot filenames, sizes and timestamps were unchanged.

Additional checks: 63 Python tests, native serial-prefix/image/LRU tests, and GPU KV prefix tests covering all
supported format combinations plus 131072-position int8 KV with a 32768-position ring passed. The unified backend
rebuilt successfully. Real model requests used 40K, not a full 262K/128K stress prompt; the advertised context sizes
were verified in the loaded models. DeepSeek's 262K/DSpark/mmap-experts-off profile remains unchanged.

## Reproduce and inspect

`bench/glm_mm3_context_cache_check.py --family glm|minimax --label LABEL --cached 0|1 --url http://127.0.0.1:8081`
runs the real API checks against an already selected profile. Use a parking-disabled profile or the saved old
binary for the MiniMax control. `bench/glm_mm3_context_cache_summary.py` checks saved evidence without inference.
Fixtures, complete responses, timings, configs, logs and summary are under
`bench/results/2026-11-02-glm-mm3-context-cache/`. Cache diagnostic lines report counts and source, without prompt
text or token IDs. Machine-local profiles are ignored by Git; copy the documented environment settings when forking.
