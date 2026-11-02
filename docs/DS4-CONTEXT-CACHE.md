# DeepSeek: exact context reuse across unrelated requests

The local DS4 profile enables `MAYA_DS4_SLOT_KEEP=1`. Shared-prefix caching and conversation parking are separate features: a kept system/tools prefix does not preserve the complete active conversation when a different request takes the model.

## Cause investigated

The existing log contained a 111183-token request that restored only a 17022-token shared prefix and reread 94161 tokens in 33.140 seconds. Four short requests had intervened after the previous long conversation. The engine defaults conversation parking to off, and the local DS4 profile did not enable it. With no parked conversation, replacing the active state leaves only shared prefixes/checkpoints that remain available. The historical request token streams were not recorded, so additional client-side history edits cannot be excluded from that individual event.

The controlled reproduction uses public repository text: a roughly 17K system prefix and 111K total prompt, four unrelated short requests, then a continuation of the long conversation. Context stays 262144, DSpark stays on and `--mmap-experts off` stays set. The same saved fixture is used before and after. Control codes at the beginning, middle and end check the restored content; a changed-prefix case must return a new code without reusing past the first differing token.

## Local profile

```json
{
  "MAYA_DS4_SLOT_KEEP": "1",
  "MAYA_DS4_SLOT_CAP": "4",
  "MAYA_DS4_SLOT_RAM_GB": "1",
  "MAYA_DS4_SLOT_DIR": "E:/project-maya/data/ds4-slots"
}
```

Values belong in `maya-ds4.json`'s `env`. The first four entries are parked in RAM subject to the 1 GiB steady-state budget. Existing behavior keeps at least one RAM slot even if that slot exceeds the byte budget; serialization can temporarily exceed it. Older entries spill to the private folder, and graceful shutdown persists states there. The engine's existing disk budget and expiry controls still apply. The local folder is excluded from Git. This stores conversation state, distinct from expert-weight placement.

Setup preserves the profile's environment entries. The engine-wide default remains opt-in; this change enables it in the tuned local profile. No prompt reordering, KV splice or resident-only routing is introduced. Exact token and picture identity checks remain in force. A genuinely changed earlier history may still require rereading its suffix.

## Diagnostics and verification

Startup reports `INFO conversation_cache=1 conversation_slots=4 conversation_ram_mib=1024` after validating the slot directory. Each prompt logs counts and the selected source (`live`, `conversation`, `shared-prefix`, `checkpoint` or `cold`), plus whether parking is enabled. These diagnostic lines contain no prompt text, token IDs, tool arguments or reasoning.

`bench/ds4_context_cache_check.py --label LABEL --keep 0|1 --url http://127.0.0.1:8081` checks the real API. It leaves DSpark active and records its draft counts. The saved fixture makes control and fixed runs comparable. Raw evidence is in `bench/results/2026-11-02-ds4-context-cache/`; the final summary records measured before/after timings and runtime policy.

## Verified result — 2026-11-02

The resumed HTTP request was identical in both runs (same saved document, history and sampling). Both runs used DSpark and returned all three document codes correctly.

| Policy | Reused tokens | Newly read tokens | Engine prefill time | HTTP response wall time |
|---|---:|---:|---:|---:|
| Conversation parking off | 17022 | 94221 | 32.7259 s | 33.808 s |
| Conversation parking on | 111214 | 29 | 0.7120 s | 1.832 s |

This one resumed-request scenario reduced prefill latency about 46× and whole-response latency about 18.5×. It is not a decode-throughput speedup. The cold initial 111185-token read remains necessary.

Negative controls passed: a changed early system prefix reused zero tokens and returned its new code. Changing the middle code first diverged at token 64591; the engine correctly reused only its earlier 17022-token checkpoint, reread the suffix and returned the changed code. Returning to the original branch then reused 111242 tokens and read one token in 90.1 ms. No post-edit KV rows were reused approximately.

The native API reports context 262144, DSpark enabled, conversation cache enabled, four RAM slots and a 1024 MiB budget. The canonical profile still has `--mmap-experts off` and the 40 GB AMD helper budget. The 12 restore-point/environment tests and two unified-launcher tests passed; the unified CUDA backend built successfully.
