# Is there real VRAM headroom to raise kokoro_tts's instance count? No.

## Verdict

**No config change shipped.** PR #14 flagged `nemotron_llm`'s KV-cache
`free_gpu_memory_fraction: 0.2` as a possible source of hidden VRAM headroom
that could unblock a 4th `kokoro_tts` instance. It isn't. Verified directly
(TensorRT-LLM source + the live server's own build-time memory log, not
guessed): the KV pool is already sized close to this pipeline's real need —
at most **~100-115MB** is reclaimable by lowering the fraction further, two
full orders of magnitude short of the ~2-2.5GB a 4th `kokoro_tts` instance
needs. No live reload was attempted: the shortfall is large enough to settle
the question from measurement and source-reading alone, and a live
`nemotron_llm` reload is not risk-free on this box — this investigation's own
log-reading turned up direct evidence of a prior reload crashing the whole
Triton process under memory pressure (see section 4). Spending that risk on
an experiment already known to fall ~20x short of the target isn't a good
trade, so nothing was touched on the live server. Per the brief's own
framing, this is a valid, evidence-based "no" — matching PR #10's and PR
#14's precedent for a findings-only conclusion when the fix under
investigation turns out not to be the fix.

The bigger-picture VRAM breakdown (section 3) shows the real reason there's
no slack: `kokoro_tts` itself, not `nemotron_llm`, is the dominant and most
elastic consumer — currently **7.26GB across its 3 existing instances**
(measured live, right now), already at the empirical growth ceiling its own
`config.pbtxt` documents. That's the actual lever, and it's out of scope
here (changing `kokoro_tts`'s own memory behavior, not `nemotron_llm`'s).

## Method

- Read `triton_model_repo/nemotron_llm/1/model.py` and confirmed the live
  server is running the exact code the worktree/git history shows (NVFP4
  checkpoint, `max_batch_size=16`, `kv_cache_config={"free_gpu_memory_fraction":
  0.2}`) — no live drift from what's documented.
- Read TensorRT-LLM's own source
  (`tensorrt_llm/llmapi/llm_args.py`, `KvCacheConfig`) rather than guessing
  what `free_gpu_memory_fraction` controls.
- Cross-checked that reading against a real, live build-time log line
  captured on this exact server (`/var/log/portal/speech-cascade-triton.log`,
  from `whisper_asr`'s own TensorRT-LLM engine, which logs the identical
  mechanism): `"Memory usage when calculating max tokens in paged kv cache:
  total: 15.47 GiB, available: 0.36 GiB"` — direct proof the fraction is
  applied to *free* memory at the moment the KV pool is sized, not total
  VRAM or a fixed budget.
- Reused `docs/nemotron-batch-size-scaling.md`'s own already-measured,
  load-tested numbers for the currently-shipped `max_batch_size=16`
  config (KV pool: 0.41 GiB / 3328 tokens; `tokens_per_block: 32`) rather
  than re-deriving them — that doc's own method already validated them
  directly against `triton_launch.log` at this exact live config, and
  nothing about `nemotron_llm`'s KV sizing has changed since.
- Did **not** reload `nemotron_llm` to re-verify the number live: the
  shortfall calculated in section 2 is large enough (100MB found vs. 2-2.5GB
  needed) that a live reload could not change the conclusion, and per the
  brief's own guardrail ("if anything looks uncertain or risky, stop"), a
  reload with a known-insufficient payoff and a demonstrated real crash mode
  on this exact server (section 4) isn't a good trade.
- Got the full current per-process VRAM breakdown directly from
  `nvidia-smi --query-compute-apps` (not estimated) and matched every
  process to its model via `ps aux`, to answer the brief's "where does the
  rest of the ~14.2GB actually go" question with real numbers rather than
  the older, now-partially-stale figures in the project README.
- Confirmed the coordination lock (`RELOAD_LOCK`) was free throughout and
  never claimed, since no reload happened. Ran the full
  `tests/integration` suite as a pre-work health check (15/15 passed) — no
  post-work re-run needed since nothing on the live server changed.

## 1. What `free_gpu_memory_fraction` actually controls (verified, not assumed)

From `tensorrt_llm/llmapi/llm_args.py`'s `KvCacheConfig`:

> `free_gpu_memory_fraction`: "The fraction of GPU memory fraction that
> should be allocated for the KV cache. Default is 90%. If both `max_tokens`
> and `free_gpu_memory_fraction` are specified, memory corresponding to the
> minimum will be used."

This is exactly what PR #14's flag assumed, confirmed against the actual
engine mechanism, not just the docstring: the engine snapshots **free**
device memory at the point in its own startup sequence where it sizes the
paged-KV pool, then allocates `fraction x that snapshot`. The live server's
own log (captured during `whisper_asr`'s TensorRT-LLM build, same
mechanism, same code path) shows this exact computation happening in real
time: `"Memory usage when calculating max tokens in paged kv cache: total:
15.47 GiB, available: 0.36 GiB"` immediately followed by an allocation
line. So: correct lever, correct mechanism, sized off whatever's free at
that moment in the reload sequence (which — for a `nemotron_llm`-only
reload with `whisper_asr`/`kokoro_tts` already resident and stable — means
current system-wide free VRAM, ~1.6-2.0GB the vast majority of the time).
There's also a separate `max_gpu_total_bytes` field (an absolute byte cap,
combinable with the fraction via "whichever is smaller") — not currently
set (0 = unused), and not needed here since the fraction-based number
already turns out to be close to the real minimum (section 2).

## 2. The actual number: ~100-115MB reclaimable, not ~2GB

Reusing `docs/nemotron-batch-size-scaling.md`'s own measured numbers for
the current live config (`max_batch_size=16`, `tokens_per_block: 32`):

| Quantity | Value |
|---|---|
| Currently allocated KV pool | 0.41 GiB = 3328 tokens |
| Realistic peak concurrent need | 16 sequences (= `max_batch_size`) x ~150 tokens/turn (short voice-assistant turns, this pipeline's real traffic per the brief and prior docs) |
| ...rounded up to `tokens_per_block=32` granularity | ceil(150/32)=5 blocks = 160 tokens/sequence x 16 = 2560 tokens |
| Slack (currently allocated minus realistic need) | 3328 - 2560 = 768 tokens |
| Slack in bytes (128 KiB/token, PR #10's confirmed combined K+V figure) | 768 x 128 KiB ≈ **96 MiB** |

Even without the block-rounding conservatism (using the brief's own flatter
~2400-token estimate), the slack tops out around **300 tokens ≈ 115MB**.
Either way: **the KV pool is already sized close to real need.** The
0.41GiB actually allocated isn't "greedily claiming free VRAM" the way the
*unbounded* default (90%) would — it's TensorRT-LLM's own paged-KV block
accounting rounding a genuinely small real requirement up to block
granularity, on top of a small fixed scheduling margin. There is no ~2GB of
slack hiding in this fraction; PR #14's flag was the right thing to check,
and checking it is what rules it out with actual numbers instead of leaving
it as an open guess.

**This alone is decisive**: `kokoro_tts` needs ~2-2.5GB per instance to add
a 4th (per PR #14, confirmed again live in section 3 below at ~2.4GB).
~100MB recovered from `nemotron_llm`'s KV pool is off by roughly 20x. No
plausible further shrinkage of this one parameter closes that gap — even
`free_gpu_memory_fraction: 0.0` (disabling the margin entirely, which would
be unsafe — no slack for a burst hitting `max_batch_size=16` concurrently)
would only recover the full 0.41GiB, ~400MB, still nowhere near enough.

## 3. Bigger picture: where the ~14.2GB actually goes (measured live, right now)

Per-process VRAM via `nvidia-smi --query-compute-apps`, matched to model via
`ps aux` (steady state, all four models loaded, ambient overnight
`stress_loop.sh` traffic the whole time — same conditions PR #14 measured
under):

| Process | VRAM | What it is |
|---|---|---|
| `tritonserver` (core) | 290 MiB | Triton's own pinned/CUDA memory pools (256MiB pinned + 64MiB CUDA, both at their small defaults — not a lever) |
| `nemotron_llm` stub (relay) | 348 MiB | Triton python-backend process; just forwards to the MPI worker |
| `nemotron_llm` MPI worker (`mpi4py.futures.server`) | 4946 MiB | The real footprint: NVFP4 weights (engine 3578 MiB per `nemotron-batch-size-scaling.md`) + execution context (531 MiB) + runtime/decoder buffers (~50MiB) + KV pool (~410MiB) + ~370MiB unattributed framework/CUDA-context overhead |
| `whisper_asr` stub | 1340 MiB | TensorRT-LLM engine (this model has since moved off the README's original ONNX Runtime description onto its own TRT-LLM engine, `whisper-base-trtllm` — worth a README correction separately, not this doc's scope). Engine weights ~200MiB + execution context ~259MiB + small KV/runtime buffers (~20-30MiB, tiny — short audio, per the live build log) + framework overhead. Notably: this is now a **fixed-size compiled-engine footprint**, not a growing ONNX arena — more stable than the README's current text implies. |
| `kokoro_tts` x3 stubs | 2420 + 2418 + 2418 = **7256 MiB** | ONNX Runtime CUDA sessions. This matches `kokoro_tts/config.pbtxt`'s own documented ceiling (`"reaching ~2.4GB per instance / ~7.2GB total after repeated load testing"`) almost exactly — confirming this GPU's kokoro_tts instances are **currently sitting at that documented sustained-load ceiling**, not the ~250MB light-warmup figure. This is the dominant, elastic consumer on the box. |
| **Sum** | **14180 MiB** | vs. 14226 MiB measured `nvidia-smi` total used (46MiB gap = rounding/unattributed) |
| **Free** | **~1617-2077 MiB** | fluctuates; alert-notifier's own log shows it oscillating 1.56-2.08GiB all night |

**`kokoro_tts` alone (7.26GB) is 1.5x `nemotron_llm`'s entire footprint
(5.29GB stub+worker combined)**, and it's the one component whose memory
behavior is genuinely elastic (ONNX Runtime's CUDA arena grows under load
and doesn't shrink — the mechanism `kokoro_tts/config.pbtxt` already flags).
This is the real story behind "why is there no headroom": it isn't that
`nemotron_llm` is hoarding VRAM it doesn't need (section 2 rules that out),
it's that `kokoro_tts` has already grown, under tonight's real sustained
load, to consume nearly half the GPU across just 3 instances. A 4th
instance would plausibly grow to the same ~2.4GB ceiling under the same
traffic, which the current ~1.6-2.0GB free (even after best-case KV
trimming) cannot absorb with any safety margin — consistent with, and now
quantitatively confirming, PR #14's original caution.

## 4. Why no live experiment was run

Beyond the math in section 2 already ruling out sufficiency, reading
`/var/log/portal/speech-cascade-triton.log` in full while investigating
turned up a concrete, real crash on **this exact server**, from **this
exact reload pattern**, that's directly relevant context for anyone
tempted to try squeezing `nemotron_llm`'s KV fraction further: at
2026-08-24 20:44-20:45, a startup-time `nemotron_llm` engine build failed
outright with `CUDA error 2` (out-of-memory) on a 751MB allocation inside
TensorRT-LLM's own autotuner, while `kokoro_tts`/`whisper_asr` were loading
concurrently — memory pressure during simultaneous multi-model startup, not
a KV-cache-specific failure, but a real demonstration that this GPU's
margin is thin enough for a routine reload to fail outright under the
wrong timing. The subsequent unload attempt then hit an unrelated segfault
in the Python stub's teardown path, which **crashed the entire
`tritonserver` process** (not just the one model) — this is the "orphan
tritonserver, not supervisor-tracked" state `scratchpad/overnight/STATE.md`
already documents from earlier tonight. The server was manually recovered
by a prior agent and has been stable since (confirmed: `tests/integration`
15/15 passing right now, all 4 models `READY` via `/v2/repository/index`).
This is exactly the "OOM-class failure this project's guardrails
specifically warn about," now with a concrete precedent on this server, and
another reason (beyond the math already being decisive) not to spend a
5-6 minute JIT-build live reload chasing a ~100MB change.

## What's still open

- **The real lever is `kokoro_tts`'s own memory footprint, not
  `nemotron_llm`'s KV cache** — confirmed, not just suspected, by this
  investigation. Its `config.pbtxt` already names two candidate fixes
  (periodic instance restarts to reset the ONNX Runtime arena, or an
  explicit arena-size cap via session options) — neither attempted here,
  out of scope for a KV-cache-focused follow-up, and each carries its own
  real risk/complexity (session-option support depends on what `kokoro_onnx`
  exposes; periodic restarts need their own safe-reload orchestration).
  This is the natural next investigation if `kokoro_tts` capacity is
  revisited.
- **`whisper_asr` has migrated from ONNX Runtime to a TensorRT-LLM engine**
  (`whisper-base-trtllm`) at some point in this repo's history, but
  `README.md`'s architecture table and "Process architecture" section still
  describe it as `optimum.onnxruntime.ORTModelForSpeechSeq2Seq`. Found
  incidentally while tracing where its ~1340MiB goes; worth a
  documentation fix, unrelated to this investigation's verdict.
- A bigger GPU or reducing the model count remain the only ways to safely
  raise `kokoro_tts`'s instance count under the current per-model memory
  profile — matching this brief's own framing of that as a valid, honest
  outcome when no safe lever is found.

## Files touched by this investigation

- This document only. No config, code, or live-server changes —
  `triton_model_repo/nemotron_llm/1/model.py`'s
  `free_gpu_memory_fraction: 0.2` is untouched, live and worktree copies
  remain identical, and no model was reloaded.
