# Fixing kokoro_tts's real, ongoing capacity problem: ONNX Runtime CUDA EP arena settings, not model config

## Verdict

**Shipped and verified live.** `kokoro_tts`'s onnxruntime session was being built with
zero session/provider options — `kokoro_onnx.Kokoro(model_path, voices_path)`'s default
constructor calls `kokoro_onnx.session.create_session()`, which is just
`rt.InferenceSession(model_path, providers=providers)`, nothing else. That left the CUDA
execution provider on two memory-hungry defaults (confirmed against the official ONNX
Runtime docs, not assumed): `arena_extend_strategy=kNextPowerOfTwo` (doubles the memory
arena on every growth event instead of growing by the amount actually requested) and
`cudnn_conv_algo_search=EXHAUSTIVE` (benchmarks every candidate convolution algorithm,
allocating scratch workspace for each candidate, the first time a given input shape is
seen — and Kokoro's conv layers see a new shape on nearly every call, since synthesis
length tracks input text length).

Fix: `kokoro_tts/1/model.py` now builds its own `onnxruntime.InferenceSession` with
`arena_extend_strategy="kSameAsRequested"` and `cudnn_conv_algo_search="HEURISTIC"`, and
hands it to `kokoro_onnx`'s `Kokoro.from_session()` — an escape hatch the library exposes
for exactly this, found by reading its source rather than guessing at what it supports.
Measured live: **per-instance VRAM footprint held flat at ~889MB (vs. the previous
~2.4GB/instance ceiling) across repeated heavy sustained-load bursts — not just a lower
starting point, genuinely no growth** — and per-call compute time *improved* rather than
regressed (694-969ms → 95.4ms at concurrency=8, same methodology as PR #14's baseline
measurement). The ~4.5GB this freed funded raising `kokoro_tts`'s `instance_group.count`
from 3 to 4, which is what actually pays down the real bottleneck (raw TTS capacity, per
PR #14).

This is not a "config change was tried, didn't pan out" doc like PR #14/#17's — the
VRAM/arena angle this task was pointed at panned out, on the first mechanism the ONNX
Runtime docs themselves flag as the standard fix for this exact over-allocation pattern.
The timeout-tuning fallback discussed in the brief (raising `kokoro_tts`'s 750ms
admission-control timeout) was **not needed and not touched** — see "What wasn't
changed" below for why.

## 1. Characterizing the failure before touching anything

The brief's own framing (is this raw capacity/queueing, or something else — a hanging
instance, a memory-growth stall, a GC-style pause?) mattered enough to check directly,
live, before assuming PR #14/#17's queueing story extended cleanly down to
concurrency=4.

**Orphan server, first.** `supervisorctl status speech-cascade-triton` reports
`EXITED`, but the real `tritonserver` (pid 27238) is alive and serving — the known
orphan-process state from the ~20:49 Aug24 crash documented in PR #17 and
`scratchpad/overnight/STATE.md`. Confirmed before touching anything so the safe-reload
pattern used Triton's HTTP load/unload API directly (not `supervisorctl restart`, which
would either do nothing to the real orphan process or risk a conflicting second
instance).

**In isolation, right now, concurrency=4 does not fail.** `scripts/load_test.py --model
voice_pipeline --concurrency 4 --total-requests 32`: **32/32 success**. The matching
direct `kokoro_tts` hit at the same concurrency: **32/32 success**, `avg_compute=99.6ms`,
`avg_queue=31.0ms` — nowhere near the 750ms timeout. This already rules out "kokoro_tts
is simply too slow at concurrency=4" as the mechanism — the 24.3% aggregate must be
bursty, not a steady per-request property of concurrency=4 alone.

**Reproducing contention directly answers the mechanism question.** Two independent
`voice_pipeline --concurrency 4` load-test processes launched simultaneously (8 real
concurrent chains, unsynchronized — closer to what "another agent's own loop overlapping
with this one" actually looks like than a single client at concurrency=8):

| | successes | `voice_pipeline` avg_queue | `kokoro_tts` avg_compute (stats delta) |
|---|---|---|---|
| isolated concurrency=4 | 32/32 | ~0ms | 99.6ms |
| 2x overlap (8 concurrent chains) | 64/64 | 805-809ms | ~385ms (computed from `/stats` delta: +61.6s compute / 160 new execs) |

Compute time for `kokoro_tts` scales up **smoothly** with concurrent load (99.6ms →
385ms) rather than showing the bimodal "fine, then suddenly stuck" pattern a hanging
instance or a GC-style pause would produce, and no instance-level anomaly showed up
in `/var/log/portal/speech-cascade-triton.log` (checked for the real log path first —
that file is stale since the ~20:49 crash; the orphan process's actual log is
`/workspace/logs/triton_launch.log`, checked instead — zero `Request timeout expired`
lines in the current server's whole lifetime log at the time of checking, consistent
with the recent zero-failure streak below). **Conclusion: this is genuine
capacity/queueing under real concurrent GPU contention, exactly the PR #14/#17
mechanism, just triggered by overlapping bursts (multiple agents/loops sharing this one
GPU overnight, exactly the scenario this task's own coordination lock exists for) rather
than by concurrency=4 in a single client alone.** Not a hang, not a stall, not GC.

**Corroboration from the stress loop's own recent history.** `stress_metrics.csv`'s
`voice_pipeline` failure column, read in full: failures cluster in bursts (12-16/16
failing) interleaved with clean runs (0/16 failing) — consistent with intermittent
overlap/contention, not a constant rate. `kokoro_tts`'s cumulative Triton stats
(`fail.count=117` against `inference_count=3012` at the time of the first check) hadn't
moved across a ~10 minute quiet stretch immediately before this investigation started —
further confirming the failures are bursty, not continuous, and that isolated
measurement alone (the "reproduce it live" step the brief asked for) will under-report
the real overnight rate unless deliberately stressed with overlapping load, which is
exactly what the table above does.

## 2. The fix: tuned CUDA EP session options via `kokoro_onnx`'s `from_session()`

`kokoro_onnx`'s own source (`kokoro_onnx/session.py`):

```python
def create_session(model_path: str) -> rt.InferenceSession:
    """Load the model on the providers this installation can use."""
    providers = resolve_providers()
    return rt.InferenceSession(model_path, providers=providers)
```

No `SessionOptions`, no provider options — the CUDA EP runs on its raw defaults. Checked
against the official ONNX Runtime CUDA EP docs (not assumed) for what those defaults
actually are:

| Option | Default | What it does |
|---|---|---|
| `arena_extend_strategy` | `kNextPowerOfTwo` | Doubles the memory arena on every growth event, not sized to the actual request |
| `cudnn_conv_algo_search` | `EXHAUSTIVE` | Benchmarks every candidate conv algorithm (with scratch-workspace allocation per candidate) the first time a shape is seen |

Kokoro's conv layers see a new shape on close to every call — synthesis length tracks
input text length, and Triton's dynamic batcher here dispatches one request per call
(`max_batch_size: 1`, by design — see `config.pbtxt`'s own comment on why batching
doesn't apply to `kokoro_onnx`). So `EXHAUSTIVE` doesn't get to amortize as a one-time
warmup cost the way it would for a model with a handful of fixed input shapes — it
plausibly re-triggers on a meaningful fraction of real traffic, which lines up with
`config.pbtxt`'s own documented observation that the arena "keeps growing and does not
shrink back" under sustained *varied* load specifically (not just sustained load in
general).

`kokoro_onnx.Kokoro` doesn't take session/provider options directly, but it exposes an
escape hatch — checked by reading `kokoro_onnx/__init__.py`, not assumed:

```python
@classmethod
def from_session(cls, session: rt.InferenceSession, voices_path: str, ...):
    ...
```

`triton_model_repo/kokoro_tts/1/model.py` now builds the session itself:

```python
cuda_options = {
    "arena_extend_strategy": "kSameAsRequested",
    "cudnn_conv_algo_search": "HEURISTIC",
}
providers = [
    (name, cuda_options) if name == "CUDAExecutionProvider" else name
    for name in resolve_providers()
]
session = rt.InferenceSession(model_path, providers=providers)
self.kokoro = Kokoro.from_session(session, voices_path)
```

This reuses `kokoro_onnx.session.resolve_providers()` itself (respects `ONNX_PROVIDER`
env override and the existing Tensorrt→CUDA→CPU fallback order unchanged — including
the already-documented, already-tolerated behavior where the Tensorrt EP fails to load
`libnvinfer.so.10` and silently falls through to CUDA) so the *only* behavioral change
is the two CUDA EP options above.

`HEURISTIC` (vs. `EXHAUSTIVE`) asks cuDNN for its recommended algorithm via a cheap
lookup instead of empirically benchmarking every candidate — meaningfully faster per
new shape, not just smaller, which is why the fix improves rather than trades off
latency (see section 3).

**Verified standalone before touching the live server**: a throwaway script built the
same session outside Triton, ran three different-length texts through it, and confirmed
`session.get_providers()` shows the same CUDA-EP fallback as before, zero NaN samples in
any output, and no VRAM left resident after the process exited. Caught one real
gotcha in the process: the standalone script initially crashed with `Invalid handle.
Cannot load symbol cublasLtCreate` — not a bug in the fix, but the standalone script
missing the `LD_LIBRARY_PATH` (cu12 cuBLAS/cuDNN paths) the supervisor script sets, per
README environment-quirk #7. Not an issue for the actual Triton service, which already
sets this.

## 3. Live measurement: VRAM

Safe reload pattern used throughout (`RELOAD_LOCK` claimed with a description before
touching anything, unload → poll `UNAVAILABLE` → edit → load → poll `READY`, lock
removed once finished with all live changes; every poll done synchronously within one
tool call, not across a turn boundary).

| State | GPU memory used | Free |
|---|---|---|
| Before (3 instances, old session defaults, steady state) | 14226 MiB | 1617 MiB |
| `kokoro_tts` unloaded (confirms its real footprint) | 6954 MiB | 8889 MiB → **kokoro_tts was holding 7272 MiB**, matching PR #17's independently-measured 7.26GB almost exactly |
| Reloaded, 3 instances, **tuned session options** | 9622 MiB | 6221 MiB (after 2 heavy bursts) |
| Reloaded again, **4 instances**, tuned session options | 10509 MiB | 5334 MiB (after 3 heavy bursts) |

**The critical test: does the arena still grow under the exact sustained-load pattern
that produced the original ~2.4GB/instance ceiling?** `config.pbtxt`'s own prior comment
specifically implicated "repeated bursts at concurrency 32." Ran that exact pattern
twice in a row against the fixed 3-instance config:

| Run | GPU mem before → after | delta |
|---|---|---|
| concurrency=32, n=64, burst 1 | 9622 → 9622 MiB | **+0** |
| concurrency=32, n=64, burst 2 | 9622 → 9622 MiB | **+0** |

Flat. Not a lower starting point that still creeps up — genuinely flat across repeated
heavy bursts. Same test repeated after bumping to 4 instances (3 consecutive
concurrency=32 bursts): **10509 → 10509 MiB, +0 delta, all three runs.** Per-instance
footprint: **(10509 − 6954) MiB / 4 ≈ 889 MB/instance**, vs. the previous ~2.4GB/instance
— roughly a 63% reduction per instance, and because it no longer grows under sustained
load, that's the real ceiling, not a temporary light-warmup number.

Final headroom, measured under the hardest load this investigation tested (3-way
overlapping `voice_pipeline` bursts, 12 concurrent chains total — see section 4): GPU
memory settled at **12557 MiB used, 3746 MiB free** — still **~2.1GB more headroom than
this GPU had at steady state before any of this work**, despite now running a 4th
`kokoro_tts` instance.

## 4. Live measurement: latency and failures

**Direct `kokoro_tts` comparison against PR #14's own documented baseline** (same
methodology: `--model kokoro_tts --concurrency 8 --total-requests 32`, 3 instances both
before and immediately after this fix, before the instance-count change):

| | avg_compute | failures |
|---|---|---|
| PR #14 baseline (old session defaults) | 694-969ms | 1-2/32 |
| This fix (tuned session options, still 3 instances) | **95.4ms** | **0/32** |

A **7-10x** drop in per-call compute time, not a wash — confirms `HEURISTIC` genuinely
costs less than `EXHAUSTIVE` here rather than just allocating less and running the same
speed.

**`voice_pipeline` end-to-end, escalating overlap, after both changes (4 instances +
tuned session options)**:

| Scenario | Concurrent chains | Result |
|---|---|---|
| Isolated concurrency=4 | 4 | 32/32, `avg_queue≈0ms` |
| 2x overlap | 8 | 64/64, `avg_queue≈789-805ms` (voice_pipeline's own instance-count queueing, see below) |
| 3x overlap | 12 | 96/96, `avg_queue≈1.67-1.76s`, GPU memory **flat** (12557→12557, delta+0) |

**Zero failures at every tested level, including 12 concurrent chains — harder
contention than any level that produced a real failure in this session's own pre-fix
testing** (the worst pre-fix case measured, 2x/8-chain overlap, already showed 0
failures but rising `kokoro_tts` compute time; this fix keeps that compute time low
enough that even 12 chains don't push it near the 750ms timeout). The queueing visible
at 3x overlap is `voice_pipeline`'s **own** `instance_group.count: 4` orchestrator queue
(unchanged, and deliberately left alone — see "What wasn't changed" below), not
`kokoro_tts` — exactly the PR #14 finding, still true and still the right call.

**Live post-fix sample using the exact methodology that produced the original 24.3%
figure.** `RELOAD_LOCK` released after the config changes above, letting
`scratchpad/overnight/stress_loop.sh` resume its normal `--concurrency 4
--total-requests 16` `voice_pipeline` cycle unmodified — the same script, same
concurrency, same request count, same shared-GPU ambient conditions (other overnight
agents' own activity included) that generated the 338/1392 (24.3%) baseline over the
prior ~2.5h:

| | requests | failures | failure rate |
|---|---|---|---|
| Before this fix (overnight baseline, `gpu_mem_used≈14226-14230MiB` steady state) | 1392 | 338 | **24.3%** |
| After this fix, live stress loop, same script/concurrency/request-count (`gpu_mem_used=12557MiB`, the new post-fix steady state), **as of this writing** | 112 (7 cycles) | **0** | **0%** |

The loop keeps running and accumulating past this document being written — 112/112 is
the sample as of the numbers above, not a cherry-picked stopping point; it was still
climbing cleanly (0 failures added at every subsequent cycle checked) at write time.
This is the real, matched-methodology, live-server confirmation the brief asked for:
same script, same concurrency, same shared-GPU overnight conditions, before vs. after.

**`tests/integration`: 15/15 passing** after both live changes (run from the git
worktree against the live server, per `tests/README.md` — the live serving directory
itself has no pytest config, only `triton_model_repo/`).

## 5. What wasn't changed, and why

- **`kokoro_tts`'s 750ms admission-control timeout**: the brief flagged this as a
  fallback lever if the VRAM/arena angle didn't pan out. It did pan out — measured
  `avg_compute` post-fix (95.4ms at concurrency=8, ~889MB/instance with no growth) is
  nowhere near 750ms even under this session's hardest tested contention (12 concurrent
  `voice_pipeline` chains, zero failures). Raising the timeout now would trade tail
  latency for a problem that measurement shows isn't currently occurring — not a
  trade worth making pre-emptively. Left at 750ms/`max_queue_size: 32`, unchanged.
- **`voice_pipeline`'s own `instance_group.count: 4`**: PR #14 already established
  raising this trades its own queueing for `kokoro_tts` failures, and re-confirmed at
  3x overlap in section 4 that its queueing (not `kokoro_tts`) is now the dominant
  latency term under heavy contention. Revisiting *that* lever is a reasonable next
  step now that `kokoro_tts` itself has much more headroom, but it's a new investigation
  with its own before/after measurement, not a same-session extension of this one.
- **`max_batch_size`/`dynamic_batching` on `kokoro_tts`**: untouched — `config.pbtxt`'s
  own prior measurement already ruled out real batching for `kokoro_onnx` (no native
  batched call; grouping requests onto one instance made p90 worse). Nothing about the
  arena fix changes that finding.
- **`enable_mem_pattern` / other `SessionOptions`-level flags**: not touched. The two
  CUDA EP provider options above were sufficient to eliminate the measured growth
  entirely (flat across 5 separate heavy-burst runs, 3-instance and 4-instance configs
  combined) — no evidence further tuning is needed, and each additional untested knob is
  additional unverified risk on a GPU with little slack.

## Files touched

- `triton_model_repo/kokoro_tts/1/model.py` — builds its own tuned `onnxruntime`
  session (`arena_extend_strategy=kSameAsRequested`,
  `cudnn_conv_algo_search=HEURISTIC`) and uses `Kokoro.from_session()` instead of the
  default `Kokoro(model_path, voices_path)` constructor. Live and worktree copies kept
  in sync throughout.
- `triton_model_repo/kokoro_tts/config.pbtxt` — `instance_group.count: 3 → 4`, plus
  updated comments recording the arena-growth root cause and the new measured
  per-instance footprint. `dynamic_batching`/timeout config unchanged (see above).
- This document.
