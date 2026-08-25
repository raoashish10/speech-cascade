# voice_pipeline's queueing at concurrency=8: what it actually is, and why raising instance_group.count doesn't fix it

## Verdict

**Don't raise `voice_pipeline`'s `instance_group.count` above its current value
of 4. No config change is shipped here.**

The premise carried into this investigation (`docs/nemotron-batch-size-scaling.md`
section 5, echoed in PR #11's description) was that `voice_pipeline`'s
`avg_queue=1686ms` at concurrency=8 comes from "its own single-instance
orchestrator queueing." **That premise is wrong, and checking it first is the
actual finding here.** `voice_pipeline` has run with `instance_group.count: 4`
since long before PR #11 was written -- confirmed both in git history and on
the live server. The single-instance bug that sentence describes was real, but
it was already found and fixed (see the README's "Two real bugs this
surfaced" section), and PR #11's text is a stale, unverified carry-over of
that old description, not a fresh check of the current config.

The queueing itself is real and reproducible (confirmed 3x tonight, numbers
below, still ~1.7s), but its actual cause is simpler: **concurrency=8 is 2x
`voice_pipeline`'s instance count of 4**, and each full ASR->LLM->TTS chain
through one instance takes ~2s, so at concurrency=8 roughly half of all
requests spend about one full turn waiting for a free orchestrator instance.
That part of the fix is obvious -- raise the count. It was tried (6 and 8,
live, VRAM-verified free as expected) and **made things measurably worse**:
`voice_pipeline`'s own queue time did drop to ~0, but request failures jumped
from ~3-6% to as high as 53%, because the queueing at `count: 4` was
inadvertently protecting `kokoro_tts` -- which has only 3 GPU instances and a
750ms admission-control timeout, and is *already* marginal at this
concurrency even without touching `voice_pipeline` at all. Removing
`voice_pipeline`'s own queue just lets more concurrent chains pile onto
`kokoro_tts` at once, and it starts timing out instead of Triton queueing
gracefully. This is a real, valid "raising it doesn't help" result, per the
brief's own framing of that as a legitimate outcome.

A genuine, unrelated bug in the load-testing tooling itself was also found
and fixed while chasing accurate downstream numbers for this investigation
(details in its own section below) -- `scripts/load_test.py` was silently
reporting `avg_compute=0ms`/`avg_queue=0ms`/`execs=0` for `whisper_asr` in
every run, a metrics-parsing bug, not a serving-side issue.

## Method

- Confirmed the coordination lock (`RELOAD_LOCK`) was free before touching
  anything, and re-checked before every reload (an unrelated overnight
  soak-test loop, `stress_loop.sh`, was running concurrently the whole time --
  see "Ambient load" note below).
- Read `triton_model_repo/voice_pipeline/config.pbtxt` and `1/model.py`
  directly rather than trusting the doc's description -- this is what
  surfaced `instance_group.count: 4`, not 1.
- Baseline: `scripts/load_test.py --model voice_pipeline --concurrency 8
  --total-requests 32`, run 3 times (once before touching anything, once
  mid-sweep, once after reverting), to confirm the queueing is real and
  reproducible, not a one-off measurement artifact.
- Built a small wrapper (`all_model_metrics.py`, kept in this investigation's
  scratch dir, not shipped) that diffs Triton's own `/metrics` for **all
  four** models around each `voice_pipeline` run, not just the one under
  test -- this is what let the bottleneck actually be traced to `kokoro_tts`
  rather than staying a black box inside `voice_pipeline`'s own "compute"
  time (which includes time spent blocked on nested BLS calls).
- **Safe reload pattern used for every change**: `RELOAD_LOCK` written first
  (contents: what/why/expected duration), unload, poll
  `/v2/repository/index` until `voice_pipeline` reports `UNAVAILABLE`, edit
  `config.pbtxt`, load, poll until `READY`, lock removed immediately after.
  `voice_pipeline` holds no model weights, so every reload in this
  investigation completed in a few seconds, not the ~5-6 minutes
  `nemotron_llm` reloads cost.
- VRAM checked via `nvidia-smi --query-gpu=memory.used,memory.free` before
  and after every reload (not just estimated).
- After reverting to the original config, ran the full
  `tests/integration` suite (`/venv/main/bin/python -m pytest
  tests/integration -v`) to confirm no regression.

**Ambient load note**: a separate overnight stress/soak loop
(`stress_loop.sh`, another agent's legitimate background work per this
session's coordination convention) was running the entire time, firing its
own `load_test.py` calls against all four models on a ~90s cycle. This adds
noise to any single run's numbers but does not change the conclusions below
-- if anything it makes the `kokoro_tts` contention finding more
representative of real overnight conditions, not less. Multiple repeated
runs are reported specifically to average out this noise rather than
over-reading one measurement.

## 1. Confirming the queueing is real (and where it actually sits)

`voice_pipeline`, concurrency=8, `instance_group.count: 4` (current/original
config), 3 separate runs:

| run | client p50 | client p90 | successes/32 | server avg_compute | server avg_queue |
|---|---|---|---|---|---|
| 1 (pre-change baseline) | 3.967s | 4.276s | 31 | 1990.4ms | 1728.5ms |
| 2 (mid-sweep, with per-model breakdown) | 3.818s | 4.528s | 30 | 2017.5ms | 1723.6ms |
| 3 (post-revert confirmation) | 3.996s | 4.344s | 31 | 2012.2ms | 1758.6ms |

This closely reproduces PR #11's originally-recorded numbers (p50=4.044s,
avg_queue=1686ms) -- reproducible **despite** `nemotron_llm` itself having
changed substantially since that measurement (NVFP4 quantization, decoupled
streaming, a different chat template and `max_tokens`; see the config diffs
in this project's git history). `nemotron_llm` alone at the same
concurrency=8 measures p50=0.732s with `avg_queue=1.5ms` today -- essentially
unchanged from PR #11's own note that the LLM itself isn't the problem. The
queueing is structural to `voice_pipeline`'s own scheduling, not sensitive to
what's happening inside the LLM stage.

Per-model Triton-side breakdown for run 2 and run 3 above (all four models,
diffed around the same `voice_pipeline` load-test window):

| model | run | execs | server avg_compute | server avg_queue | failures |
|---|---|---|---|---|---|
| whisper_asr | 2 | 31 | 105.3ms | 48.7ms | 0 |
| whisper_asr | 3 | 23 | 84.0ms | 56.0ms | 0 |
| nemotron_llm | 2 | 36 | 0.5ms | 0.3ms | 0 |
| nemotron_llm | 3 | 32 | 0.5ms | 0.3ms | 0 |
| kokoro_tts | 2 | 46 | 694.3ms | 52.2ms | 2 |
| kokoro_tts | 3 | 31 | 968.9ms | 38.7ms | 1 |
| **voice_pipeline** | 2 | 32 | **2017.5ms** | **1723.6ms** | 0 |
| **voice_pipeline** | 3 | 32 | **2012.2ms** | **1758.6ms** | 0 |

(`whisper_asr`/`kokoro_tts` exec counts don't match `voice_pipeline`'s 32
one-for-one because the ambient `stress_loop.sh` traffic also hits them
directly during the same window -- see the ambient-load note above.)

**`voice_pipeline`'s own `avg_queue` (~1.7s) dwarfs every downstream model's
own queue time (all under 60ms) and is the single largest component of the
~3.8-4.0s client p50.** This confirms the queueing is real, large, and sits
specifically at `voice_pipeline`'s own Triton-level instance scheduling --
exactly consistent with 4 instances being too few for 8 concurrent chains,
and not with anything happening inside the calls it makes.

One more thing worth flagging even at the untouched baseline: `kokoro_tts`'s
own `avg_compute` (694-969ms) is already close to its own 750ms
admission-control timeout, with 1-2 failures per 32 requests -- **before any
change was made here.** This turns out to matter a lot for the next section.

## 2. Raising instance_group.count: VRAM is free, reliability is not

`voice_pipeline` holds no model weights of its own (`KIND_CPU`, pure
orchestration -- see `README.md`'s "Process architecture" section), so the
README already claimed extra instances are "nearly free." That claim was
previously verified going from 1 to 4; this investigation re-verified it
going further, to 6 and 8:

| `instance_group.count` | GPU mem used | stub processes | (VRAM delta from baseline) |
|---|---|---|---|
| 4 (baseline) | 14230 MiB | 4 | -- |
| 6 | 14230 MiB | 6 | +0 MiB |
| 8 | 14230 MiB | 8 | +0 MiB |
| 4 (reverted) | 14230 MiB | 4 | +0 MiB |

**Confirmed: VRAM cost is genuinely zero at every tested count, exactly as
claimed.** This is not the constraint.

But end-to-end behavior at concurrency=8, swept across `count`:

| `instance_group.count` | client p50 | successes/32 | failures/32 | vp avg_queue | vp avg_compute | kokoro_tts avg_compute | kokoro_tts failures |
|---|---|---|---|---|---|---|---|
| 4 (baseline) | 3.8-4.0s | 30-31 | 1-2 | ~1.7s | ~2.0s | 694-969ms | 1-2 |
| 6 | 2.920s | 25 | 7 | 616.5ms | 2084.5ms | 1054.9ms | 7 |
| 8 | 1.925s | 15 | **17** | 0.1ms | 1966.6ms | 1031.2ms | **17** |

The pattern is monotonic and unambiguous: raising `voice_pipeline`'s instance
count shortens `voice_pipeline`'s own queue almost linearly (1.7s -> 0.6s ->
0.1ms) and lowers the *successful*-request p50 -- but the number of requests
that fail outright rises just as fast, from ~5% to over half. Every single
failure in every run is the same error: `kokoro_tts failed: Request timeout
expired`.

**The mechanism**: `voice_pipeline`'s own queue at `count: 4` was
functioning as unintentional backpressure -- it caps how many full pipeline
chains can be simultaneously mid-flight, which in turn caps how many
concurrent calls ever reach `kokoro_tts` (3 GPU instances) at once. Raising
`voice_pipeline`'s count removes that cap: at `count: 8`, up to 8 chains can
be mid-flight together, all racing to call the same 3 `kokoro_tts`
instances, which pushes `kokoro_tts`'s own compute time past its own 750ms
timeout for a large fraction of calls. The client-side p50 numbers above look
better only because they're computed over a shrinking set of *lucky*
survivors -- not because the system is actually serving more load correctly.
This is precisely the "downstream models become the new bottleneck instead"
outcome the brief flagged as an equally valid possible finding.

`whisper_asr` (1 GPU instance, but with real request-batching via
`dynamic_batching`) did not show the same effect -- its own queue/compute
stayed modest (see section 1's table) across every count tested, because
Triton's dynamic batcher absorbs concurrent single-instance load far better
than `kokoro_tts`'s per-request-serial dispatch does (see `kokoro_tts`'s own
`config.pbtxt` comment on why it can't use real batching -- `kokoro_onnx` has
no batched call).

## 3. Why 4 (i.e., why not ship anything)

The three options actually on the table:

- **Raise `instance_group.count`** (tested 6, 8): free in VRAM, but trades
  `voice_pipeline`'s own graceful queueing for `kokoro_tts` timeout failures
  at a worse overall rate. Not shipped.
- **Leave it at 4** (current): the queueing is real and substantial, but it's
  currently the thing keeping failure rate low (~5%) by acting as
  backpressure in front of the pipeline's actual scarce resource. Kept
  as-is.
- **Fix the real constraint (`kokoro_tts` capacity)**: this is the change
  that would actually help, but it's explicitly out of scope for tonight --
  `kokoro_tts` instances are real ONNX Runtime CUDA sessions (~2-2.5GB each,
  per the README, and known to grow further under sustained load), and this
  GPU currently has only ~1.6GB free with all four models loaded. There is
  no VRAM headroom to safely add a 4th `kokoro_tts` instance right now
  without risking the OOM class of failure this project's guardrails
  specifically warn about. Raising it would need either freeing VRAM
  elsewhere first or revisiting `kokoro_tts`'s own memory footprint (e.g.
  the arena-growth behavior the README already flags, or a smaller/INT8
  weight variant, both listed under "Not yet done") -- a separate
  investigation, not a same-night follow-on to this one.

Net: **no config change is the correct, evidence-based outcome here**,
matching PR #10's precedent of a findings-only PR when the investigated fix
turns out not to be the right one.

## 4. A real, unrelated bug found and fixed along the way

While building the per-model metrics diff needed for section 1/2 above, an
independent bug in `scripts/load_test.py`'s own metrics parsing was found:

Triton exposes each **GPU-instance** model's Prometheus counters (e.g.
`nv_inference_exec_count`) as **two lines** -- one tagged with `gpu_uuid=...`
(the real, incrementing value) and one without (which stays `0` on this
single-GPU box). Which line appears first in `/metrics` differs by model --
confirmed directly via `curl localhost:18002/metrics`:

```
nv_inference_exec_count{gpu_uuid="GPU-...",model="nemotron_llm",version="1"} 681
nv_inference_exec_count{model="nemotron_llm",version="1"} 0
nv_inference_exec_count{gpu_uuid="GPU-...",model="kokoro_tts",version="1"} 441
nv_inference_exec_count{model="kokoro_tts",version="1"} 0
nv_inference_exec_count{model="whisper_asr",version="1"} 0
nv_inference_exec_count{gpu_uuid="GPU-...",model="whisper_asr",version="1"} 263
```

`nemotron_llm`/`kokoro_tts` have the real line first; `whisper_asr` has it
**second**. `load_test.py`'s `get_metric()` returned the first matching
line it found, which happened to be correct for `nemotron_llm`/`kokoro_tts`
but silently wrong for `whisper_asr` -- **every prior `load_test.py` run
against `whisper_asr` (including the ones already reported in this
project's own README and stress-loop logs) has reported
`avg_compute=0.0ms`, `avg_queue=0.0ms`, `execs=0`, `avg_batch_size=nan` for
it, regardless of what actually happened server-side.** Client-side latency
percentiles (measured independently by the load-test script itself, not
from Triton's metrics) were never affected by this -- only the
Triton-reported `avg_compute`/`avg_queue`/`execs` columns for `whisper_asr`
specifically.

Fixed by summing every matching line instead of returning the first
(`0 + real == real` regardless of ordering, and stays correct even if a
model genuinely spans multiple GPUs). Verified directly:

```
# before the fix: avg_compute=0.0ms avg_queue=0.0ms execs=0 avg_batch_size=nan
# after:
$ python3 scripts/load_test.py --model whisper_asr --concurrency 4 --total-requests 16
Triton avg_compute=212.6ms  avg_queue=5.2ms  execs=4  avg_batch_size=4.00
```

Shipped in this PR (`scripts/load_test.py`, live and worktree copies kept in
sync). This doesn't change any of section 1/2's `voice_pipeline`-level
conclusions above (those used a local wrapper with the same fix from the
start), but it's a real correctness fix worth having for every future
load-test run against `whisper_asr`.

## What's still open

- **`kokoro_tts`'s capacity is the actual bottleneck for pushing
  `voice_pipeline` past its current concurrency=8 failure rate**, not
  anything about `voice_pipeline` itself. A 4th `kokoro_tts` instance (or a
  smaller-footprint TTS variant) is the real next lever, but needs VRAM
  headroom this GPU doesn't currently have -- not attempted here.
- **`kokoro_tts` is already marginal at concurrency=8 even at today's
  baseline** (1-2/32 failures with `voice_pipeline` untouched) -- this
  predates this investigation and isn't caused by anything changed here, but
  is worth someone's attention independent of this doc.
- The `load_test.py` metrics fix (section 4) means `whisper_asr`'s
  Triton-side numbers in this project's *existing* docs (README,
  `nemotron-batch-size-scaling.md`) that came from `avg_compute`/`avg_queue`
  columns for `whisper_asr` specifically were reading `0`/`0` the whole
  time -- those docs' client-side latency numbers are unaffected and still
  correct, but anyone revisiting `whisper_asr`'s server-side queue/compute
  history should re-measure rather than trust the old `0` values.

## Files touched by this investigation

- `scripts/load_test.py`: `get_metric()` bug fix (sum matching Prometheus
  lines instead of returning the first match) -- see section 4. Applied to
  both the live server and this git worktree.
- This document.
- `triton_model_repo/voice_pipeline/config.pbtxt`: **no net change.**
  `instance_group.count` was tested live at 6 and 8 (safe unload/confirm/load
  pattern, `RELOAD_LOCK` held for each, VRAM and `tests/integration` checked
  before/after) and reverted to its original value of 4 once the sweep
  confirmed higher counts trade queueing for `kokoro_tts` failures rather
  than fixing anything. Live and worktree copies of this file are identical.
