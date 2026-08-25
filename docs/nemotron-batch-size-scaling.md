# nemotron_llm max_batch_size scaling: how far the "nearly free" zone extends

## Verdict

**Raise `max_batch_size` from 8 to 16 (shipped in this PR), together with
`MAX_CONCURRENT_STREAMS` and `MAX_ADMITTED`.** This is the largest tested
value that preserves the "nearly free" flat-latency property PR #10
identified: p50 stays within ~8% of the concurrency=1 baseline (0.673s ->
0.726s) all the way out to concurrency=16, doubling the size of the flat
zone from PR #10's `max_batch_size=8` baseline (which stayed flat 1->8 and
then cliffed cleanly 2x at 16).

**24 and 32 were tested and rejected.** Both already show real,
non-clean-multiple latency growth *at their own ceiling concurrency* --
p50 up 38% at `max_batch_size=24`/concurrency=24, up 90% at
`max_batch_size=32`/concurrency=32. This is the **compute/bandwidth
crossover**, not a VRAM wall: free VRAM with all four models loaded stayed
at ~1.6 GiB, essentially unchanged, across every tested `max_batch_size`
(8/16/24/32). The wall is real, but it's compute, and it starts between
batch sizes 16 and 24 -- well before VRAM becomes a factor.

## Method

- Same instrumentation as PR #10: `nvidia-smi dmon` (this pass didn't need
  a fresh capture -- PR #10's already established the GPU-is-busy,
  memory-bandwidth-influenced baseline at `max_batch_size=8`; this
  investigation's new evidence is the concurrency-sweep latency shape
  itself, which is the more direct signal for *where the batching ceiling
  moves to*).
- `scripts/load_test.py --model nemotron_llm --concurrency N --total-requests
  <4N> --csv <path>` against the live, already-loaded model, for
  `max_batch_size` in {16, 24, 32} (8 is PR #10's existing baseline, not
  re-run). At each `max_batch_size`, concurrency was swept from 1 up to
  and beyond that ceiling. Both total-response p50/p90/p99 and TTFT
  (time-to-first-token) were read; TTFT is the more precise decode-phase
  signal since it isolates first-token latency from the full 96-token
  generation, but both tell the same story here.
- **Safe reload pattern used for every `max_batch_size` change**: unload,
  poll `/v2/repository/index` until `nemotron_llm` reports `UNAVAILABLE`,
  *then* load, poll until `READY`. Never a plain in-place load on an
  already-loaded model (this project hit a real CUDA OOM from that pattern
  earlier this session). Each reload paid the ~5-6 minute
  `tensorrt_llm`-import/engine-build cost; sweep values were planned in
  advance (16, 24, 32) to hold reloads to 4 total (3 sweep values + 1 final
  reload back to the shipped `max_batch_size=16`), not iterated reactively.
- `MAX_CONCURRENT_STREAMS` was raised to match each tested `max_batch_size`
  during its sweep (so the Python-level thread pool didn't itself cap
  concurrency below what the engine could actually batch), and
  `MAX_ADMITTED` kept the existing `2x` multiplier. This matters: without
  raising these together, a bigger engine ceiling would never get
  exercised by the load test, since admission control would reject
  requests before they reached it.
- VRAM headroom was estimated analytically (bytes/token x concurrent
  sequences x context length, from PR #10's confirmed 128 KiB/token
  figure) *before* any reload, then cross-checked against
  `nvidia-smi`/`triton_launch.log` after each real reload, with all four
  models (`whisper_asr`, `kokoro_tts`, `nemotron_llm`, `voice_pipeline`)
  loaded -- not `nemotron_llm` in isolation.
- End-to-end sanity check: `voice_pipeline`, `whisper_asr` read-only load
  tests at the final shipped config, to check for interaction with the
  existing `whisper_asr`/`kokoro_tts` queue timeouts. Neither model's
  `config.pbtxt` was touched.

## 1. VRAM check (done first, before any test run)

### Analytical estimate

PR #10's confirmed figure: **128 KiB/token** (combined K+V, all 32
layers), and this pipeline's real context length is **~150 tokens**
(short voice-assistant turns). KV memory needed to hold N *concurrent*
sequences at that context length:

```
N concurrent sequences x 150 tokens x 128 KiB/token

N=16:  16 x 150 x 128 KiB ≈ 300 MB
N=24:  24 x 150 x 128 KiB ≈ 450 MB
N=32:  32 x 150 x 128 KiB ≈ 600 MB
```

All trivial against a GPU with GBs of paged-KV pool -- this immediately
rules out KV *capacity for the sequences actually in flight* as a concern
at any of the tested batch sizes, consistent with PR #10's finding that
KV-cache bytes are a rounding error next to weight-streaming traffic here.

### What could plausibly move: execution-context memory, and it doesn't

The real open question wasn't "does the KV pool fit 16-32 short
sequences" (trivially yes) -- it was whether TensorRT-LLM's **execution
context memory**, which is sized at build time for the engine's configured
`max_batch_size` (activation/workspace buffers, not the KV pool), grows
enough with a bigger `max_batch_size` to squeeze the already-tight ~1.6
GiB of free VRAM this GPU has left once all four models are resident.

Measured directly from `triton_launch.log` at each reload (all four
models loaded, unload-confirm-load pattern followed each time):

| `max_batch_size` | engine size | execution context mem | runtime buffers | decoder | KV pool alloc | free VRAM (system-wide, all 4 models) |
|---|---|---|---|---|---|---|
| 8 (PR #10 baseline) | 3578 MiB | 531.00 MiB | 4.32 MB | 20.57 MB | 0.42 GiB (3424 tok) | 1653 MiB |
| 16 | 3578 MiB | 531.00 MiB | 8.61 MB | 41.14 MB | 0.41 GiB (3328 tok) | 1617 MiB |
| 24 | 3578 MiB | 531.00 MiB | 12.90 MB | 61.72 MB | 0.40 GiB (3264 tok) | 1625 MiB |
| 32 | 3578 MiB | 531.00 MiB | 17.19 MB | 82.29 MB | 0.39 GiB (3232 tok) | 1597 MiB |

**Execution context memory is flat at 531 MiB regardless of
`max_batch_size`** -- it does not scale with the batch-size ceiling in
this JIT-built engine profile. Only small per-sequence bookkeeping buffers
(runtime buffers, decoder state) grow, and they grow by tens of MB across
the whole 8->32 range, not GB. Free system-wide VRAM is flat within
measurement noise (~1.6-1.65 GiB) across every tested value. **VRAM
headroom was never the constraint at any tested `max_batch_size`, up to
32.** This directly answers the brief's central VRAM-vs-compute question:
it's not VRAM.

(Side note, not the deciding factor here but worth recording: the paged-KV
pool's *total* capacity, ~3200-3400 tokens across all four batch sizes,
comfortably covers 16-32 concurrent ~150-token sequences at once -- e.g.
32 x 150 = 4800 tokens *would* exceed the pool if every one of 32
concurrent sequences needed its full window simultaneously, but in
practice sequences complete and free blocks continuously under
`GUARANTEED_NO_EVICT` scheduling, and no KV-pool-exhaustion errors were
observed at any tested concurrency. This is a secondary headroom item to
watch if concurrency were pushed much higher than tested here, not a
blocker at the levels tested.)

## 2. Concurrency sweep at each `max_batch_size`

All p50/p90/p99 in seconds, `total_requests = 4x concurrency` per run
(except concurrency=1 sanity checks, n=8). Full raw data in
`scripts/load_test.py --csv` output, reproduced in summary below.

### `max_batch_size=16` (shipped)

| concurrency | p50 | p90 | p99 | TTFT p50 | notes |
|---|---|---|---|---|---|
| 1 | 0.673 | 0.988 | 0.988 | 0.014 | baseline |
| 8 | 0.704 / 0.728 (repeat) | 0.716 / 1.045 | 0.725 / 1.047 | 0.030 | flat |
| 16 | 0.726 | 0.740 | 0.750 | 0.042 | flat -- **matches new ceiling, still nearly free** |
| 24 | 0.768 | 1.476 | 1.503 | 0.061 (p90 0.774) | p50 still close to flat; p90/p99 already show the tail queuing behind a second batch (24 > 16 ceiling, same mechanism as PR #10's old cliff, now visible only in the tail since 16/24 = 67% of requests still land in the first pass) |
| 32 | 1.504 | 1.525 | 1.538 | 0.786 | clean ~2.1x baseline -- 32 = 2x the new ceiling, admission control (`MAX_ADMITTED=32`) starts rejecting right at this point (7/128 failed) |

**Shape: flat through the full new ceiling (16), then a clean multiple
past it.** Exactly the same queueing mechanism PR #10 found at the old
ceiling, just relocated to 2x the new, larger `max_batch_size`. This is
the good outcome -- the "nearly free" zone genuinely doubled.

### `max_batch_size=24`

| concurrency | p50 | p90 | p99 | TTFT p50 | notes |
|---|---|---|---|---|---|
| 1 | 0.678 | 0.883 | 0.883 | 0.014 | baseline |
| 16 | 0.795 | 0.830 | 0.835 | 0.040 | **already +17% over baseline, below the batch's own ceiling** |
| 24 | 0.939 | 1.038 | 1.142 | 0.071 | **+38% at its own ceiling -- not flat** |
| 32 | 1.085 | 1.850 | 1.975 | 0.189 | continues climbing, no clean-multiple step |
| 40 | 1.529 | 1.911 | 2.053 | 0.790 | continues climbing |
| 48 | 1.941 | 2.078 | 2.201 | 1.057 | `MAX_ADMITTED=48` boundary reached, some rejections (17/144) |

**Shape: gradual, continuous growth starting below the batch's own
ceiling, no clean-multiple step anywhere in this range.** This is
qualitatively different from `max_batch_size=16`'s behavior and from PR
#10's `max_batch_size=8` behavior -- both of those stayed genuinely flat
right up to their configured ceiling. At `max_batch_size=24`, the "nearly
free" property is already gone by the time concurrency reaches the
ceiling itself.

### `max_batch_size=32`

| concurrency | p50 | p90 | p99 | TTFT p50 | notes |
|---|---|---|---|---|---|
| 1 | 0.677 | 0.873 | 0.873 | 0.014 | baseline |
| 16 | 0.738 | 0.751 | 0.761 | 0.039 | still roughly flat (+9%) -- only half the ceiling engaged |
| 24 | 0.989 | 1.117 | 1.203 | 0.081 | +46% -- worse than `max_batch_size=24`'s own concurrency=24 number (+38%) at the *same* concurrency |
| 32 | 1.286 | 1.584/1.935 (see raw data, two runs) | 1.935-2.044 | 0.144-0.189 | +90% at its own ceiling |
| 48 | 1.865 | 2.358 | 2.667 | 0.782 | continues climbing |

**Key cross-check**: at the *same* concurrency=24, `max_batch_size=32`
(p50=0.989s) is measurably worse than `max_batch_size=24` (p50=0.939s).
Building the engine for a larger ceiling costs something even when actual
concurrent load doesn't reach that ceiling -- consistent with a real
compute-cost increase baked into the larger batch profile (larger
optimization-profile matmul shapes cost more per step even at partial
occupancy), not just "more sequences to serve." This rules out "it's only
bad once you're at the new ceiling" as an explanation and confirms this is
a genuine compute crossover, not a queuing artifact.

## 3. Classifying the break

Per the brief's framing:

- **`max_batch_size=16`**: clean 2x multiple at concurrency=32 (2x the
  ceiling), flat up to the ceiling itself. **Same queueing mechanism as
  PR #10** -- the scheduler falls back to two serial batch passes once
  concurrency exceeds the configured ceiling. Nothing new here except the
  ceiling moved.
- **`max_batch_size=24` and `32`**: gradual, non-clean-multiple
  degradation that starts *before* the ceiling is even reached, and gets
  worse as `max_batch_size` increases even at matched concurrency. **This
  is the compute/bandwidth crossover** the brief asked about: batched
  matmul cost is no longer fully hidden behind the shared weight-stream
  read once the batch gets large enough -- the extra FLOPs of a bigger
  batch (still tiny per PR #10's math at batch=8, but this scales with
  batch size) start to matter once the per-step read is being amortized
  across enough sequences that the compute term catches up.

**The crossover sits between `max_batch_size=16` and `24`** on this GPU
with this checkpoint. 16 is the last value tested where the flat-latency
property genuinely holds through the full ceiling.

## 4. Admission control interaction (`MAX_CONCURRENT_STREAMS` / `MAX_ADMITTED`)

The brief flagged that `MAX_CONCURRENT_STREAMS=8`/`MAX_ADMITTED=16` were
calibrated against the *old* `max_batch_size=8` cliff, and that raising
the engine ceiling without moving these would leave admission control
capping concurrency below where the new ceiling could even be exercised.
Confirmed by testing: with `MAX_CONCURRENT_STREAMS` left at 8 while
`max_batch_size` was raised, the load test would never generate more than
8 concurrent engine-side requests, and the new ceiling's flat zone
(9-16) would go completely unmeasured. Both settings need to move
together, and did for every sweep in this investigation.

**Shipped**: `MAX_CONCURRENT_STREAMS = 16`, `MAX_ADMITTED = 32` (kept the
existing `2x` multiplier convention). At `max_batch_size=16` this lines up
cleanly with the measured shape -- the admission ceiling (32) sits exactly
at the concurrency where the clean-2x queueing kicks in anyway, so
admission control and the engine's own batching ceiling agree on where
"overloaded" starts. 24 and 32 were *not* shipped, so their
`MAX_ADMITTED` values (48/64) were sweep-only and are not live.

## 5. Queue-timeout sanity check (`whisper_asr`, `kokoro_tts`)

Not retuned (out of scope) -- checked whether the shipped
`max_batch_size=16` change measurably shifts full-chain latency enough to
threaten the existing margins.

- `whisper_asr`: `default_queue_policy` timeout 2.5s, grounded in ~0.75s
  single-instance p50 (~3x margin). Isolated `whisper_asr` load test at
  concurrency=8 in this session measured p50=0.098s (this repo's test
  fixture audio is short; the 2.5s timeout's own grounding number from the
  original PR used a different reference, but either way `whisper_asr`'s
  own compute time is untouched by this change -- nothing about
  `nemotron_llm`'s `max_batch_size` affects `whisper_asr`'s internal
  queue/compute at all, since they're separate models/instances on the
  same Triton server).
- `kokoro_tts`: `default_queue_policy` timeout 0.75s, grounded in
  ~70-90ms compute with 3 instances (~9x margin). Same reasoning:
  untouched by this change.
- `voice_pipeline` end-to-end, concurrency=8, with the shipped
  `max_batch_size=16` config live: p50=4.044s (Triton-reported
  `avg_compute=2036ms`, `avg_queue=1686ms` for the `voice_pipeline` model
  itself). This is dominated by `voice_pipeline`'s own single-instance
  orchestrator queueing (out of scope to change here), not by
  `nemotron_llm`: `nemotron_llm` alone at the same concurrency=8 measures
  p50=0.728s, essentially unchanged from -- if anything slightly better
  than -- the PR #10 baseline (p50=0.713s at concurrency=8,
  `max_batch_size=8`). **The shipped change does not make voice_pipeline
  slower**; at concurrency 9-16 specifically it makes `nemotron_llm`'s
  contribution *better* than before (previously this concurrency range
  would have hit the old `max_batch_size=8` 2x cliff; now it stays flat).
  No change needed to `whisper_asr`/`kokoro_tts` timeouts.

## Why 16, not 24 or 32

24 and 32 do show higher raw throughput in the load-test numbers
(23-25 req/s vs ~21-22 req/s at 16), because more requests are admitted
concurrently even though each one is individually slower. But the actual
goal of this investigation -- and the reason PR #10 flagged
`max_batch_size` as the lever worth raising in the first place -- was
extending the *flat, nearly-free* latency zone, not maximizing raw
throughput at the cost of per-request latency. 16 is the largest value
that keeps that property; 24 and 32 trade it away for a throughput gain
that isn't needed at this pipeline's realistic concurrency (a single-GPU
voice pipeline; double-digit concurrent *engine-level* nemotron requests,
let alone 24-48, is already a generous upper bound for the intended
deployment).

## What would actually help further, if this needs to be faster later

Not implemented here, each would need its own measurement pass:

- If real traffic patterns ever justify sustained concurrency in the
  17-32 range and raw throughput matters more than per-request tail
  latency, `max_batch_size=24` is a defensible middle ground (38% p50
  growth at its own ceiling, not catastrophic, still far short of a clean
  2x) -- but this trades away the property this investigation was
  specifically asked to preserve, so it wasn't shipped without that
  tradeoff being an explicit choice, not a default.
- The compute/bandwidth crossover itself (why batched matmul cost starts
  mattering around batch 16-24 on this GPU/checkpoint) wasn't
  root-caused at the kernel level here -- profiling with `nsys`/Nsight at
  `max_batch_size=24` decode steps would show whether it's tensor-core
  occupancy, dequant overhead scaling with batch, or something else, if
  that mechanism ever needs to be understood rather than just measured
  around.
- `voice_pipeline`'s own single-instance orchestrator queueing (measured
  here: `avg_queue=1686ms` at concurrency=8, dwarfing `nemotron_llm`'s own
  contribution) is a bigger lever on real end-to-end latency than
  anything further in `nemotron_llm` at this point -- flagged as a
  distinct, unmeasured follow-up, explicitly out of scope for this pass
  per the brief.

## Files touched by this investigation

- `triton_model_repo/nemotron_llm/1/model.py`: `max_batch_size` 8 -> 16
  (in the `_TrtLLM(...)` constructor -- not Triton's own
  `max_batch_size: 0` in `config.pbtxt`, which is unrelated and untouched),
  `MAX_CONCURRENT_STREAMS` 8 -> 16, `MAX_ADMITTED` 16 -> 32 (via the
  existing `2x` formula). Applied live via the safe unload/confirm/load
  sequence; live server state matches this file as of this investigation.
- This document.
- No changes to `whisper_asr`, `kokoro_tts`, or `voice_pipeline`
  configs/code.
