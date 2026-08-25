# Overnight autonomous session, 2026-08-25 (00:40–08:30 PDT)

Summary of a ~7.5 hour unattended session: continuous stress-testing plus 9
investigations, run in parallel via background agents coordinating over a shared
GPU through a simple file-based reload lock. Full raw log:
`scratchpad/overnight/STATE.md` (session-local, not part of this repo).

## Headline finding

The continuous stress-test loop's own accumulated data surfaced a real, ongoing
**24.3% failure rate on `voice_pipeline`** (full end-to-end voice requests) at a
completely realistic load level (concurrency=4) — not a lab number, live production
behavior nobody was specifically looking for. Root-caused to `kokoro_tts`'s ONNX
Runtime CUDA session running on default memory-arena settings that grow unboundedly
under sustained load and never shrink back. Fixed (PR #18): tuned two CUDA EP
session options, which also cut TTS compute time 7–10x (694–969ms → 95ms) as a
side effect and freed enough VRAM to add a 4th `kokoro_tts` instance.

**That fix alone was not the whole story.** ~50 minutes after landing, a
`GPUFreeMemoryCritical` alert fired at 24MiB of GPU memory free — the arena had
crept back up under real sustained mixed traffic wider-ranging than the fix's own
verification bursts. Fixed live (manual unload/reload) and durably (the stress loop
now proactively resets `kokoro_tts` whenever free VRAM drops below 2GiB — fired 8
times cleanly over the following ~5 hours, unattended, no intervention needed).
Documented as an addendum to PR #18 rather than treated as solved-and-forgotten.

## Every PR opened (all drafts — nothing auto-merged)

| # | Title | Outcome |
|---|---|---|
| [#11](https://github.com/raoashish10/speech-cascade/pull/11) | Raise `nemotron_llm` `max_batch_size` 8→16 | Shipped — measured; 24/32 tested and rejected (real compute wall, not VRAM) |
| [#12](https://github.com/raoashish10/speech-cascade/pull/12) | Fix `SentenceAccumulator` splitting on abbreviations | Shipped — "Dr. Smith"/"e.g." no longer split mid-sentence |
| [#13](https://github.com/raoashish10/speech-cascade/pull/13) | Add `scripts/quantize_nvfp4.py` | Shipped — closes a real IaC gap, source-verified against upstream, not run (VRAM too tight at the time) |
| [#14](https://github.com/raoashish10/speech-cascade/pull/14) | `voice_pipeline` queueing at concurrency=8 | Findings only — the "count=1" premise was stale/wrong; also fixed a real `whisper_asr` metrics-parsing bug found along the way |
| [#15](https://github.com/raoashish10/speech-cascade/pull/15) | Cut `nemotron_llm` think-leakage | Shipped, partial — 46%→13%, honest about residuals |
| [#16](https://github.com/raoashish10/speech-cascade/pull/16) | Fix 96-token cap-hit rate | Shipped — 97%→7%, root-caused as a missing stop condition, not a framing problem |
| [#17](https://github.com/raoashish10/speech-cascade/pull/17) | VRAM headroom investigation | Findings only — ruled out a dead end before #18 found the real cause |
| [#18](https://github.com/raoashish10/speech-cascade/pull/18) | Fix `kokoro_tts`'s real 24.3% failure rate | Shipped — the headline fix, with the incident addendum above |

## Load-test numbers

1,091 `scripts/load_test.py` runs logged by the overnight stress loop, in addition
to each investigation's own dedicated measurements (cited in their own docs).

**Aggregate across the whole night** (mixes pre-fix, incident, and post-fix
periods — see the cleanly-isolated before/after below for the honest comparison):

| Model | Runs | Requests | Failures | Fail rate | p50 (median run) | p90 | p99 | Throughput (median) |
|---|---|---|---|---|---|---|---|---|
| `whisper_asr` | 304 | 4,864 | 0 | 0.00% | 57ms | 69ms | 69ms | 65.6 req/s |
| `kokoro_tts` | 304 | 4,864 | 16 | 0.33% | 112ms | 216ms | 226ms | 28.8 req/s |
| `nemotron_llm` | 227 | 7,264 | 0 | 0.00% | 116ms | 679ms | 883ms | 25.0 req/s |
| `voice_pipeline` | 256 | 4,096 | 338 | 8.25% | 900ms | 1.69s | 2.16s | 3.5 req/s |

**Cleanly isolated before/after** (the CSV has no timestamp column, so a single
aggregate blurs three distinct phases together — these two windows don't):

| Window | `voice_pipeline` runs | Requests | Failures | Fail rate |
|---|---|---|---|---|
| First 322 rows — confirmed pre-fix baseline | 87 | 1,392 | 338 | **24.28%** |
| Most recent 200 rows — current, hardened state | 50 | 800 | 0 | **0.00%** |

All 4 models logged zero failures across the most recent 200-row window (3,200
requests total).

### `nemotron_llm` `max_batch_size` sweep (PR #11)

| Concurrency | p50 | p90 | p99 | TTFT p50 |
|---|---|---|---|---|
| 1 | 0.673s | 0.988s | 0.988s | 14ms |
| 8 | 0.704–0.728s | 0.716–1.045s | 0.725–1.047s | 30ms |
| 16 (= new ceiling) | 0.726s | 0.740s | 0.750s | 42ms |
| 24 | 0.768s | 1.476s | 1.503s | 61ms |
| 32 (2× ceiling) | 1.504s | 1.525s | 1.538s | 786ms |

Flat through the full new ceiling, then a clean ~2.1× jump past it. 24 and 32 were
tested and rejected — a real compute wall, not VRAM (free VRAM held ~1.6GB across
every tested value).

### Think-leakage + token-cap fixes, 69-prompt eval harness (PRs #15, #16)

| Metric | Original (PR #7/#9) | + think-fix (PR #15) | + stop-sequences (PR #16, shipped) |
|---|---|---|---|
| Hit 96-token cap | 68/69 (99%) | 67/69 (97%) | **5/69 (7%)** |
| Leaked `<think>` reasoning | 32/69 (46%) | 9/69 (13%) | **3/69 (4%)** |
| Median response length | — | 76 words | **14 words** |
| Repetitive decode loop | — | 0/69 | 0/69 |

Root cause confirmed empirically: the model forms a correct, complete first
sentence at a median of 16–21 tokens (well inside the "1-2 sentences" target), then
drifts into unrelated reasoning-style rambling after a paragraph break — a
*stopping* problem, not a *framing* one. `repetition_penalty` was tested and ruled
out: dropping it to 1.0 barely moved the cap-hit rate (67→65/69) but reintroduced
the original decoding-degeneration bug on 10/69 prompts, so it was reverted.
Honest residuals: 5/69 hardest-category prompts (ambiguous/multi-part/unfulfillable)
still hit the cap; 2/69 are now too short; 2/69 leak a trailing `</think>` token.

### `kokoro_tts` arena fix, direct measurement (PR #18)

| Metric | Before (default ONNX settings) | After (tuned CUDA EP) |
|---|---|---|
| Per-instance VRAM (sustained load) | ~2.4 GB | **~889 MB** |
| Compute time @ concurrency=8 | 694–969ms | **95.4ms** |
| Failures @ concurrency=8 | 1–2/32 | **0/32** |
| `voice_pipeline` @ 12 concurrent chains | not tested pre-fix | **96/96, 0 failures** |

### Self-heal safety net, every firing

| Time (UTC) | Free VRAM at trigger | Free VRAM after reset |
|---|---|---|
| 13:05:30 | 1,232 MiB | 5,626 MiB |
| 13:28:44 | 1,234 MiB | 5,632 MiB |
| 14:00:15 | 1,240 MiB | 5,628 MiB |
| 14:23:30 | 1,236 MiB | 5,632 MiB |
| 14:53:24 | 1,240 MiB | 5,628 MiB |
| 15:41:28 | **212 MiB** | 5,630 MiB |
| 16:01:26 | 1,238 MiB | 5,624 MiB |
| 16:13:06 | 1,232 MiB | 5,634 MiB |

8 firings over ~5 hours, every one resolved cleanly within seconds (`kokoro_tts`
has no heavy engine-build step, unlike `nemotron_llm`). The 15:41 firing caught free
VRAM at 212MiB — lower than the ~1.2GB the check usually catches it at, meaning
growth can outpace a once-per-~90s-cycle check under fast enough load. Still
resolved with real margin to spare; noted as a real limit of this watchdog's polling
cadence in PR #18's addendum, not swept under the rug.

## Incidents handled during the session

- **GPUFreeMemoryCritical (24MiB free), 11:24 UTC** — see headline finding above.
  Root cause understood, fixed live, durably hardened.
- **A background agent hit its own API session limit mid-task** while holding the
  cross-agent reload coordination lock, blocking other work for over an hour before
  being caught. Verified the live server was never actually touched (the agent's
  edit only existed in its own worktree) before clearing the lock — no real damage,
  just a stale marker. Its partial finding was preserved rather than lost: a live
  scan found ~30% of some responses contained stray ```` ```json ```` fences that
  would get read aloud literally by TTS. A fix was drafted but **not shipped or
  tested** — sitting in an agent worktree, a real open item for a future session.

## What's still open

- The markdown/code-fence-in-speech fix above — drafted, unverified, unshipped.
- `kokoro_tts`'s own instance-count queueing under heavy overlap (PR #18's
  addendum flags this as the next real lever, now that VRAM headroom is much
  better than before this session).
- The self-heal watchdog's polling cadence (once per ~90s stress-loop cycle) is
  tuned for an overnight test driver, not hardened as a general-purpose production
  watchdog — a real production deployment of this pattern should poll faster or use
  a higher threshold.
