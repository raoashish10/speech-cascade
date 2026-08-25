# nemotron_llm response-quality tuning pass (empirical, not RL)

## What this is and isn't

This is an **empirical quality-tuning pass**, not real gradient-based RL —
there is no training infrastructure set up for this NVFP4-quantized
TensorRT-LLM checkpoint on this box, and standing one up overnight
unattended on a live-serving instance would be irresponsible. "RL-style"
here means the same thing PR #7 (chat template) and PR #9 (max_tokens
96) already did: generate many diverse prompts, score responses against
heuristics defined in advance, measure a candidate config change against
the same prompt set, and ship only what's actually measured to help. No
weights were touched.

## Verdict

**Shipped one narrow, measured change; the deeper problem is not fixed and
is flagged below for whoever picks this up next.**

The baseline (PR #7/#9 config, live at the start of this pass) has a real,
severe, previously-uncaught defect: **32/69 (46%) of realistic
voice-assistant prompts produced a response that opens a visible
`<think>...</think>` reasoning block and leaks raw internal monologue as
the "spoken" answer**, despite the existing "detailed thinking off" system
prompt. This is a much bigger problem than the 8-prompt regression suite
(`test_regression.py`) can see, since that suite only asserts
non-empty output.

The fix — forcing an already-closed empty `<think>\n\n</think>\n\n` block
into the prompt immediately after the chat template's generation header,
the standard technique for this family of toggleable-reasoning checkpoints
— **cut think-leakage from 32/69 to 9/69** (a real, reproducible,
73%-relative reduction) with no measurable regression on cap-hit rate or
response length. It's shipped, live and in git.

**What it did NOT fix**, and what nothing this session tried has fixed:
**68/69 (baseline) and 67/69 (candidate) of all 69 prompts still hit the
96-token `max_tokens` cap without ever reaching a natural stop.** Median
response length is ~76 words in both configs — roughly 3-5x the "1-2 short
spoken sentences" target — regardless of whether the reasoning-leakage fix
is applied. This checkpoint, at this quantization and decoding
configuration, essentially never produces a fully-resolved, concise answer
to a real-world voice-assistant-style question. PR #9 already tried
strengthening the user-turn instruction once and found it made things
*more* confused, not less — so this looks less like a prompt-wording
problem and more like a real ceiling on this checkpoint's
instruction-following at NVFP4/greedy-decode, or a problem this eval
harness's own framing (chat-template + system instruction) can't reach.
Recorded as an open problem below rather than something this pass claims
to have solved.

The fix also introduced a **different failure mode** that wasn't
pre-registered as a heuristic and was only noticed while reading candidate
outputs: markdown code fences, `\boxed{}` LaTeX, and numbered "Step 1:"
formatting appeared in candidate responses far more often than baseline
(code fences: 11/69 vs 1/69). None of that is speakable text. This is a
real, if smaller, cost of the change — disclosed honestly rather than
buried, see "New artifact class" below.

## Method

1. **Prompt set**: 69 prompts across 11 categories (factual, ambiguous,
   multi-part, unfulfillable commands, yes/no, no-good-answer,
   context-free follow-ups with pronouns, adversarial/edge — single word,
   gibberish, ASR-typo-style garbling, instruction-probing, personal/
   opinion, simple math) — `scratchpad/eval/prompts.json` (not committed;
   regenerable, listed in full in this doc's method if needed). Built
   *before* any output was inspected, informed by what real ASR transcripts
   from `whisper_asr` will actually look like (no punctuation guarantees,
   filler words, misrecognitions) and by this pipeline's specific
   properties (no conversation history is kept, so "What about the second
   one?"-style follow-ups are a real, expected input class, not a
   hypothetical).
2. **Scoring heuristics, fixed before looking at any output**:
   - `hit_max_tokens_cap`: generated token count (via the checkpoint's own
     tokenizer, loaded locally/CPU-only, no GPU contention) >= 90 out of
     the configured `max_tokens=96` — a response this close to the cap
     essentially never got a natural stop.
   - `think_leakage`: regex match for `<think`, `</think`, or reasoning-
     mode phrasing anywhere in the response text.
   - `word_count` + `too_short_flag` (<2 words) / `too_long_flag` (>70
     words) — thresholds chosen relative to the target "1-2 short spoken
     sentences" (roughly 15-30 words).
   - `repetitive_loop_flag`: trigram-repetition ratio (unique
     trigrams/total trigrams) < 0.6 — a simple, cheap n-gram loop detector.
   - `refusal_pattern_hit` / `fake_confirmation_pattern_hit`: regex
     families for honest "I can't ..." language vs. hallucinated
     "Done!"/"Timer set"/"Added to your list"-style fake confirmations,
     scored only on the `command_unfulfillable` category (8 prompts:
     "set a timer", "play music", "turn off the lights", etc. — things
     this pipeline genuinely cannot do).
   - **On-topic / correctness** is explicitly *not* a hard heuristic — it's
     my own qualitative read, called out as such wherever it's used below,
     not dressed up as a metric.
3. **Baseline run**: all 69 prompts against the live server as-is
   (chat-template + max_tokens=96 + repetition_penalty=1.15 + `<|eot_id|>`
   stop, i.e. PR #7/#9's shipped config), via `tritonclient.grpc.aio`
   streaming (`stream_infer`), the same call shape
   `streaming_gateway/triton_client.py`'s `generate_stream()` uses. Pure
   inference, no reload, safe at any time.
4. **Candidate change**: exactly one line added to
   `triton_model_repo/nemotron_llm/1/model.py`'s prompt construction (see
   diff below). Nothing else touched — same `max_tokens`, same
   `repetition_penalty`, same stop sequence, same chat template messages.
5. **Live reload**: coordinated via
   `scratchpad/overnight/RELOAD_LOCK` per this session's multi-agent
   protocol (checked it was clear, claimed it, released it the moment the
   reload finished). Safe-reload pattern: unload -> poll
   `/v2/repository/index` until `UNAVAILABLE` -> load -> poll until
   `READY`. No in-place load on an already-loaded model.
6. **Candidate run**: the identical 69-prompt set, same scoring code,
   against the reloaded server.
7. **Regression check**: `pytest tests/integration -v` — 15/15 passed both
   before and after the reload.

## The change

```python
prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt += "<think>\n\n</think>\n\n"
```

"detailed thinking off" in the system turn is this checkpoint's documented
toggle for suppressing its reasoning phase, but the baseline data shows
it's a soft preference the model overrides in nearly half of realistic
prompts, not a hard constraint. Forcing an *already-closed* empty think
block into the prompt removes the model's ability to open one at all —
it has no choice but to continue directly into what would be the
post-`</think>` answer content. Same convention used for
DeepSeek-R1-distill and other Nemotron-Nano-family checkpoints with a
toggleable chain-of-thought.

## Results

### Pre-registered hard metrics, full 69-prompt set

| Metric | Baseline | Candidate |
|---|---|---|
| `hit_max_tokens_cap` | 68/69 (99%) | 67/69 (97%) |
| `think_leakage` | **32/69 (46%)** | **9/69 (13%)** |
| `repetitive_loop_flag` | 1/69 | 0/69 |
| `too_long_flag` (>70 words) | 54/69 (78%) | 52/69 (75%) |
| `too_short_flag` (<2 words) | 0/69 | 0/69 |
| median response length | 76 words | 76 words |
| mean response length | 74.3 words | 72.8 words |
| errors | 0 | 0 |

think-leakage was fixed on 27 prompts and newly introduced on 4 (net -23,
32 -> 9). Every other metric moved by 0-2 prompts out of 69 — noise, not
signal.

### Per-category think-leakage (baseline), for context on where it hit hardest

| Category | n | think-leak (baseline) |
|---|---|---|
| adversarial_edge | 10 | 6 |
| ambiguous | 8 | 5 |
| instruction_probe | 5 | 4 |
| no_context_followup | 6 | 4 |
| multi_part | 5 | 4 |
| factual | 10 | 3 |
| no_good_answer | 5 | 2 |
| command_unfulfillable | 8 | 2 |
| yes_no | 5 | 1 |
| opinion_personal | 4 | 1 |
| math_simple | 3 | 0 |

No category was immune; the no-context-followup and adversarial/edge-case
prompts (exactly the kind of thing a real ASR transcript produces) were
among the worst-hit.

### New artifact class (not pre-registered — noticed while reading candidate output, disclosed honestly)

| Signal | Baseline | Candidate |
|---|---|---|
| contains `` ``` `` code fence | 1/69 | 11/69 |
| contains `\boxed{}` | not measured (rare, seen ad hoc) | 10/69 |
| contains a "Step 1:" pattern | not measured ad hoc | 5/69 |

Suppressing the reasoning phase seems to have pushed some of what used to
be `<think>` scratch content into markdown/code-formatted "structured
answer" text instead — still not speakable, just a different kind of
unspeakable. Example (`cmd-06`, "Add eggs to my shopping list."):

```
BASELINE:  "I'm sorry for any confusion, but I can certainly help with the
            question. However, since you've asked for a direct answer..."
            (rambling, incoherent, but honest that nothing happened)

CANDIDATE: ```json
           {
               "eggs": {
                   "shopping_list": "done"
               }
           }
           ```
           **Step-by-Step** ...
            (a fabricated "done" confirmation dressed as JSON -- arguably
            a worse failure: a confident, structured, false claim that the
            action succeeded)
```

### Command-unfulfillable: honest refusal vs. hallucinated confirmation (n=8, small sample, exploratory)

| | Baseline | Candidate |
|---|---|---|
| `refusal_pattern_hit` | 3/8 | 1/8 |
| `fake_confirmation_pattern_hit` | 1/8 | 1/8 |

Neither config reliably produces a clean "I can't do that" for things this
pipeline genuinely cannot do (set timers, play music, control lights, send
texts) — both mostly produce confused, unresolved meta-commentary that
gets truncated mid-thought. The candidate is not an improvement here; if
anything it's slightly worse on the refusal count, though n=8 is too small
to read much into a swing of 2.

### Qualitative: does the response even contain the right fact? (my own judgment, factual category, n=10, spot-checked)

Not a pre-registered heuristic — read manually, disclosed as a judgment
call. Baseline responses were so often off-topic (see `fact-01`,
`fact-06`, `fact-09`, `fact-10` — none of these ever state the actual
answer) that only ~1/10 baseline responses clearly and correctly stated
the requested fact anywhere in the first few sentences (`fact-05`,
boiling point). Candidate responses reach the correct fact near the start
noticeably more often (`fact-05`, `fact-07`, `fact-08`, partially
`fact-02` — roughly 4/10), consistent with the model no longer burning its
early tokens on an internal monologue that sometimes never resolves even
internally. But two candidate responses (`fact-01`: "the capital of
France is **[not specified]**... cannot be determined"; `fact-03`: "no
such historical figure [Romeo and Juliet's author] ever existed") open
with a **confident, wrong, hallucinated claim** rather than either the
right answer or an honest "I don't know" — arguably worse than baseline's
incoherent-but-not-actively-false rambling on those same two prompts. This
read is subjective and n=10 is small; it should not be taken as more
certain than it is, but it's consistent with the token-budget story above:
the model still burns most of its 96-token budget on hedging/qualifying
regardless of whether that hedging happens inside a visible `<think>` tag
or not, and content correctness is not something this pass measured
rigorously enough to claim a verdict on either way.

## What's shipped

- `triton_model_repo/nemotron_llm/1/model.py`: the one-line prompt
  addition above, with an inline comment pointing back to this doc.
- Live server: reloaded with this change, confirmed `READY`,
  `pytest tests/integration -v` passes 15/15 post-reload.
- `RELOAD_LOCK` claimed for the duration of the reload and released
  immediately after.

## What's explicitly NOT fixed / follow-up for whoever picks this up next

1. **The core problem — responses don't stay to "1-2 short spoken
   sentences" — is unresolved.** 67-68/69 prompts hit the 96-token cap
   regardless of this change. PR #9 already tried strengthening the
   user-turn instruction once (reverted, made things worse). This pass
   didn't re-attempt a prompt-wording fix for it, to keep this pass to one
   isolated variable — but the next thing worth trying, as its own
   separately-measured experiment, might be: (a) explicit "no markdown, no
   code, no lists — plain conversational sentences only" language in the
   user turn (targets the new artifact class directly), and/or (b) a hard
   stop sequence on `` ``` `` and two-newline-plus-bullet patterns, and/or
   (c) accepting that this checkpoint at NVFP4/greedy won't reliably
   self-truncate, and building a downstream cleanup step (truncate to the
   last complete sentence before the token cap, strip markdown before
   handing text to `kokoro_tts`) as a pragmatic mitigation instead of
   continuing to chase a model-side fix.
2. **The new markdown/code-artifact class** (11/69 code fences, 10/69
   `\boxed{}`) is a real, if smaller, regression introduced by this
   change, not yet mitigated.
3. Whether `kokoro_tts` actually mangles markdown/code-fence text audibly,
   or silently drops/mispronounces it, was **not tested** — this pass only
   scored LLM text output, not the resulting audio. Worth checking before
   assuming severity either way.
4. Honest-refusal-vs-hallucination for unfulfillable commands remains
   weak in both configs and wasn't specifically targeted by this pass.

## Files touched by this pass

- `triton_model_repo/nemotron_llm/1/model.py` — the one-line change +
  comment (git-tracked, this doc).
- `docs/nemotron-response-quality.md` — this document.
- Eval harness and raw results (`prompts.json`, `run_eval.py`,
  `results_baseline.json`, `results_candidate.json`) live in this
  session's scratchpad, not committed to the repo — regenerable from the
  method above if a future session wants to re-run or extend it.
