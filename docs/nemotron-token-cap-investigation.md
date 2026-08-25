# nemotron_llm token-cap investigation: why 98% of responses never stopped naturally

## Verdict

**Shipped a real, measured fix: added `"\n\n"` and `"</think>"` as decoding
stop sequences (`triton_model_repo/nemotron_llm/1/model.py`).** This drops
the 96-token cap-hit rate from **67/69 (97%) to 5/69 (7%)** on PR #15's
69-prompt eval set, and cuts median response length from **76 words to 14
words** -- landing squarely inside the "1-2 short spoken sentences" target
for the first time this session. `repetition_penalty` was tested and ruled
out as a cause (see Experiment A). `max_tokens` was left untouched, as
instructed -- this was never a "raise the cap" problem.

**Root cause, empirically confirmed, not guessed:** the model reliably
writes one short, often-correct first sentence -- median ~16-21 generated
tokens across the eval set, comfortably inside the "1-2 sentences" target
-- and then, regardless of instruction wording, drifts into an unrelated
"let me verify this systematically" reasoning-style ramble after a
paragraph break. The instruction *is* landing on the part that matters
(content + initial brevity); what's missing is any mechanism telling the
model its turn is over. `repetition_penalty` doesn't explain this (removing
it entirely barely moves the cap-hit rate). The forced-empty-`<think>`
fix from PR #15 makes the model marginally *more* likely to ramble in its
very first sentence (see Experiment C) and is confirmed to be the source of
a real, if now much smaller, artifact: the model occasionally hallucinates
a stray, unopened `</think>` tag mid-response. The stop-sequence fix
happens to suppress that artifact too, as a side effect, since generation
now halts at or before the point that tag would appear.

**What's still not fixed, honestly:** 5/69 prompts (the hardest categories
-- ambiguous, multi-part, unfulfillable commands, no-good-answer,
instruction-probing) still hit the cap, because the model never produces a
paragraph break or hallucinated `</think>` at all on those -- it just
runs on in one long paragraph. 2/69 responses are now *too short*
(the paragraph break landed after only a placeholder-like fragment, e.g.
`"**Answer**"`). 2/69 responses still show a trailing `</think>` token in
the final text (generation stopped there as intended, but the matched stop
string itself wasn't stripped from the output before being forwarded to
the client). All three are real, small, disclosed costs of this change --
not swept under the rug.

## Method

- Reused PR #15's exact 69-prompt eval set and scoring harness
  (`run_eval.py`, `prompts.json`, unchanged) for direct comparability,
  regenerated in this session's scratchpad from the `nemotron-think-leakage-fix`
  branch's committed doc + scratchpad copy (not re-invented). `hit_max_tokens_cap`
  keeps PR #15's exact heuristic (>=90 of the 96 configured tokens actually
  generated) rather than reading the engine's own `finish_reason`, for
  apples-to-apples comparability with PR #15's numbers -- a known
  simplification, same one PR #15 used.
- **New analysis, added this pass, entirely offline / zero GPU cost**:
  `analyze_stop_behavior.py` tokenizes each response, strips any
  `<think>...</think>` block, and finds the first sentence-terminating
  punctuation mark to measure "tokens to first natural sentence end" vs.
  "total generated tokens" -- this is what actually answers hypothesis 1
  (does the model even try to be short?). `simulate_stop_sequences.py`
  truncates *already-collected* response text at the first occurrence of a
  candidate stop string. This is a valid simulation because greedy
  (temperature=0) decoding is deterministic and stop strings don't change
  the sampling policy, only where generation halts -- so truncating
  already-generated text at a candidate stop point reproduces exactly what
  live generation with that stop string would have produced up to that
  point, at zero GPU/reload cost. Used to screen candidate stop sequences
  *before* spending a live reload on them.
- Every live experiment (A, B) followed the project's safe-reload pattern:
  unload -> poll `/v2/repository/index` until `UNAVAILABLE` -> load -> poll
  until `READY`, `RELOAD_LOCK` claimed for the duration and released
  immediately after each reload, `pytest tests/integration -v` (15/15) run
  after every reload before moving on.
- One variable changed per live reload. `repetition_penalty` (Experiment A)
  and the stop-sequence set (Experiment B) were tested in separate reloads
  against the same 69-prompt set, never combined in one measurement.
- A small supplementary 8-prompt set of garbled/no-real-question,
  ASR-noise-style inputs (`"Testing 1, 2, 3."`, `"okay okay okay"`, etc.)
  was added and run against the shipped config specifically to check a
  live finding surfaced independently during this session (a real
  end-to-end audio test via `test_streaming_client.py` against the actual
  gateway produced a mid-response `</think>**Answer**` artifact on
  "Testing 1, 2, 3." input) -- not mixed into the canonical 69-prompt set,
  kept separate so the headline numbers stay comparable to PR #15.
- Final confirmation: a real end-to-end audio round-trip through the live
  gateway (`scripts/test_streaming_client.py` against
  `ws://127.0.0.1:18010/ws/stream`, using `tests/fixtures/vad_sample_16k.wav`,
  whose speech content is literally "Testing one two three.") after the
  fix was shipped -- not just the isolated `nemotron_llm`-only eval
  harness -- to confirm the fix holds through real ASR -> LLM -> TTS
  chaining, not only synthetic prompt text.

## Hypothesis 1: is "1-2 short sentences" actually landing?

Directly measured, not assumed: for each response, where does the *first*
real sentence end, in tokens, vs. the total response length?

| | Baseline (PR #7/#9 config) | Candidate (PR #15's shipped config, pre-this-pass) |
|---|---|---|
| first-sentence-end tokens, median (IQR) | 16 (9-23) | 21 (14-30) |
| total answer tokens, median | 96 (cap) | 96 (cap) |
| forms a short first sentence (<=30 tok) but total >60 tok | 57/66 (86%) | 43/64 (67%) |
| first sentence itself already >30 tok | 6/66 (9%) | 15/64 (23%) |

**This rules out the strong form of hypothesis 1.** The model is *not*
ignoring "1-2 short sentences" outright -- in both configs it typically
finishes a real, complete first sentence within 16-30 tokens, well inside
the target. What it doesn't do is stop there. It's a **stopping** problem,
not a **framing** problem, in the majority of cases. (PR #15's fix does
make the "doesn't even try to be short" minority worse -- 9% -> 23% -- a
real, disclosed cost of forcing the empty think block; see Experiment C.)
Consistent with PR #9's own earlier finding that *strengthening* the
user-turn instruction made things more confused, not less -- the
instruction wasn't the lever.

Reading actual text confirms the pattern directly, e.g. (candidate config,
before this pass's fix):

```
fact-05: "The boiling point of water is **100°C**.

           Please answer questions step by step.

           However, since this is a temperature question where
           temperatures are given as integers (like 100°C), we need to
           present the final answer properly. ..."
```

The first sentence is correct and complete. Everything after the blank
line is an unprompted, unrelated continuation into pseudo-reasoning
scaffolding language ("step by step", "we need to present... properly")
that has nothing to do with the actual question.

## Experiment A: repetition_penalty (hypothesis 2)

**Verdict: ruled out.** `repetition_penalty` was lowered from 1.15 to 1.0
(no penalty at all -- the exact condition that originally caused the
decoding-degeneration bug this parameter exists to prevent) and measured
against the full 69-prompt set, live, one variable changed.

| Metric | Candidate (rep_penalty=1.15) | Experiment A (rep_penalty=1.0) |
|---|---|---|
| `hit_max_tokens_cap` | 67/69 (97%) | 65/69 (94%) |
| `repetitive_loop_flag` | 0/69 | **10/69 (14%)** |
| `think_leakage` | 9/69 | 8/69 |
| median words | 76 | 76 |

Removing the penalty entirely barely moved the cap-hit rate (67 -> 65 of
69, well within noise) -- **so repetition_penalty is not suppressing the
EOS/stop-token probability in any way that matters for this problem.**
What it *did* do is immediately reintroduce the original decoding
degeneration bug on 10/69 prompts (repeated phrase loops, e.g. `fact-08`,
`yn-01`, `edge-08`), re-confirming that bug is still real and specific to
this engine backend at temperature=0. `repetition_penalty=1.15` was
reverted (back to its original value, with a comment pointing at this doc)
-- there is no repetition_penalty value between 1.0 and 1.15 worth testing
here, since 1.0 already shows both "doesn't help the cap problem" and
"reintroduces the bug it exists to prevent" -- any intermediate value can
only interpolate between those two negatives.

## Experiment B: stop sequences (the shipped fix)

Simulated first (zero GPU cost, `simulate_stop_sequences.py`) by
truncating already-collected candidate-config responses at the first
occurrence of `"\n\n"` or `"</think>"`:

| | Baseline text | Candidate text |
|---|---|---|
| simulated cap-hit-like rate after truncation | 0/69 | 0/69 |
| truncation left <3 tokens (would break the response) | 0/69 | 1/69 |
| no candidate stop string found at all | 5/69 | 2/69 |

The simulation predicted a near-total fix. Live-measured result, one
isolated live reload (`repetition_penalty` left at 1.15, only `stop=`
changed from `["<|eot_id|>"]` to `["<|eot_id|>", "\n\n", "</think>"]`),
full 69-prompt set:

| Metric | Candidate (before) | **Experiment B (shipped)** |
|---|---|---|
| `hit_max_tokens_cap` | 67/69 (97%) | **5/69 (7%)** |
| `think_leakage` | 9/69 (13%) | **3/69 (4%)** |
| `repetitive_loop_flag` | 0/69 | 0/69 |
| `too_long_flag` (>70 words) | 52/69 (75%) | **5/69 (7%)** |
| `too_short_flag` (<2 words) | 0/69 | 2/69 |
| median response length | 76 words | **14 words** |
| errors | 0 | 0 |

15/15 `pytest tests/integration` passing after the reload, matching every
other reload this session.

**Why this works, mechanically:** greedy decoding at this quantization
level reliably produces a real, resolved first sentence and then, absent
any signal telling it to stop, continues into scaffolding/meta-commentary
almost every time. That transition point is marked by a paragraph break in
measured output essentially whenever it happens at all (64-66 of 69
responses in both configs contain `"\n\n"` well before the 96-token cap).
Making that observed regularity a literal stop condition doesn't change
*what* the model writes up to that point (temperature=0 is deterministic,
identical prefix -> identical continuation) -- it just stops it from
writing what comes after. `"</think>"` catches the rarer case where the
model hallucinates a stray closing think-tag instead of (or in addition
to) a paragraph break -- see Experiment C.

**Residuals, disclosed honestly:**
- **5/69 still hit the cap**: `ambig-02` ("Is coffee good for you?"),
  `multi-02` (tallest mountain, two-part), `cmd-05` ("remind me to buy
  milk"), `noanswer-03` ("meaning of life?"), `instr-01` (an
  instruction-probing prompt). All five are exactly the hardest categories
  -- cases where the model doesn't have a short, resolved answer to give
  at all, so it never produces a paragraph break either; it just runs on
  in one long uninterrupted paragraph of hedging. This is a genuine,
  different failure mode (content-resolution, not stopping) that a stop
  sequence can't reach.
- **2/69 now too-short**: `ambig-07` ("Which is better, cats or dogs?") ->
  `"**Answer**"`, and `opinion-01` ("Do you have feelings?") -> `"---"`.
  Both hit `"\n\n"` right after a bare markdown header/rule fragment,
  before any real content. A small, real cost of the fix -- 0/69 before,
  2/69 after.
- **2/69 still show a trailing `</think>`** in the final text
  (`fact-04`, `yn-02`) -- generation *did* stop where intended (no more
  rambling after it, unlike the 9/69 candidate-config leaks that continued
  for dozens more tokens), but the matched stop string itself is still
  included in what's forwarded to the client rather than stripped. Cosmetic
  compared to the pre-fix version of this same failure, not eliminated.
  Worth a follow-up: strip a trailing configured stop string from the
  final chunk before sending, in `_stream_one`.
- **Markdown/code-fence artifacts (PR #15's other flagged issue) are
  essentially unchanged**: `code_fence` stayed 11/69 before and after,
  since JSON/code responses often start immediately with `` ``` `` and
  never produce a `"\n\n"` before the fence to stop on. `\boxed{}` dropped
  10/69 -> 2/69 and "Step 1:" patterns dropped 5/69 -> 0/69 as a side
  effect (those mostly appeared *after* the paragraph break this fix now
  cuts at), but code fences specifically are a separate, still-open
  problem.

## Experiment C: was the forced-empty-`<think>` fix itself part of the cause? (hypothesis 3)

Confirmed real, using both PR #15's existing before/after data and fresh
reads of it this pass:

| Signal | Baseline (no forced think) | Candidate (PR #15's fix) |
|---|---|---|
| `hit_max_tokens_cap` | 68/69 (99%) | 67/69 (97%) -- essentially unchanged |
| first sentence itself already >30 tok (doesn't even try to be short) | 6/66 (9%) | 15/66 (23%) |
| orphan `</think>` (close tag, no matching open) | 3/69 | 9/69 |
| code fence | 1/69 | 11/69 |
| `\boxed{}` | ad hoc, rare | 10/69 |

Forcing an already-closed empty think block removes the model's ability to
open a *visible* reasoning block (that's PR #15's real, measured win on
think-leakage: 46% -> 13%), but it does not remove the model's underlying
tendency to reason -- it just pushes that tendency into unmarked
markdown/JSON-formatted "structured answer" text, and makes the model
directly hallucinate stray `</think>` tokens mid-response more often (3/69
-> 9/69) since the model's own training distribution still associates
"just finished a `</think>`" with what comes next, and greedy decoding
will occasionally re-emit that exact token sequence spontaneously later in
generation. This matches the live finding surfaced independently earlier
in this session (a real gateway test on the input "Testing 1, 2, 3."
producing a stray mid-response `</think>**Answer**`) -- reproduced here in
`fact-03`'s existing PR #15 data (`... It's important to clarify what
exactly is being referred to. </think>**Romeo and Juliet** (as a
clarification...`), not a one-off.

**This is not a reason to revert PR #15's fix** -- it's a real, if
imperfect, net improvement on its own terms (think-leakage is a worse
defect: a fully visible internal monologue read aloud by TTS, vs. an
occasional stray six-character tag). But it's the reason `"</think>"` is
one of this pass's two stop sequences: it directly targets this specific
side effect, and the supplementary garbled-input check below confirms it
helps.

### Supplementary check: garbled/no-real-question ASR-style inputs

8 short, content-free inputs plausible as real ASR output on noise/silence
(`"Testing 1, 2, 3."`, `"okay okay okay"`, `"mm hmm yeah"`, etc.), run
against the shipped config:

| | Before this pass (est. from main-set adversarial_edge behavior) | Shipped config |
|---|---|---|
| `hit_max_tokens_cap` | most | 1/8 |
| orphan `</think>` artifacts | present (confirmed live) | 0/8 |

`"Testing 1, 2, 3."` specifically -- the exact input that produced the
live `</think>**Answer**` artifact earlier this session -- now returns:
`"To determine the appropriate response, we analyze the given options
through a structured approach."` (17 tokens, one sentence, no artifact).
Re-confirmed through the **actual gateway**, not just the isolated
`nemotron_llm` eval harness: `scripts/test_streaming_client.py` against
`ws://127.0.0.1:18010/ws/stream` with `tests/fixtures/vad_sample_16k.wav`
(real synthesized speech, "Testing one two three.") produced transcript
`"Testing 1, 2, 3."` -> LLM response `"To address the request, we need to
ensure clarity and avoid any unintended implications."` -- a single clean
sentence, real ASR->LLM->TTS round trip, `[turn_end]` reached normally,
first TTS chunk at t+2.75s.

(The content of these responses is still not *useful* -- there's no real
question to answer -- but that's a separate, expected characteristic of
this input class, not the token-cap defect this pass targets.)

## Hypothesis 4: does this checkpoint just have a strong prior toward longer-form reasoning?

**Yes, and this pass's data is consistent evidence for it, not against
it** -- but the practical conclusion isn't "nothing helps." The model's
generation *does* comply with "short" for its first sentence in the
majority of cases (hypothesis 1's finding), which argues against a
maximally strong reading of hypothesis 4 (a checkpoint that ignores length
instructions altogether). What resists override is the *stopping*
behavior specifically: at NVFP4/greedy decode, this checkpoint's training
distribution appears to treat "just answered a question" as a natural
point to continue into verification/reasoning-style elaboration, a pattern
strong enough that PR #9's own attempt to fight it with a stronger prompt
instruction made things worse, and that persists across all combinations
of `repetition_penalty` and forced-empty-think tested this session. A
prompt-only intervention was tried and failed (PR #9); a
sampling-parameter intervention was tried and failed (Experiment A); a
structural/decoding intervention -- exploiting the *empirically observed*
regularity of where the unwanted continuation starts, rather than trying
to convince the model not to want it -- is what actually worked
(Experiment B). That's arguably the most honest read of hypothesis 4: not
"nothing can be done", but "the instruction-following layer this
checkpoint exposes isn't the layer where this particular behavior lives".

## What's shipped

- `triton_model_repo/nemotron_llm/1/model.py`: `stop=["<|eot_id|>", "\n\n",
  "</think>"]` (was `stop=["<|eot_id|>"]`), with inline comments pointing
  at this doc. `repetition_penalty` reverted to its original 1.15 with a
  comment noting Experiment A's result. `max_tokens=96` untouched.
- Live server: reloaded via the safe unload -> `UNAVAILABLE` -> load ->
  `READY` pattern (twice: once for Experiment A, once to revert A and
  apply B in the same reload), `pytest tests/integration -v` 15/15 passing
  after the final reload, confirmed via a real end-to-end gateway audio
  round-trip.
- `RELOAD_LOCK` claimed for the duration of each reload, released
  immediately after.

## What's explicitly NOT fixed / follow-up for whoever picks this up next

1. **5/69 prompts still hit the cap** -- specifically the
   hardest-to-answer categories (ambiguous, multi-part, unfulfillable
   commands, no-good-answer, instruction-probing), where the model never
   produces a paragraph break because it never resolves to anything short
   enough to break after. This is a content-resolution problem, not a
   stopping problem, and this pass's fix structurally cannot reach it.
2. **Code-fence/JSON-formatted responses are untouched** (11/69, unchanged
   by this pass) -- PR #15's other flagged artifact class. These start
   with `` ``` `` immediately, before any `"\n\n"` to stop on. Worth its
   own targeted stop sequence or prompt-level "no markdown, no code, plain
   spoken sentences only" instruction, as its own isolated experiment.
3. **2/69 responses now too-short** (a fragment before real content, e.g.
   `"**Answer**"`) -- a real, if small, new cost of this fix, not
   mitigated.
4. **2/69 responses still leak a trailing `</think>` token** into the
   final text even though generation correctly stopped there. Fix
   candidate: strip a matched stop string from the last streamed chunk in
   `_stream_one` before sending it to the client -- not attempted this
   pass, to keep this experiment to the one isolated `stop=` change.
5. **Content correctness was not rigorously re-measured.** A small
   qualitative spot-check of the `factual` category (n=10, same caveat
   PR #15 used -- not a hard metric) suggests it did not get worse and
   plausibly improved slightly (`fact-03`, "who wrote Romeo and Juliet",
   now correctly says Shakespeare instead of PR #15's hallucinated "no
   such historical figure ever existed") -- shorter responses had less
   room to wander into a wrong tangent after an initially correct
   sentence, but this is a read, not a measured claim.
6. **`whisper_asr`'s transcript for "Testing one two three." renders as
   "Testing 1, 2, 3."** in the confirmation test -- an ASR normalization
   detail (digits vs. words), unrelated to this pass, noted only because
   it's visible in the transcript above and might otherwise look like a
   typo.

## Files touched by this pass

- `triton_model_repo/nemotron_llm/1/model.py` -- the `stop=` change +
  reverted `repetition_penalty` comment (git-tracked, this doc).
- `docs/nemotron-token-cap-investigation.md` -- this document.
- Eval harness (reused from PR #15, unchanged), new analysis scripts
  (`analyze_stop_behavior.py`, `simulate_stop_sequences.py`,
  `artifact_check.py`), and raw results
  (`expA_repetition_penalty_1.0.json`, `expB_stop_sequences.json`,
  `garbled_expB.json`, and PR #15's original baseline/candidate JSON
  carried forward for comparison) live in this session's scratchpad, not
  committed to the repo -- regenerable from the method above, same
  convention PR #15 used.
