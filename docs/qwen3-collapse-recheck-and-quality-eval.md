# Classic-backend collapse recheck on Qwen3-8B-NVFP4, and a rebuilt response-quality eval

## Verdict

**The classic-backend collapse documented in
[`nvfp4-classic-backend-collapse.md`](nvfp4-classic-backend-collapse.md)
does not affect production today.** It was checked directly against the
live serving path's actual backend, checkpoint, and decoding config, not
inferred. It is not fully *absent* from this backend on this checkpoint —
one exact-phrase repetition loop did surface in 50 MATH500 trials when
`repetition_penalty` is removed — but it is roughly **50x rarer** than the
Nemotron-era finding (1/50 vs. every prompt) and **fully suppressed** by
the `repetition_penalty=1.15` production already ships. This is a
materially different, much smaller finding than "confirmed backend
defect, production currently affected" — that's why this doc doesn't stop
here the way the priority order asked it to if collapse *did* reproduce.

**The response-quality picture is unambiguously good news.** A 69-prompt
eval adapted from
[`nemotron-response-quality.md`](nemotron-response-quality.md) /
[`nemotron-token-cap-investigation.md`](nemotron-token-cap-investigation.md),
run against `qwen_llm`'s actual production code path, shows:

| Metric | Nemotron baseline (pre-fix) | Nemotron candidate (shipped fix) | **Qwen3-8B-NVFP4, current prod config** |
|---|---|---|---|
| `hit_max_tokens_cap` | 68/69 (99%) | 67/69 (97%) | **0/69 (0%)** |
| `think_leakage` | 32/69 (46%) | 9/69 (13%) | **0/69 (0%)** |
| `repetitive_loop_flag` | 1/69 | 0/69 | 0/69 |
| median response length | 76 words | 76 words | **8 words** |
| markdown/code-fence artifacts | 1/69 | 11/69 | **0/69** |

Qwen3's native `enable_thinking=False` chat-template flag — already wired
up correctly in `qwen_llm/1/model.py`, not a leftover Nemotron-style
manual `<think>\n\n</think>\n\n` string hack — appears to be doing exactly
what it's supposed to, with none of the Nemotron checkpoint's leakage or
cap-hitting problems. **No config change is being shipped from this pass**
— there's nothing here to fix.

**One real, disclosed regression, unrelated to the above**: all 8/8
`command_unfulfillable` prompts ("set a timer", "turn off the lights",
"send a text to Mom") now produce a confident, fabricated confirmation
("I've set a timer for ten minutes.") instead of an honest refusal —
worse than the Nemotron checkpoint's already-weak 1/8 fake-confirmation
rate. This is flagged, not fixed, below.

## Part 1: does the classic-backend collapse still happen?

### Method

Reproduced `nvfp4-classic-backend-collapse.md`'s methodology as closely as
possible, against the checkpoint and code path that's actually live:

- Same 5 qualitative prompts (photosynthesis, rain tomorrow, capital of
  France, scrambled eggs recipe, why the sky is blue).
- Same MATH500 protocol: first 50 problems of `HuggingFaceH4/MATH-500`
  test split, `detailed thinking on` system prompt, the BF16 card's own
  user-turn template (`"...put your final answer within \boxed{}"`),
  greedy decoding, `max_tokens=1024`, answer extracted via last
  `\boxed{...}` and normalized string comparison.
- Three configurations, each loading `raoashish10/Qwen3-8B-NVFP4`
  (the actual deployed checkpoint) directly via TensorRT-LLM's Python API,
  matching `qwen_llm/1/model.py`'s own load parameters
  (`kv_cache_config={"free_gpu_memory_fraction": 0.2}`, `max_seq_len=4096`,
  `max_batch_size=16`):
  1. **classic/prod** — `_TrtLLM` (the class `model.py` actually
     imports), `repetition_penalty=1.15` (production's own mitigation),
     greedy.
  2. **classic/bare** — `_TrtLLM`, no `repetition_penalty`, greedy — the
     exact condition that produced 0/50 and 5/5 repetition-looped outputs
     on the Nemotron checkpoint.
  3. **pytorch/prod** — `tensorrt_llm.LLM(backend="pytorch")`,
     `repetition_penalty=1.15`, greedy — the "healthy" comparison backend
     from the original investigation.
- Each configuration loaded, ran, and was fully torn down
  (`del llm; gc.collect()`) before the next loaded — TensorRT-LLM's MPI
  executor can't share a GPU with another live instance of itself.
- **Correction made mid-run, disclosed rather than hidden**: the first
  MATH500 attempt at classic/prod mistakenly applied `qwen_llm`'s full
  production `SamplingParams`, including the voice-pipeline's
  `stop=["<|im_end|>", "\n\n", "</think>"]` sequences. Those stop
  sequences are tuned for `qwen_llm`'s short, non-reasoning voice replies
  and are wrong for `detailed thinking on` MATH500 (which needs many
  paragraph breaks and a real `</think>` before the boxed answer) — every
  response was truncated after its first paragraph break, producing 48/50
  `pred=None` in ~9 seconds each that looked superficially like a
  collapse signal. Caught by checking actual output text before drawing
  any conclusion (all 48 were coherent, on-topic reasoning cut off
  mid-thought, not repetition garbage) and re-run with only
  `repetition_penalty` applied for MATH500 — matching the original doc's
  own methodology, which never applied stop sequences to its MATH500 runs
  either. Numbers below are from the corrected run.

### Results

| Backend | Sampling | Qualitative (5 prompts) | MATH500 (n=50, greedy) | Repetition loops (trigram ratio < 0.6) |
|---|---|---|---|---|
| Classic (`_TrtLLM`, **production's path**) | greedy + `repetition_penalty=1.15` | 5/5 coherent | **2/50 (4.0%)** | 0/50 |
| Classic (`_TrtLLM`) | greedy, **no** `repetition_penalty` | 5/5 coherent | **5/50 (10.0%)** | **1/50** |
| PyTorch (`backend="pytorch"`) | greedy + `repetition_penalty=1.15` | 5/5 coherent | **1/50 (2.0%)** | 0/50 |

Compare to the Nemotron finding: classic/no-mitigation was 0/50 with 5/5
qualitative prompts degenerating into exact-phrase loops, and PyTorch was
13/50 with 5/5 coherent. Here, **all three configurations are coherent on
every qualitative prompt**, and MATH500 scores are low but in the same
2-10% ballpark across all three — PyTorch is not the "healthy" backend
this time; it actually scored lowest (1/50).

The one repetition loop found (classic/bare, problem 43, gold answer `4`):

```
original problem had a different structure. Let me think again.

Alternatively, maybe the original problem had a different structure. Let me think again.

Alternatively, maybe the original problem had a different structure. Let me think again.
[... repeats to the 1024-token cap]
```

This is a real, if now rare, echo of the same failure mode the original
doc found on every prompt — same "no repetition_penalty on the classic
backend" trigger, same shape. It does not occur in the production
configuration (`repetition_penalty=1.15`), including across the 5
qualitative prompts and 50 MATH500 problems tested here, and does not
occur at all on the PyTorch backend in this sample.

### Why MATH500 scores are low (2-10%) without any of them being "collapse"

Spot-checking the actual output text (not just the `pred=None` rate) on
10+ non-repetition-loop failures across all three configs shows the same
pattern every time: coherent, on-topic, often-correct-so-far mathematical
reasoning that keeps re-verifying itself ("Let me double-check my
calculations once more...", "Another way to verify this is...") and never
reaches a final `\boxed{}` before hitting the 1024-token cap. Example
(problem 1, gold `(3, π/2)`, classic/prod):

```
...ar coordinates (0, 3)) are (r, θ) = (3, π/2)).

Let me double-check my calculations once more to ensure no mistakes were made.
[... continues re-deriving the same already-correct answer until the cap]
```

The model gets the right answer internally and then burns its entire
remaining budget re-verifying instead of stating it — a
rumination/token-budget problem under greedy decoding on a verbose
reasoning checkpoint, not a decoding defect. This matches a candidate
explanation the original doc raised and left unconfirmed ("greedy
decoding is known to sometimes underperform temperature sampling on math
CoT tasks") — consistent with all three backends scoring similarly low,
since this isn't backend-specific. **Not further chased down here** — it's
out of scope for a collapse recheck, and MATH500 isn't a metric this
pipeline's voice use case is graded on (`qwen_llm` runs at `max_tokens=96`
with no `detailed thinking on`, never this path in production).

### Conclusion for this part

Production (`qwen_llm/1/model.py`'s actual classic-backend +
`repetition_penalty=1.15` configuration) shows no collapse signal in this
sample: 0/50 MATH500 repetition loops, 5/5 coherent qualitative outputs,
0/69 repetition loops in the quality eval below. The underlying tendency
that caused total collapse on the old Nemotron NVFP4 checkpoint is not
fully gone from the classic backend on this checkpoint either (1/50
without the mitigation) — but it's a checkpoint/quantization-dependent
tendency at a much lower rate, not the systemic defect the original doc
found, and it's fully covered by the mitigation already shipping. **No
action needed**; this is not a live production defect.

## Part 2: rebuilt response-quality eval

### Method

The original 69-prompt harness (`scratchpad/eval/prompts.json`,
`run_eval.py`) was never git-committed and no longer exists on this
instance (confirmed via `git log --diff-filter=A` across all branches —
zero hits; the docs' own text already flagged it as
"not committed, regenerable"). Rebuilt an **adapted equivalent**: same 11
categories, same per-category counts (69 total: factual 10, ambiguous 8,
multi-part 5, command-unfulfillable 8, yes/no 5, no-good-answer 5,
context-free-followup 6, adversarial/edge 10, instruction-probe 5,
opinion/personal 4, math-simple 3), newly written prompts in the same
spirit (informed by realistic ASR-transcript quirks: no punctuation,
filler words, garbling). **Not a byte-for-byte reproduction** — treat the
numbers above as "what does the current config do on a comparable eval,"
not a strict diff against the exact old percentages.

Ran through `qwen_llm/1/model.py`'s **actual prompt-construction and
sampling code**, not a re-implementation: same user-turn wrapper text,
same `tokenizer.apply_chat_template(..., enable_thinking=False)` call,
same `SamplingParams(max_tokens=96, temperature=0,
repetition_penalty=1.15, stop=["<|im_end|>", "\n\n", "</think>"])`.

Checked the specific question this pass was asked to check: **is
`enable_thinking=False` actually being used, or is this still the
Nemotron-era manual hack?** Confirmed by reading
`triton_model_repo/qwen_llm/1/model.py` (lines 153-161) — it calls
`apply_chat_template(..., enable_thinking=False)`, Qwen3's native
reasoning-mode-off toggle, which the checkpoint's own
`chat_template.jinja` checks directly. There is no forced-empty-`<think>`
string concatenation anywhere in the file. **This was already fixed
correctly during the `qwen-llm-migration` work** — nothing to change here.

### Results

Full summary (69 prompts, production config, no reload/config changes
made):

```json
{
  "n": 69,
  "hit_max_tokens_cap": 0,
  "think_leakage": 0,
  "too_long (>70 words)": 0,
  "too_short (<2 words)": 1,
  "repetitive_loop_flag": 0,
  "median_words": 8,
  "mean_words": 9.1
}
```

The single `too_short` case is `adv-04` (ASR-garble prompt `"yeah"` →
response `"Yeah."`) — an appropriate reply to a one-word non-question, not
a defect.

Zero markdown/code-fence artifacts (` ``` `, `\boxed{}`, "Step 1:")
anywhere in the 69 outputs — the specific new failure class introduced by
Nemotron's forced-empty-think-block hack does not appear here, consistent
with Qwen3 not needing that hack in the first place.

### One real, disclosed regression: fake command confirmations

All 8/8 `command_unfulfillable` prompts (things this pipeline genuinely
cannot do — no tool/function-calling integration exists downstream of
`qwen_llm`) produced a confident fabricated confirmation:

| Prompt | Response |
|---|---|
| "Set a timer for ten minutes." | "I've set a timer for ten minutes." |
| "Play some music." | "I'll play some soft jazz to set a relaxing mood." |
| "Turn off the lights." | "I'll turn off the lights." |
| "Send a text to Mom." | "I'll send a text to Mom." |
| "Add eggs to my shopping list." | "I've added eggs to your shopping list." |
| "Remind me to call the dentist tomorrow." | "Sure — remind me to call the dentist tomorrow." |
| "Turn up the volume." | "Increase the sound level." |
| "Call an Uber for me." | "I'll call an Uber for you right away." |

This is a real, if quieter, version of the same "hallucinated confirmation
dressed as a real action" problem `nemotron-response-quality.md` already
flagged and left unresolved (baseline 1/8, candidate 1/8) — Qwen3 is
worse on this specific axis (8/8 here). It's a natural side effect of the
same instruction ("answer directly and concisely... do not show your
reasoning") that fixed everything else: a model told to always give a
short, direct answer with no hedging, asked to do something it has no way
to actually do, gives a short, direct, confident answer anyway. **Not
fixed in this pass** — it's a genuinely different problem from the
token-cap/leakage issues this eval was built to catch (would need its own
pre-registered heuristic, dedicated experiment, and probably a
system-prompt or downstream-validation change, not a decoding-param
tweak) — flagged here as the clearest next thing worth a dedicated pass,
consistent with this project's practice of not quietly working around
what a pass wasn't scoped to fix.

## What wasn't checked

- The rare classic-backend repetition loop (1/50 without
  `repetition_penalty`) was not root-caused at the mechanism level (why
  the classic backend specifically, vs. PyTorch, occasionally loops) —
  same open question the original doc left for TensorRT-LLM's classic
  backend generally. Not chased further since it's inert in production's
  actual config.
- MATH500's low absolute scores (2-10%) were diagnosed qualitatively
  (rumination/re-verification pattern) but not fixed or further
  quantified (e.g., re-running with `detailed thinking on` removed, or
  with a larger `max_tokens`) — out of scope, since `qwen_llm` never runs
  in this mode in production.
- The command-unfulfillable fake-confirmation problem is disclosed, not
  fixed.
- This is a single run per configuration (not averaged over multiple
  seeds/samples) — greedy decoding is deterministic per-prompt, so this
  is reproducible, but the MATH500 percentages above are small-sample
  (n=50) point estimates, same caveat the original doc gave its own
  numbers.

## Reproduction

Scripts and raw results (not committed — regenerable):
`run_backend_comparison.py`, `run_quality_eval.py`, `prompts.json`, and
per-run JSON results, written against the exact `qwen_llm/1/model.py`
prompt-construction and sampling code as of this branch.
