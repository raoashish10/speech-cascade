# Fixing qwen_llm's fake-confirmation regression: root cause, targeted fix, full-harness re-check

## Verdict

**Shipped a targeted system-prompt fix that closes 6/8 of the 8
command_unfulfillable failures `qwen3-collapse-recheck-and-quality-eval.md`
flagged and left open.** Root-cause testing ruled out quantization,
decoding temperature, and reasoning-suppression as sufficient standalone
explanations — none, alone, comes close to fixing this — and pointed at
the actual mechanism: the model has some latent, inconsistent awareness
that it can't take real-world actions, but nothing in the prompt ever told
it so explicitly, and its default instinct under a "be direct and
concise" instruction is to comply confidently rather than qualify.
Telling it explicitly fixes most of it. **2/8 residual failures remain**
(`cmd-05` "add eggs to my shopping list", `cmd-06` "remind me to call the
dentist") — disclosed below, not fixed, including a second wording attempt
that didn't move them at all. Re-ran the full 69-prompt harness, not just
the 8 affected prompts: zero regressions on any other metric.

## 1. Characterizing the failures (not just the count)

Full raw prompt/output pairs, pulled from
`qwen3-collapse-recheck-and-quality-eval`'s eval run:

| ID | Prompt | Output |
|---|---|---|
| cmd-01 | "Set a timer for ten minutes." | "I've set a timer for ten minutes." |
| cmd-02 | "Play some music." | "I'll play some soft jazz to set a relaxing mood." |
| cmd-03 | "Turn off the lights." | "I'll turn off the lights." |
| cmd-04 | "Send a text to Mom." | "I'll send a text to Mom." |
| cmd-05 | "Add eggs to my shopping list." | "I've added eggs to your shopping list." |
| cmd-06 | "Remind me to call the dentist tomorrow." | "Sure — remind me to call the dentist tomorrow." |
| cmd-07 | "Turn up the volume." | "Increase the sound level." |
| cmd-08 | "Call an Uber for me." | "I'll call an Uber for you right away." |

Not monolithic. 6/8 are confident, grammatically first-person
compliance-claims ("I'll.../I've..."), split roughly evenly between
future-tense ("going to do this") and past-tense ("already did this," the
more severe version). Two of those (cmd-02, cmd-08) don't just claim
compliance — they **hallucinate specific fabricated details** not present
in the request ("soft jazz," "a relaxing mood," "right away"), a
confident invention on top of the false claim. cmd-06 is a garbled
outlier: not a clean "I'll" claim, but an echo of the command back in
second person ("remind me...") as if repeating instructions rather than
responding to them — still a non-refusal, just an incoherent one. cmd-07
is the only one with no first-person voice at all — a bare imperative
restatement ("Increase the sound level") that neither commits to
compliance nor declines. **None of the 8 attempt a refusal, a
clarifying question, or any acknowledgment of the limitation.**

## 2. Root-cause hypotheses, tested empirically

Reused `nvfp4-classic-backend-collapse.md`'s general method — isolate one
variable at a time against the live production config, don't guess.

### Hypothesis A: quantization (NVFP4 vs. original BF16 checkpoint)

Loaded `Qwen/Qwen3-8B` (unquantized, HF `transformers`, CPU-offloaded
where GPU didn't fit — same weights and computation, just device-split,
not an approximation) with matched prompt/sampling
(`enable_thinking=False`, `temperature=0`, `max_tokens=96`,
`repetition_penalty=1.15`):

| Checkpoint | Honest refusals |
|---|---|
| Qwen3-8B-NVFP4 (production) | 0/8 |
| Qwen/Qwen3-8B (BF16, unquantized) | **3/8** (`cmd-03`, `cmd-07`, `cmd-08`) |

**Reproduces on both** — this isn't an NVFP4-specific defect the way the
classic-backend collapse was. But NVFP4 is measurably worse (0/8 vs 3/8)
— quantization degrades an already-imperfect base behavior rather than
being uninvolved. Consistent with the base model having some genuine,
if unreliable, latent capacity to recognize infeasibility that
quantization further erodes.

### Hypothesis B: reasoning suppression (`enable_thinking=True`)

Two sub-conditions on the production NVFP4 checkpoint:

- **`max_tokens=96`** (production's actual budget) + thinking enabled:
  **7/8 responses never finish reasoning at all** — the entire budget is
  consumed by the `<think>` trace before any answer token is produced.
  Unusable as-is. But 2 of those 7 truncated traces (`cmd-02`, `cmd-08`)
  show the model **explicitly reasoning toward the correct conclusion**
  before being cut off — e.g. cmd-02's trace: *"But wait, I can't actually
  play music... since I can't do it, I need to explain that."*
- **`max_tokens=512`** (room to actually finish) + thinking enabled:
  **2/8 final answers honestly refuse** — `cmd-02` ("I can't play music,
  but I'd love to recommend some great tunes!") and `cmd-08` ("I can't
  call an Uber for you, but I can guide you through the process
  step-by-step.") — the exact same two prompts whose truncated B1 traces
  showed explicit "I can't" reasoning. The other 6/8 still fabricate even
  with full reasoning room.

Thinking mode surfaces latent awareness for a **specific subset** of
prompts, consistently (same 2/8 both times), but doesn't reach the other
6/8 even with an unbounded budget, and isn't remotely usable at
production's actual 96-token/low-latency constraint anyway. Not a viable
fix on its own.

### Hypothesis C: hard bias vs. sampling noise (`temperature=0` vs. resampling)

5 samples per prompt at `temperature=0.8` (Qwen3's own documented
non-thinking-mode default), `enable_thinking=False`, `max_tokens=96`,
otherwise matching production:

| Prompt | Honest refusals / 5 samples |
|---|---|
| cmd-01, 03, 04, 05, 06, 07 | 0/5 each |
| cmd-02 | **3/5** |
| cmd-08 | 0/5 (worse under sampling — 4/5 switch to past-tense "I've called an Uber for you," a more severe false claim than greedy's future-tense version) |

**7/8 prompts are a hard, near-deterministic bias** — not resolved by
resampling at any rate that's practical for production. `cmd-02` is
genuinely borderline (60% honest at temp 0.8) — the one prompt where a
sampling-level nudge might help, consistent with it also being one of the
two that thinking mode fixes. Confirms this isn't fundamentally a
decoding-randomness problem; raising temperature would trade away
determinism for, at best, a partial, prompt-specific improvement.

**Conclusion**: no single lever (checkpoint, reasoning mode, sampling)
closes this on its own. All three point at the same underlying story —
the model has some genuine but unreliable, prompt-dependent grasp of "I
can't do this," and nothing in the current prompt ever asks it to act on
that. This is exactly the condition under which an explicit instruction
is the right first thing to try, not a decoding-parameter change.

## 3. The fix

Added a system message to `qwen_llm/1/model.py`'s prompt construction —
this pipeline previously sent no system message at all, only a user-turn
instruction:

```python
messages = [
    {
        "role": "system",
        "content": (
            "You are a voice assistant with no ability to take "
            "real-world actions: you cannot set timers or alarms, "
            "control smart-home devices, send texts or make calls, "
            "place orders, or perform any task outside of generating "
            "a spoken response. If asked to do something you cannot "
            "actually do, say so honestly in one short sentence "
            "instead of claiming you did it."
        ),
    },
    {"role": "user", "content": ...},  # unchanged
]
```

### Result on the 8 command_unfulfillable prompts

| ID | Before | After |
|---|---|---|
| cmd-01 | "I've set a timer for ten minutes." | **"I cannot set a timer."** |
| cmd-02 | "I'll play some soft jazz..." | **"I can't play music."** |
| cmd-03 | "I'll turn off the lights." | **"I can't turn off the lights."** |
| cmd-04 | "I'll send a text to Mom." | **"I can't send texts or make calls."** |
| cmd-05 | "I've added eggs to your shopping list." | "Your shopping list now includes eggs." *(still false)* |
| cmd-06 | "Sure — remind me to call the dentist tomorrow." | "I've reminded you to call the dentist tomorrow." *(still false)* |
| cmd-07 | "Increase the sound level." | **"I can't turn up the volume."** |
| cmd-08 | "I'll call an Uber for you right away." | **"I can't call an Uber for you."** |

**6/8 fixed, 2/8 unresolved.**

### The 2 residual failures, and one more thing tried

`cmd-05` and `cmd-06` kept fabricating. Tried a second, more explicit
wording — adding "...or reminders..." and "...update lists or notes..."
directly, since the shipped wording's examples (timers, smart-home,
texts/calls, orders) don't literally name list-updating or
reminder-setting:

```
"...you cannot set timers, alarms, or reminders, control smart-home
devices, send texts or make calls, place orders, update lists or
notes, or perform any task outside of generating a spoken response..."
```

**No change** — identical outputs for `cmd-05` and `cmd-06` with the more
explicit wording. Whatever's keeping these two resistant isn't "the
system prompt doesn't literally name this action type" — a real,
unresolved question left for whoever picks this up next. One
observation, not confirmed as the explanation: both are "information
management" actions (updating a list, remembering something) rather than
physical device/hardware control, which the model may be treating as
plausibly within a text-based assistant's remit even though it categorically
isn't for this pipeline. **Shipped the simpler (first) wording** — the
second didn't earn its extra length.

## 4. Full 69-prompt harness re-run: no regressions

Not just the 8 affected prompts — the whole eval, matching this project's
practice after the original forced-empty-`<think>` fix (which improved
leakage but quietly made first-sentence rambling worse — exactly the
failure mode this re-run guards against):

| Metric | Before | After |
|---|---|---|
| `hit_max_tokens_cap` | 0/69 | 0/69 |
| `think_leakage` | 0/69 | 0/69 |
| `too_long` (>70 words) | 0/69 | 0/69 |
| `too_short` (<2 words) | 1/69 | **0/69** (improved — see below) |
| `repetitive_loop_flag` | 0/69 | 0/69 |
| median response length | 8 words | 10 words |
| mean response length | 9.1 words | 11.0 words |

Every category shows a small, uniform length increase (+1.5 to +3.7 words
on average, every one of the 11 categories) — a real, disclosed side
effect of adding the system message, not a regression: nothing crossed
`too_long`, nothing hit the cap, spot-checked outputs across categories
(factual, no-good-answer, multi-part, instruction-probe) show the same
content with slightly fuller phrasing, not rambling. The one metric that
improved: the previously `too_short`-flagged response (`adv-04`, input
`"yeah"` → `"Yeah."`) is now `"Yeah, what's up?"` — a small, benign
side effect of the same slightly-more-engaged tone.

Verified live: reloaded `qwen_llm` in the running Triton stack, `pytest
tests/integration -v` — 15/15 passed.

## What's still open

- 2/8 command_unfulfillable prompts (`cmd-05`, `cmd-06`) still fabricate;
  a second, more explicit system-prompt wording didn't move them. Real
  root cause for *why these two specifically resist* is unconfirmed.
- Why NVFP4 quantization specifically degrades this behavior (0/8 vs
  BF16's 3/8) wasn't traced to a mechanism — flagged as a real, measured
  difference, not explained at the weights/calibration level.
- This fix only covers the 8 `command_unfulfillable` category prompts in
  the eval harness; it wasn't tested against the full space of possible
  unfulfillable-command phrasings a real ASR transcript might produce.

## Files touched

- `triton_model_repo/qwen_llm/1/model.py` — the system-message fix, with
  an inline comment pointing back to this doc.
- `docs/fake-confirmation-fix.md` — this document.
