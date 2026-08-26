# nemotron_llm NVFP4 checkpoint: classic TensorRT backend causes near-total generation collapse

## Verdict

**Confirmed: this is a defect in TensorRT-LLM's classic/JIT `_TrtLLM`
backend (the one `nemotron_llm/1/model.py` actually uses in production),
not a bad NVFP4 quantization.** The checkpoint itself, loaded through
TensorRT-LLM's PyTorch backend instead, produces coherent output and a
much higher MATH500 score. **Production is currently serving through the
broken path** — `nemotron_llm`'s real-world output quality is worse than
anyone benchmarking it against the BF16 model card's numbers would expect,
for reasons that have nothing to do with the 4-bit quantization itself.

| Backend | Sampling | Qualitative output | MATH500 pass@1 (n=50, greedy, single run) |
|---|---|---|---|
| Classic TensorRT (`_TrtLLM`, **production's path**) | greedy, no repetition penalty | All 5 test prompts degenerated into exact-phrase repetition loops | **0/50 (0.0%)** |
| Classic TensorRT (`_TrtLLM`) | greedy + `repetition_penalty=1.15` (production's own existing mitigation) | Less exact looping, still rambling/off-topic, one outright incoherent output | **0/50 (0.0%)** |
| PyTorch (`tensorrt_llm.LLM(backend="pytorch")`) | greedy + `repetition_penalty=1.15` | All 5 coherent, correct, well-formed | **13/50 (26.0%)** |

For reference, the BF16 base model's own published MATH500 pass@1 (reasoning
on) is 96.2%, averaged over 16 sampled runs at temperature 0.6 / top_p 0.95
— a different methodology from the single greedy run here, so the two
numbers aren't directly comparable. But **0% vs. 26% from switching only
the backend, with identical weights, tokenizer, prompts, and sampling
params, isn't explainable by quantization loss** — it's the backend.

The remaining gap between 26% and the BF16 baseline is real and
**unresolved** — see "What this doesn't explain" below.

## Why this was investigated

While preparing this NVFP4 checkpoint for a Hugging Face push, a sanity
check against the live `nemotron_llm` Triton service (5 real prompts, then
a 50-problem MATH500 benchmark) showed severely degraded output: every
qualitative prompt degenerated into repetition loops, and MATH500 scored
0/50. Before concluding the quantization was bad and abandoning the
checkpoint, the disambiguating question was: does this reproduce on a
different TensorRT-LLM backend against the exact same checkpoint files?

## Method

1. **Environment fix first.** The initial benchmark attempts failed with a
   low-level TensorRT builder crash
   (`pybind11::init(): factory function returned nullptr`, CASK kernel
   library assertion) — reproducible even with a bare
   `tensorrt.Builder()` call and a fully idle GPU, and independently
   reproduced by the live Triton service itself (both `nemotron_llm` and
   the unrelated `whisper_asr` TensorRT-LLM model failed to load). Root
   cause: `/venv/main`'s `tensorrt` package had been silently downgraded to
   `10.13.3.9.post1` from the pinned, working `10.14.1.48.post1` (recorded
   in `deploy/requirements-main.txt`) — a side effect of a concurrent,
   unrelated dependency change in progress on this shared instance
   (`kokoro-onnx`/`onnxruntime` removal). Reinstalling the pinned
   `tensorrt==10.14.1.48.post1` (+ matching `tensorrt_cu13*` packages)
   fixed the builder crash outright. This is a separate, already-resolved
   issue — noted here only because it's a prerequisite for every result
   below being trustworthy.
2. **Qualitative check**: 5 realistic prompts (factual, arithmetic,
   explanatory, everyday-advice, no-good-answer), plain chat template
   (`detailed thinking off`), no production voice-pipeline wrapping
   (no forced empty `<think>` block, no 96-token cap, no custom stop
   strings) — max_tokens=300.
3. **MATH500 benchmark**: first 50 problems of
   `HuggingFaceH4/MATH-500`'s test split, `detailed thinking on` system
   prompt, the same user-turn template as the BF16 card's own MATH500
   entry (`"...Your final answer should be in \boxed{}...")`, greedy
   decoding, max_tokens=1024. Answer extracted via last `\boxed{...}` in
   the completion, normalized string comparison against the dataset's
   `answer` field. Single greedy run — **not** the BF16 card's 16-run
   averaged sampling methodology, so treat the 26% figure as a lower-bound
   sanity signal, not a rigorous benchmark reproduction.
4. All three configurations loaded the identical checkpoint directory
   (`models/Llama-3.1-Nemotron-Nano-4B-v1.1-NVFP4`) directly via
   TensorRT-LLM's Python API — `tensorrt_llm.llmapi.llm._TrtLLM` for the
   classic backend (what `nemotron_llm/1/model.py` calls), and
   `tensorrt_llm.LLM(..., backend="pytorch")` (the public, non-underscore
   class, default backend `"pytorch"`) for the PyTorch backend. No new
   package installs — both classes ship in the already-installed
   `tensorrt_llm==1.2.1`.
5. The live Triton service was stopped for the duration of each benchmark
   run (TensorRT-LLM's MPI-based executor cannot share a GPU with another
   instance of itself — attempting to run both at once produced an
   indefinite hang, not a clean error) and restarted immediately after.

## Example outputs

**Prompt: "Can you explain what photosynthesis is?"**

- Classic backend, no repetition penalty: `"...the process of photosynthesis requires oxygen, and therefore the answer is that oxygen is required for photosynthesis, and therefore the answer is that oxygen is required for photosynthesis..."` (repeats to the token limit)
- Classic backend + repetition_penalty=1.15: coherent for one paragraph, then drifts into unrelated dietary-supplement / "checking food groups" tangents
- PyTorch backend + repetition_penalty=1.15: a correct, complete, well-structured explanation with location, inputs, and light/dark reaction stages — no looping, no drift

**Prompt: "Is it going to rain tomorrow?"**

- Classic backend + repetition_penalty=1.15: spirals into a hallucinated, unrelated "innovative business practices" sub-problem mid-response
- PyTorch backend + repetition_penalty=1.15: correctly declines to predict the weather and suggests checking a real forecast — the appropriate answer

## What this doesn't explain

26% MATH500 (PyTorch backend) is still far below the BF16 card's 96.2%
figure, and that gap has **not** been root-caused here. Candidate
explanations, none confirmed:

- Methodology: single greedy run vs. 16-run averaged sampling at
  temperature 0.6 — greedy decoding is known to sometimes underperform
  temperature sampling on math CoT tasks for other models.
- `repetition_penalty=1.15` (carried over from production's classic-backend
  workaround) may itself be actively harmful on the PyTorch backend for
  math problems, where legitimate short-range token repetition (repeated
  digits, repeated algebraic terms) is common and penalizing it could
  induce the observed high rate of "never reaches `\boxed{}`" failures
  (`pred=None` dominates the wrong answers, more than wrong-but-boxed
  answers).
- Some genuine accuracy loss from NVFP4 quantization is expected and
  plausible; 26% vs. 96% is a bigger drop than typical for this precision
  point, which is why "quantization is fine, decoding config just needs
  tuning on the PyTorch backend" should be verified, not assumed.

**Next step to close this out**: re-run the same MATH500 subset on the
PyTorch backend without `repetition_penalty`, and separately with the
BF16 card's own sampling recipe (temperature 0.6, top_p 0.95, multiple
samples). If either materially closes the gap, the remaining story is
"decoding config," not "quantization." If neither does, revisit the
calibration (`--kv_cache_qformat`, calibration dataset, `--calib_size`) in
`scripts/quantize_nvfp4.py`.

## Operational implication (separate from the Hugging Face checkpoint question)

This is not just a benchmarking footnote: `nemotron_llm/1/model.py` in
this repo serves the **classic** TensorRT backend (`_TrtLLM`) in
production right now. Every finding above about repetition loops and
0% MATH500 applies to what the live voice pipeline is actually generating
for real requests today, independent of whatever gets pushed to Hugging
Face. Switching `model.py` to `tensorrt_llm.LLM(backend="pytorch")` is a
real, separate fix worth its own PR and its own regression testing
(latency, VRAM footprint, and streaming/`generate_async` API parity with
the classic backend haven't been checked here — this investigation only
used the batched, non-streaming `generate()` call).
