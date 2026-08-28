# NVFP4 LLM candidate investigation

Status: **COMPLETE — recommendation reached.** See "Final verdict" at the
bottom. Investigation-only, per the brief: `nemotron_llm` has not been
touched in production.

## Scope recap

Selecting a replacement for `nemotron_llm`
(`Llama-3.1-Nemotron-Nano-4B-v1.1-NVFP4`), evaluated against:
`docs/nvfp4-classic-backend-collapse.md`), NVFP4-quantized (W4A16, SM120),
must fit 16GB alongside ASR+TTS, evaluated on MMLU + IFEval + voice-fitness +
latency + concurrent-load bandwidth contention (Gate 4). See
`/workspace/task1-llm.md` for the full brief.

## Pre-flight: environment state (verified 2026-08-27)

This is a **fresh Vast.ai instance** — it inherited this repo's git history
but none of the actual runtime state:

- No ML stack installed at boot (no torch, tensorrt_llm, modelopt, lm_eval).
- No HF token configured (`credentials.huggingface: false`) — blocks the
  gated `nvidia/Nemotron-Post-Training-Dataset-v2` calibration set and gated
  model repos (e.g. `meta-llama/Llama-3.1-8B` requires license acceptance).
- `/workspace` is **not** a persistent volume — nothing here survives
  recycle/destroy except what's pushed to git or synced to S3.
- Recovered the actual production artifacts via
  `aws s3 sync s3://ashish-s3-coding-bucket/speech-cascade-inference/` (per
  `deploy/REBUILD.md` step 2) instead of rebuilding from scratch — this
  restores the real BF16/FP8/NVFP4 checkpoints, the exact frozen venv
  (`deploy/requirements-main.txt`: torch==2.9.1+cu128, tensorrt_llm==1.2.1,
  nvidia-modelopt==0.37.0), and the Triton server binary. In progress at
  time of writing.

## Measured production VRAM baseline (from README.md, cross-checked)

| Stage | VRAM |
|---|---|
| Total GPU | 16303 MiB |
| `nemotron_llm` alone (current 4B, classic backend) | ~8185 MiB (3578 MiB engine weights + 1597 MiB KV cache pool @ 0.2 free-fraction cap + ~3010 MiB framework/MPI-worker overhead) |
| All three model-serving stages loaded (nemotron_llm + whisper_asr + TTS) | ~9933 MiB (5910 MiB free) |

Implied ASR+TTS combined footprint: ~1748 MiB. **Caveat**: the README's "all
three loaded" figure may predate the Kokoro→Magpie swap (`f440a32`) — not
re-verified as current in this pass. Will re-measure directly once the
service is up rather than trust this number blindly.

Budget for a new LLM candidate to preserve the same ~5.9GB headroom the
current deployment has: **≤ ~8.6GB total footprint**. Hard ceiling (no
headroom left at all beyond a ~1-1.5GB safety margin for OS/Triton core):
**≤ ~13GB**.

## Gate 1 — paper fit-check (all 4 candidates, cheap, before spending quantization time)

Methodology: weight bytes ≈ (dense params − embed/lm_head params) × 0.5B
(W4A16) + embed/lm_head kept at native precision (matches how the deployed
Nemotron checkpoint handles `lm_head`/`embed_tokens` — see
`docs/kv-cache-investigation.md`) + ~5-10% NVFP4 block-scale overhead.
Framework overhead assumed ~3GB, anchored to the classic-backend measurement
above — **explicitly flagged as uncertain**: the brief requires evaluation on
the **PyTorch backend**, which may have materially different overhead
(no MPI-worker duplication, but different activation/CUDA-graph buffers) —
must be confirmed empirically at Gate 2, not assumed.

| Candidate | Params | Est. weight footprint | + ~3GB overhead (unverified for PyTorch backend) | vs. 8.6GB soft / 13GB hard budget |
|---|---|---|---|---|
| 1. Qwen3-14B dense | 14.8B | ~9.0–10.5GB | ~12–13.5GB | **At/over hard ceiling — high risk, paper estimate says tight fit at best** |
| 2. Qwen3-8B dense | 8.2B | ~5.3–6.3GB | ~8.3–9.3GB | Fits hard ceiling; slightly over soft (same-headroom) budget |
| 3. Qwen2.5-7B-Instruct | 7.6B | ~5.7GB | ~8.7GB | Comparable to current model's footprint |
| 4. Llama-3.1-8B | 8.0B | ~5.9GB | ~8.9GB | Comparable to current model's footprint; **gated on HF — needs a token with license acceptance, not yet available** |

KV cache impact at this pipeline's real usage pattern (~150 token contexts)
is negligible for all four candidates (<1% of per-step traffic, consistent
with `docs/kv-cache-investigation.md`'s finding for the current model) — not
a fit-check driver, left out of the table above.

**Gate 1 verdict**: none of the four candidates are so obviously oversized
that they should be eliminated on paper alone (per the brief: "rule out
before spending quantization time if it clearly won't fit" — candidate 1 is
tight but not *clearly* over). Proceeding top-down as instructed, starting
with candidate 1, with candidate 2 as the realistic expected landing spot
given the paper math above.

## Gate 2+ progress log

### Candidate 1: Qwen3-14B dense

**Environment setup (complete)**: restored production venv exactly
(`deploy/requirements-main.txt` via `--no-deps` to avoid a real pip resolver
conflict between `librosa==1.0.0`'s stated numpy>=2.1.0 requirement and the
`numpy==1.26.4` pin TensorRT-LLM's compiled bindings need — matches
README.md quirk #10's own warning about this exact class of problem).
Confirmed `tensorrt_llm==1.2.1` imports cleanly. Built a **separate isolated
venv** (`/venv/quantize`) for the quantization step itself — `nvidia-modelopt
[hf]==0.46.0` (needed to match the pinned `hf_ptq.py` script version) pulls a
newer `torch` that conflicts with `tensorrt_llm`'s `torch==2.9.1` pin;
running quantization in its own venv keeps `/venv/main` (the serving
environment) untouched. This mirrors the original session's own pattern
("built a fresh isolated venv for TensorRT-LLM benchmarking").

**Quantization (complete)**: downloaded `Qwen/Qwen3-14B` BF16 (28GB,
ungated). Ran `hf_ptq.py` with `--qformat nvfp4 --kv_cache_qformat none`
(same recipe as the deployed Nemotron checkpoint — `hf_quant_config.json`
diffed identical in shape) — **adapted from the brief's calibration recipe**:
used `--dataset cnn_dailymail` alone (512 samples) instead of the
`cnn_nemotron_v2_mix` combo, since no `HF_TOKEN` is available on this
instance to access the gated `nvidia/Nemotron-Post-Training-Dataset-v2` half
of that mix. **The BF16 source model (28GB) did not fit in 16GB VRAM** —
used `--use_seq_device_map --gpu_max_mem_percentage 0.7` (accelerate
CPU-offload for the portion that doesn't fit), confirmed working, peak GPU
memory during calibration 12.5GB. Calibration took ~12 minutes. Output:
9.9GB NVFP4 checkpoint. Deleted the BF16 source immediately after (disk
constrained on this instance — 100GB total, no persistent volume).

**A real bug hit and fixed**: the exported checkpoint's `tokenizer_config.json`
was regenerated by the quantize venv's newer `transformers` (5.14.1) in a
schema `tensorrt_llm`'s pinned older `transformers` (4.57.3, in `/venv/main`)
can't parse (`'list' object has no attribute 'keys'` — the new schema's
`extra_special_tokens` is a list, the old loader expects a dict; the new
export also silently dropped `added_tokens_decoder` and `chat_template`
entirely). **This is the same class of cross-version tokenizer metadata bug
`reports/session-report.md` documents hitting during the original Nemotron
quantization** (different specific field, same root cause: quantization-time
and serving-time `transformers` versions disagree on the checkpoint's
tokenizer metadata schema). Fixed by re-downloading the four small tokenizer
sidecar files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`,
`merges.txt`) from the original `Qwen/Qwen3-14B` repo and overwriting the
quantize-venv-regenerated versions — weights are untouched by this, tokenizer
files don't change during quantization, so this is metadata-only, matching
the session report's own precedent for this exact class of fix.

**Gate 2 — build + load + coherence: PASSED.** Served via
`trtllm-serve serve --backend pytorch --max_seq_len 4096 --max_batch_size 8
--free_gpu_memory_fraction 0.2` (matching the deployed Nemotron config's own
limits for a fair comparison). Loaded successfully; standalone footprint
**12.0GB** (close to the Gate-1 paper estimate of 12-13.5GB). Tested via the
OpenAI-compatible chat completions endpoint:

- Default request (no `enable_thinking` override): **reproduces a
  Nemotron-like failure mode** — the entire 96-token budget was consumed by
  `<think>...</think>` reasoning content, `finish_reason: "length"`, zero
  actual answer produced. Qwen3 ships with thinking-mode on by default.
- With `chat_template_kwargs: {"enable_thinking": false}` (the model's
  built-in, documented non-thinking toggle — confirmed present in this
  checkpoint's own `chat_template.jinja`): clean, correct, concise answers,
  `finish_reason: "stop"` on simple factual/conversational prompts.
- **Still worth flagging for Gate 3**: with only a bare user turn (no system
  prompt), non-thinking-mode responses can still run long and
  markdown-formatted (numbered lists, headers) — hit the 96-token cap on a
  "help me reschedule an appointment" prompt despite non-thinking mode being
  on. With a short voice-assistant system prompt ("respond in 1-2 short
  spoken sentences, no lists, no markdown") added, the same prompt produced
  a clean 20-token, naturally-stopping, speech-appropriate answer.
  **Conclusion so far**: Qwen3-14B does NOT reproduce Nemotron's failure mode
  when configured correctly (non-thinking mode + a voice-appropriate system
  prompt, mirroring how `nemotron_llm`'s `model.py` already forces an empty
  `<think></think>` block) — but *does* reproduce it if deployed naively.
  This is a real, actionable integration requirement for whoever eventually
  wires this in, not just a benchmarking footnote.

**Gate 3 — quality (MMLU + IFEval via lm-eval-harness, real numbers)**:

Ran against the same live PyTorch-backend server via lm-eval-harness's
`local-chat-completions` model type (`--apply_chat_template`,
`chat_template_kwargs={"enable_thinking": false}`, temperature=0).

- **A real scoring bug hit and fixed, same class as the brief's own MATH500
  warning**: the first `mmlu_generative` run scored **0.0% on every single
  subject** (57/57) — mathematically implausible for a real model, so
  investigated rather than reported. Root cause: `mmlu_generative`'s stock
  prompt template (`doc_to_text: "...Answer:"`, `until: ["\n"]`) is written
  for base/completion-style models continuing a few-shot prompt with a bare
  letter — not for a chat-templated instruct model, which naturally answers
  in prose ("The answer is C.") that the task's `exact_match` filter (first
  line only) never matches. Fixed by adding a `system_instruction` telling
  the model to answer with only the letter — confirmed by hand via curl
  first (`"C"` alone, no prose) before rerunning the full eval. This is
  exactly the class of harness/scoring-methodology bug the brief warns about
  re: MATH500's `\boxed{}` parsing — verify surprising benchmark numbers
  against raw output before trusting them.
- **`mmlu_generative`** (57 subjects, 20 samples/subject after the fix,
  n=1140 total — capped for time; not the full ~14k-question test set):
  **77.5%** overall (stem 74.7%, other 75.0%, social sciences 83.75%,
  humanities 78.5%).
- **`ifeval`** (n=100 of 541): **79.0%** prompt-level strict accuracy,
  **85.3%** inst-level strict accuracy (81.0% / 87.1% loose). This is the
  brief's heavily-weighted axis (verifiable instruction-following, the exact
  failure class — think-leakage, rambling — the current 4B struggles with),
  and it's a strong result.
- Both runs used `enable_thinking: false` (see Gate 2) — these numbers
  reflect the non-thinking configuration this model would actually need to
  run in production, not the (worse, slower) thinking-mode default.

**Voice-fitness check (n=20 representative voice-assistant prompts, with a
short voice-appropriate system prompt + `enable_thinking: false`,
`max_tokens=96` matching production)**: **100% clean-stop rate (20/20),
0% token-cap-hit, median response length 26 completion tokens.** This is a
dramatic contrast with the current 4B's documented 97% token-cap-hit rate —
when configured correctly (non-thinking + a voice system prompt, see Gate 2),
Qwen3-14B does not reproduce that failure mode at all on this sample.

**Standalone latency** (10 prompts, streaming, PyTorch backend, unbatched):
TTFT p50 = **0.032s**, tokens/s p50 = **80.7**, total p50 = **0.286s**. Fast
in isolation — the open question is what happens under real pipeline
concurrency (next).

**Gate 4 — concurrent GPU footprint: FAILED.** This is where candidate 1
is eliminated. Measured on this instance's actual RTX 5070 Ti (15.47GiB
usable, not the full 16GB nominal):

- Qwen3-14B-NVFP4 standalone (PyTorch backend, `free_gpu_memory_fraction=0.2`
  KV cache cap, matching production): **12.5GB** (matches the Gate-1 paper
  estimate of 12-13.5GB closely).
- Attempted to load `magpie_tts` (the model actually in the production
  Triton repo today, per the flagged discrepancy in
  `docs/tts-replacement-investigation.md`) alongside it: **CUDA OOM** —
  `Tried to allocate 20.00 MiB. GPU 0 has ... 21.38 MiB free.` Magpie's own
  load process needed ~3.2GB and there wasn't room.
- Re-measured Magpie's real standalone footprint in isolation to rule out a
  fluke: **~3.1GB** (load: 3115MiB, +inference: 3125MiB) — **not** the
  ~1.6GB `README.md`'s "Process architecture" section claims (own separate
  discrepancy worth fixing in that doc; this session's own live measurement
  is what's reported here).
- **12.5GB (LLM) + 3.1GB (TTS alone, no ASR yet) = 15.6GB — already over
  this instance's ~15.47GiB usable budget**, before `whisper_asr` or any
  Triton/system overhead is even added. Confirmed empirically, not just
  estimated: this candidate cannot share the GPU with TTS, let alone the
  full ASR+LLM+TTS cascade, at its current NVFP4-weights-only footprint.

**Verdict for candidate 1 (Qwen3-14B dense): ELIMINATED at Gate 4.**
Quality (MMLU 77.5%, IFEval 79%/85.3%) and voice-fitness (100% clean-stop)
are both genuinely strong — better than the current 4B on every quality axis
tested — but it does not clear the pipeline-viability gate on this 16GB
card. Per the brief's own instruction, this is a hard elimination, not a
"note and proceed": falling back to candidate 2.

---

### Candidate 2: Qwen3-8B dense

**Quantization**: same pipeline as candidate 1 — `Qwen/Qwen3-8B` BF16 (16GB)
→ `hf_ptq.py --qformat nvfp4 --kv_cache_qformat none --dataset
cnn_dailymail --calib_size 512` (same offload flags, though 8B nearly fits
unassisted). Output: 6.0GB checkpoint, peak calibration memory 12.6GB. Same
tokenizer sidecar bug hit and fixed pre-emptively this time (replaced
regenerated `tokenizer_config.json`/`tokenizer.json`/etc. with pristine
originals before first load attempt).

**Gate 2 — build + load + coherence: PASSED.** Served identically
(`trtllm-serve --backend pytorch --max_seq_len 4096 --max_batch_size 8
--free_gpu_memory_fraction 0.2`). Standalone footprint **8.7GB** — far more
comfortable than candidate 1's 12.5GB. Coherent, correct answers on
spot-check prompts.

**Gate 3 — quality**:
- `ifeval` (n=100): **79.0%** prompt-strict, **85.3%** inst-strict (79.0% /
  85.9% loose) — essentially identical to candidate 1's IFEval result
  despite being a smaller model.
- `mmlu_generative` (57 subjects × 20 samples, n=1140): **72.9%** — lower
  than candidate 1's 77.5% (expected: fewer parameters), but still a strong
  score, and MMLU is the brief's secondary-weighted axis vs. IFEval.
- **Voice-fitness** (same 20-prompt sample, voice system prompt,
  `enable_thinking: false`, `max_tokens=96`): **100% clean-stop (20/20),
  median 23 completion tokens** (mean 26.5) — matches candidate 1's clean
  result.
- **Standalone latency** (10 prompts, streaming): TTFT p50 **0.029s**,
  tokens/s p50 **131.6** (faster than candidate 1's 80.7 tok/s, as
  expected for a smaller model), total p50 **0.189s**.

**Gate 4 — concurrent GPU footprint + real contention test: PASSED.**

- Static fit: Qwen3-8B-NVFP4 (8.7GB) + Magpie TTS (measured standalone at
  ~3.1GB, same measurement as candidate 1's section) = **11.8GB**, leaving
  **~3.7GB** free on this instance's 15.47GiB usable budget — before
  `whisper_asr` is even added. No OOM (unlike candidate 1).
- Added `whisper_asr` (TensorRT-LLM `WhisperTRTLLM`, the actual production
  runtime, not the old ONNX path): loads in ~157MB engine + ~0.3GB KV cache
  + small overhead, total well under 1GB. All three fit simultaneously with
  real headroom to spare.
- **Real concurrent-load measurement** (not just static fit): ran a
  sustained 8-way-concurrent load generator against the LLM server
  (steady ~24-25.5 req/s throughout) while separately timing Magpie TTS and
  Whisper ASR generation calls on the same GPU:
  - Magpie TTS latency: baseline (LLM idle) p50 **1.225s** vs. under
    concurrent LLM load p50 **1.202-1.216s** across two separate runs —
    **no measurable degradation**, within run-to-run noise.
  - Whisper ASR latency under the same concurrent LLM load: p50 **0.012s**,
    mean 0.041s (one slow outlier at 0.308s, likely a cold-cache/scheduler
    artifact, not a trend) — negligible.
  - LLM's own throughput was unaffected by TTS/ASR running alongside it
    across all runs (consistently ~24-25.5 req/s at concurrency=8).
- This is a **clean pass** — a genuinely different outcome from candidate 1,
  not just "less bad." Qwen3-8B shares this GPU with the rest of the
  pipeline with real, measured headroom, not just on paper.

**Verdict for candidate 2 (Qwen3-8B dense): CLEARS ALL FOUR GATES.**

## Comparison against the current 4B baseline

Numbers pulled from this repo's own prior investigation docs (not
re-measured fresh in this pass — see caveat below):

| Metric | Current model (`docs/nemotron-token-cap-investigation.md`, post-fix) | Qwen3-14B-NVFP4 | Qwen3-8B-NVFP4 |
|---|---|---|---|
| Fits 16GB w/ ASR+TTS | yes (deployed) | **NO — Gate 4 fail** | **yes** |
| MMLU | not measured in this repo | 77.5% (n=1140) | 72.9% (n=1140) |
| IFEval | not measured in this repo | 79.0%/85.3% | 79.0%/85.3% |
| Voice-fitness: clean-stop rate | 93% (post-PR#15 stop-sequence fix; the brief's cited 97%-cap-hit / 3% clean-stop figure is the *pre-fix* number) | 100% (n=20) | 100% (n=20) |
| Voice-fitness: median response | 14 words (post-fix) | 26 tok (~19-22 words) | 23 tok (~17-19 words) |
| Standalone TTFT / tok/s | not directly comparable (different serving path) | 0.032s / 80.7 | 0.029s / 131.6 |

**Important caveat**: the brief's own framing of the current model's failure
rate (97% token-cap-hit, 46%→13% think-leakage) describes the **pre-fix**
state — `docs/nemotron-token-cap-investigation.md` and
`docs/nemotron-response-quality.md` show PR #14/#15/#16 already brought this
down to ~7% cap-hit / 14-word median on their own 69-prompt eval set. This
investigation did **not** re-run a fresh MMLU/IFEval pass against the
currently-deployed 4B checkpoint (out of scope for the time available) — the
comparison above is against the two new candidates' own measurements plus
the existing docs' own numbers for the baseline, not a freshly re-verified
one-for-one apples-to-apples run. Flagging this rather than implying a
head-to-head MMLU/IFEval number exists for the baseline when it doesn't.

## Final verdict

**Recommendation: Qwen3-8B-NVFP4 is the clear winner among the candidates
evaluated.** It clears all four gates — fits the GPU with real measured
headroom (not just on paper), shows no concurrent-load degradation for
either TTS or ASR, and its IFEval score (the brief's heavily-weighted axis)
matches the larger 14B candidate exactly while running faster (131.6 vs
80.7 tok/s) and using 30% less VRAM (8.7GB vs 12.5GB). Its MMLU score
(72.9%) is lower than the 14B's (77.5%), but the 14B is disqualified outright
by Gate 4, so that tradeoff is moot — there is no quality path to the 14B
that doesn't also require the TTS/ASR side of the pipeline to lose real
VRAM headroom this card doesn't have.

Both quantized Qwen3 candidates show **dramatically better voice-fitness
than the current 4B's documented pre-fix behavior** (100% clean-stop vs.
that baseline's original 3%), though the current model's own *post-fix*
state (93% clean-stop, 14-word median) is a much closer, fairer comparison
than the brief's own framing suggests — worth knowing before treating this
as an open-and-shut case on the voice-fitness axis alone.

**Candidates 3 (Qwen2.5-7B-Instruct) and 4 (Llama-3.1-8B) were not
evaluated** — the brief's process is to work top-down and stop once a
candidate clears all four gates convincingly, which Qwen3-8B does. Llama-3.1
would additionally have needed a gated-model HF token this instance doesn't
have. If Qwen3-8B is rejected for a reason not caught here, those two are
the documented fallbacks.

**This is investigation-only, per the brief.** `nemotron_llm` has NOT been
swapped in production. Recommend: review these findings, then a separate,
deliberate integration pass (PyTorch backend wiring into
`triton_model_repo/nemotron_llm` or a new Triton model, the `enable_thinking:
false` + voice system-prompt requirement baked into `model.py` the same way
the empty `<think></think>` forcing works today, and a fresh MMLU/IFEval
baseline run against the currently-deployed 4B for a true apples-to-apples
final comparison before cutover).

## Published checkpoints

Both quantized checkpoints from this investigation are published to
Hugging Face (private repos, personal namespace), with model cards
prepending a "Quantization" section (method, calibration, hardware target,
serving backend, and the real measured results from this document) ahead
of the original Qwen3 model card:

- [`raoashish10/Qwen3-8B-NVFP4`](https://huggingface.co/raoashish10/Qwen3-8B-NVFP4) — the winner
- [`raoashish10/Qwen3-14B-NVFP4`](https://huggingface.co/raoashish10/Qwen3-14B-NVFP4) — eliminated at Gate 4 for this project's shared-16GB-GPU deployment specifically, marked clearly as such on its own model card (not a general defect in the model)

Both were verified post-upload: file sizes confirmed byte-exact against
the local source, and (for the 8B winner) a fresh `snapshot_download` from
the Hub was loaded via `trtllm-serve --backend pytorch` and sent a real
chat-completion request, confirming the *published* artifact works
end-to-end, not just the local copy it was uploaded from.
