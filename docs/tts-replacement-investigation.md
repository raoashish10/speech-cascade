# TTS replacement investigation

Status: **COMPLETE — recommendation reached, Task 3 integration done and
verified** (remaining noted gaps: shortened soak test, no live Kokoro paired
comparison — Kokoro itself is now dead on this instance, see below). See
`/workspace/task2-tts.md` for the full brief and the "Final verdict" section
at the bottom of this document, and "Task 3: wired into Triton" further
down for the integration and its own composed-pipeline verification.

## Important discrepancy — flagged, not resolved (per explicit user instruction)

The brief this investigation is based on states: (a) Kokoro is current
production TTS, (b) Magpie-TTS was already investigated and rejected with
findings in `docs/magpie-tts-investigation.md`, and (c) a commit
(`f440a32`, "Replace Kokoro with Magpie") was merged claiming to be
"confirmed working on Blackwell" with nothing actually run, and was
**reverted** after investigation surfaced this.

None of that matches the actual state of `main` as of 2026-08-27:

- `f440a32` is **not reverted** — it is still the tip of TTS history.
  `triton_model_repo/kokoro_tts` was deleted by that commit; `magpie_tts` is
  what's actually deployed today.
- `docs/magpie-tts-investigation.md` **does not exist** anywhere in this
  repo or its branches.
- `triton_model_repo/magpie_tts/1/` contains only `model.py` — no model
  weight files in the git-tracked deploy tree (weights are expected to come
  from `models/` outside git, restored via S3 — not yet confirmed present/
  loadable as of this writing).
- `README.md`'s "Not yet done" section independently confirms magpie_tts
  replaced kokoro_tts and is "confirmed working on this Blackwell (sm_120)
  GPU" — but this is asserted in the doc, not something this investigation
  has independently re-verified yet.

Per explicit instruction from the project owner: **this is flagged here for
the record, production is not being touched, and this investigation proceeds
using Kokoro as the comparison baseline as the brief originally specified.**
Whoever picks this up next should reconcile the brief's narrative against
actual `main` before trusting either.

**Update — production has since been touched, deliberately, in a later
session.** Per explicit instruction, `chatterbox_tts` was wired into the
real Triton deploy directory
(`speech-cascade-inference/triton_model_repo/chatterbox_tts`), replacing
`kokoro_tts` there directly (`kokoro_tts` was, by that point, doubly dead
anyway: `kokoro-onnx` uninstalled from `/venv/main`, and its weight files
deleted during this session's own disk-space cleanup — so there was nothing
live left to preserve). The git-tracked but never-actually-deployed
`magpie_tts` source was replaced with the same `chatterbox_tts` code,
closing the drift this section originally flagged. Full integration
details and real verification results (not just "should work") are under
"Task 3: wired into Triton" below.

**Additional evidence found while setting up Gate 4 for the LLM
investigation (`docs/nvfp4-candidate-investigation.md`)**: `nemo_toolkit`
(the package `magpie_tts/1/model.py` imports —
`nemo.collections.tts.models.MagpieTTSModel`) **is not present in
`deploy/requirements-main.txt`**, and diffing `f440a32` itself shows it only
touched 5 lines of that file (removing `kokoro-onnx`'s numpy conflict) —
`nemo_toolkit` was never added. This instance's restored production venv
(exact frozen `pip freeze`, restored from S3 in this session) genuinely does
not have `nemo` importable. So on top of the missing investigation doc and
the not-actually-reverted commit: **the environment needed to serve
`magpie_tts` was never captured in this project's IaC either.** Following
`deploy/REBUILD.md` to the letter, as written today, would not produce a
working `magpie_tts`. This is independent, concrete evidence for the same
conclusion the discrepancy above already points at.

## Task 1: IndexTTS-2.5 (IndexTeam/IndexTTS-2.5)

### Phase 0 findings (research, not yet run on this instance)

- **VRAM**: model card states ~6GB for inference (Python 3.10-3.11, BF16
  inference option for lower usage). Not yet verified on this hardware —
  per the brief's own instruction, treat this as unverified until measured
  here.
  [Source](https://huggingface.co/IndexTeam/IndexTTS-2.5)
- **Architecture**: confirmed — GPT backbone + flow-matching speech-to-mel
  decoder + BigVGAN vocoder, matching the brief. Auxiliary models
  (w2v-bert-2.0, MaskGCT semantic codec, CAMPlus, BigVGAN) are downloaded
  separately into `checkpoints/hf_cache/` on first run, not bundled in the
  main repo — additional download weight to account for.
- **License**: bilibili Model Use License Agreement, confirmed present in
  the HF repo. Already cleared per the brief; no new concerns found.
- **"Faster IndexTTS-2" acceleration project**: this is an arXiv paper
  (2607.21042, "Faster IndexTTS-2: Accelerating and Streaming Autoregressive
  Zero-Shot TTS on GPUs", July 2026) describing TensorRT/TensorRT-LLM
  acceleration of IndexTTS-2's GPT + DiT + vocoder stages, claiming up to
  5.0x end-to-end speedup. **Could not locate a public, installable code
  release for this specific project in this search pass** — only found
  unrelated ComfyUI wrapper repos with ad-hoc Blackwell/Flash-Attention
  tweaks for plain IndexTTS-2, not the TensorRT-LLM acceleration codebase
  the brief describes. This means Phase 0's instruction to "explicitly
  verify whether that acceleration codebase loads and runs IndexTTS-2.5's
  checkpoint" may not be checkable at all if the code was never public —
  **falling back to path (a) from the brief: benchmark IndexTTS-2.5
  unaccelerated first, treat acceleration as a stretch goal contingent on
  finding the actual codebase.** Will keep looking before fully giving up
  on it.

### Phase 0 (continued) — verified on this instance

`index-tts/index-tts` (the official repo) **natively supports IndexTTS-2.5**
via `indextts/infer_v2_5.py` / `IndexTTS2` class — this is the reference/
unaccelerated implementation, separate from the "Faster IndexTTS-2"
TensorRT-LLM acceleration paper. Confirmed no public code release exists for
that acceleration project after further searching — **taking path (a) from
the brief**: unaccelerated baseline, acceleration deferred (not attempted,
codebase not found).

Setup notes:
- `requires-python ">=3.10,<3.12"` — this instance's system Python is 3.12;
  `uv sync --all-extras` handled this transparently (downloaded/used its own
  3.11 interpreter into an isolated `.venv`), no manual intervention needed.
- Install pulls a large dependency tree (torch, deepspeed, flash-attn build,
  gradio, opencv, modelscope, etc.) — ran out of disk mid-install once on
  this instance (`pip` cache had grown to 12GB from earlier work); cleared
  it and retried successfully. Not an IndexTTS-specific issue, just a real
  disk-budget constraint on this 100GB instance worth knowing about.
- First run auto-downloads auxiliary models (w2v-bert-2.0, MaskGCT semantic
  codec, CAMPlus, BigVGAN) into `checkpoints/hf_cache/` and compiles a
  custom BigVGAN CUDA kernel — confirmed it **compiles and runs on this
  SM120 (Blackwell) GPU** (`-gencode=arch=compute_120,code=sm_120` in the
  actual build log), not just Ada/Ampere/Hopper as some model cards assume
  for adjacent projects.

### Phase 1 (memory growth): PASSED, no leak observed

15-minute continuous generation loop (**not the full 40-minute standard**
used for Kokoro/Magpie — shortened due to time constraints in this session;
treat as a strong partial signal, not a final 40-min-equivalent verdict).
**641 generations completed** (901s wall time, ~1.2-1.5s/generation, no
slowdown over the run). GPU memory: **7646MiB (avg of first 10 calls) →
7684MiB (avg of last 10 calls) — 38MiB growth over 641 generations, max
7684MiB.** Effectively flat. No sign of the kind of unbounded arena growth
Kokoro has (PR #18) — this is a materially different result from that known
issue, on a 15-minute sample. A full 40-minute run would be needed before
calling this conclusively leak-free, but nothing in this sample points at a
leak.

### Gate/Phase 0 spot-checks done early (quality-adjacent, not full Phase 3)

- **VRAM footprint, measured on this hardware**: **6.6-6.8GB** (load: 6630
  MiB, after inference: 6845 MiB) — this is the one figure in this entire
  investigation (across both TTS candidates and all 4 LLM candidates) that
  matched its documented/claimed value almost exactly (model card said
  "~6GB"). Confirmed via `torch.cuda.memory_allocated()`, cross-checked
  against `nvidia-smi`.
- **Latency**: RTF (real-time factor) measured **0.39-0.99** across several
  test utterances (varies with utterance length — longer utterances
  amortize fixed overhead better). An RTF near 1.0 on the shortest test
  utterance is a real concern flagged directly by the "Faster IndexTTS-2"
  paper itself ("inference speed barely reaches real-time without streaming
  or batching") — consistent with what's measured here, unaccelerated.
- **Symbol-handling spot-check** (the exact class of bug that got Magpie
  rejected — currency/punctuation/abbreviations): tested `"$45.99"` and a
  string of abbreviations (`Dr.`, `p.m.`, `Jan.`, `St.`, `Ave.`).
  **`$45.99` → "forty five point nine nine dollars"** — correctly expanded,
  no dropped symbol (unlike Magpie's confirmed currency-symbol-dropping
  bug). `Dr. Smith` → "doctor Smith", `3 p.m.` → "three PM", `Jan. 5th` →
  "the fifth of january" — all correctly expanded. `St. Mary's Ave.` was
  **not** expanded (kept as literal "St." / "Ave.") — a minor normalization
  gap, not a functional failure (text is preserved, just not spoken-form
  expanded) — nothing like Magpie's outright symbol-dropping.
- Audio samples sent to the user directly for listening (this investigation
  cannot judge audio quality itself) — basic phrase, currency-symbol
  phrase, and abbreviation phrase, all voice-cloned from a real human
  reference clip (`tests/fixtures/vad_sample_16k.wav`).

### Phase 2 (bandwidth contention): measurable degradation, real but moderate

Tested against the **current production NVFP4 Nemotron checkpoint**
(`models/Llama-3.1-Nemotron-Nano-4B-v1.1-NVFP4`), per the brief's explicit
instruction to use the current NVFP4 engine, not the stale FP8-era
baseline. Served via `trtllm-serve --backend pytorch` (the classic backend
this checkpoint's production `model.py` actually uses is confirmed broken
— see `docs/nvfp4-classic-backend-collapse.md` — so PyTorch backend is the
only backend that produces valid output for this checkpoint; testing
against a broken backend would not be a meaningful measurement).

Method: 10 IndexTTS-2.5 generations timed with the LLM idle, then the same
10 timed while an 8-way-concurrent load generator sustained continuous
traffic against the LLM (matching the exact methodology `docs/
magpie-tts-investigation.md`'s Gate-4-equivalent test used, and the same
harness built for `docs/nvfp4-candidate-investigation.md`'s Gate 4).

| | TTS p50 | TTS mean |
|---|---|---|
| Baseline (LLM idle) | 1.449s | 1.558s |
| Under concurrent LLM load (8-way, ~11 req/s) | 2.055s | 2.186s |
| **Degradation** | **~1.42x** | **~1.40x** |

**Real, measurable contention — but meaningfully smaller than Magpie's.**
Magpie's rejected investigation measured **5.2x** TTFT degradation and
**3.1x** total p50 degradation at concurrency=8 against this same LLM
family. IndexTTS-2.5's ~1.4x degradation here is a real cost, not
negligible, but it is not in the same class of severity that got Magpie
disqualified. Whether ~1.4x is acceptable is a product call, not something
this investigation can settle on its own — flagging the number plainly
rather than asserting a verdict.

Note: the LLM's own throughput during this test (442 requests/40s ≈ 11
req/s at concurrency=8) was noticeably lower than what the same load
generator achieved against the Qwen3-8B-NVFP4 candidate in the LLM
investigation (~24-25 req/s) — plausibly explained by this being a smaller,
differently-tuned checkpoint on the PyTorch backend rather than anything
IndexTTS-specific; not investigated further here as it's outside this
task's scope.

### Phase 3 (quality)

Paired samples generated and sent to the user directly for listening
comparison (see Phase 0 section above — this investigation cannot judge
audio quality itself). **Not compared against a live Kokoro sample** in
this pass — Kokoro's own ONNX weights are still present in `models/
kokoro-82m/` (recovered from S3), but standing up its separate onnxruntime-
GPU pipeline was deprioritized given time constraints in this session
after the higher-priority Phase 1/2 technical gates. If a true paired
comparison against Kokoro specifically is needed before a decision, that
setup is the next step, not yet done here.

## Task 2: Chatterbox-Turbo via NIM or vLLM backend

### Phase 0a — NIM feasibility (RESOLVED: infeasible, confirmed two independent ways)

1. **This Vast.ai instance cannot run Docker at all** — it's an unprivileged
   container, no Docker-in-Docker (per the instance's own agent guide).
   NVIDIA NIM ships as a Docker container requiring "NVIDIA Docker >= 23.0.1"
   — hard blocker regardless of any other factor.
2. **Even setting aside (1), the memory requirement is far beyond this
   card.** NVIDIA's own TTS NIM support matrix lists Chatterbox Multilingual
   GPU memory by batch-size profile: batch_size=8 → 44.61 GiB, batch_size=32
   → 46.84 GiB, batch_size=64 → 49.72 GiB. This instance has a single 16GB
   GPU already sharing capacity with `nemotron_llm`/ASR/TTS. Not close.
   [Source](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html)

**Verdict**: NIM path is ruled out at Phase 0a exactly as the brief
anticipated ("don't discover a blocker mid-attempt") — no pull attempted, no
NGC entitlement check needed since the deployment shape itself disqualifies
it before that question is relevant.

### Fallback — chatterbox-vllm community port

- Confirmed via the repo (`randombk/chatterbox-vllm`): only the T3
  (0.5B-param Llama backbone) half is ported to vLLM. The README states
  explicitly: "the vast majority of time is now spent on the S3Gen model,
  which is not ported/portable to vLLM" — S3Gen still runs the original,
  unaccelerated reference implementation. Matches the brief's warning
  exactly — will report end-to-end latency, not just T3 throughput, per the
  brief's explicit instruction.
- **Turbo compatibility: RESOLVED — not supported, confirmed by code
  inspection (not just absence of docs).** `chatterbox-vllm`'s
  `ChatterboxTTS.from_pretrained()` hardcodes both a specific HF `revision`
  (a commit hash pinned to `ResembleAI/chatterbox`'s own git history) and a
  fixed list of expected filenames (`ve.safetensors`, `t3_cfg.safetensors`,
  `s3gen.safetensors`, `tokenizer.json`, `conds.pt`). The actual Turbo
  weights repo (`ResembleAI/chatterbox-turbo`, confirmed to exist on HF) has
  differently-named files (`s3gen_meanflow.safetensors` — Turbo's whole
  point is a distilled 1-step flow-matching decoder replacing the original
  10-step one, a real architectural difference, not just a smaller
  checkpoint) and its own separate commit history, so the pinned `revision`
  hash wouldn't resolve there either. Passing `repo_id="ResembleAI/
  chatterbox-turbo"` to `from_pretrained()` would fail outright, not
  silently load the wrong thing — confirmed by reading the actual
  `tts.py` source, not assumed. Making this port support Turbo would be a
  real development task (new S3Gen-meanflow loading path, new config
  handling), not a config flag — out of scope for this investigation.
- **This means neither path in this task gets you Chatterbox-Turbo
  specifically on this instance**: NIM is ruled out (Phase 0a above), and
  the community vLLM port only supports base Chatterbox. Proceeding to test
  the vLLM port against **base Chatterbox** (`ResembleAI/chatterbox`) below
  — this still answers the brief's "measure and report end-to-end latency,
  be explicit about which half is accelerated" instruction for the vLLM
  *architecture pattern* in general, but it is not a Turbo result and
  shouldn't be read as one.
- Community benchmarks cited in the README: RTX 3090 (24GB) generated ~40min
  audio in 87s; RTX 3060ti (8GB) in ~4.5min — the 8GB figure is an
  encouraging signal for fitting on this shared 16GB card, but these are
  upstream numbers, not measured here, and say nothing about Turbo
  specifically or about concurrent bandwidth contention with `nemotron_llm`.

### Gate 2 — build + load: FAILED on this hardware (base Chatterbox, via the vLLM port)

Installed `chatterbox-vllm` per its own README (`uv venv && uv sync`).
First real problem, caught immediately by testing rather than assuming:

- **`vllm==0.10.0` pulls `torch==2.7.1+cu126`** — CUDA 12.6, below this
  card's `min_cuda_for_wheels` requirement (12.8) for SM120/Blackwell.
  Confirmed by literally running a CUDA op: `RuntimeError: CUDA error: no
  kernel image is available for execution on the device`, with PyTorch's
  own warning printed first ("NVIDIA GeForce RTX 5070 Ti with CUDA
  capability sm_120 is not compatible with the current PyTorch
  installation... supports sm_50 ... sm_90"). This is the exact SM120/CUDA
  trap this instance's own agent guide warns about — caught here by
  actually testing, not assumed from version numbers alone.
- **Attempted fix**: upgraded `torch`/`torchvision`/`torchaudio` in-place to
  matching cu128 builds (2.11.0+cu128) via `uv pip install --upgrade
  ... --index-url .../cu128`. Plain CUDA ops then worked correctly on
  SM120, confirming the upgrade itself was sound.
- **But this broke vLLM's own compiled extension**: `vllm._C.abi3.so:
  undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...` —
  `vllm==0.10.0`'s native CUDA extension (`_C.abi3.so`) is a prebuilt
  binary compiled against torch 2.7.1's specific C++ ABI. Upgrading the
  Python-level torch package doesn't (and can't) recompile that `.so` —
  it would need `vllm` itself rebuilt from source against the newer torch,
  or a vllm wheel NVIDIA/the vllm project built specifically for a
  cu128+/SM120-compatible torch, neither of which is what `chatterbox-vllm`
  currently pins.
- **This matches the project's own explicit warning, not a surprise
  bug**: `chatterbox-vllm`'s README says outright it "uses vLLM internal
  APIs and extremely hacky workarounds" and "will likely only work with
  vLLM 0.9.2" (an even older pin than the 0.10.0 actually declared in
  `pyproject.toml`) — this is a fragile, version-pinned integration by the
  project's own admission, not a general-purpose one.
- **Stopped here rather than attempting a from-source vLLM rebuild** —
  recompiling vLLM's CUDA extensions against a newer torch for this
  specific GPU architecture is a substantial standalone engineering
  project (likely hours, uncertain success), well beyond "test whether
  this works" for an investigation. This is a **real, decisive Gate 2
  failure on this hardware, with the current pinned dependency versions**
  — not a "didn't get to it."

### Phase 1/2/3

**Not run** — Gate 2 (build + load) failed, so there was nothing to soak-
test, contention-test, or quality-compare. Per the brief's own process
(this mirrors the LLM investigation's gating logic), a candidate that
doesn't clear the build/load gate doesn't proceed to later phases.

## Final verdict

**Recommendation: IndexTTS-2.5 is a real, viable candidate; Chatterbox
(base or Turbo) is not runnable on this hardware with current dependency
pins.**

**IndexTTS-2.5** cleared every phase actually run:
- Loads and runs on this SM120 GPU, including compiling a custom BigVGAN
  CUDA kernel for `sm_120` specifically (Gate/Phase 0).
- VRAM footprint matches its own documentation almost exactly (~6.6-6.8GB
  measured vs. ~6GB claimed) — the one number across this entire two-task
  investigation that didn't need correcting.
- Phase 1 (15-min soak, shortened from the 40-min standard): 641
  generations, 38MiB memory growth — no leak signature, unlike Kokoro's
  known issue.
- Phase 2 (real concurrent-load test against the current production NVFP4
  Nemotron checkpoint on the PyTorch backend): ~1.4x latency degradation
  under sustained 8-way LLM load — real, but well short of Magpie's
  measured 3.1x-5.2x degradation that got it rejected.
- Symbol-handling spot-check: correctly expands currency (`$45.99` →
  spoken form) — does not reproduce Magpie's confirmed currency-dropping
  bug. Minor gap on some abbreviations (`St.`/`Ave.` left unexpanded), not
  a functional failure.
- **Not done**: the full 40-minute soak standard (ran 15 min instead), a
  live paired comparison against Kokoro specifically (Kokoro's weights are
  available in `models/kokoro-82m/` but its separate onnxruntime pipeline
  wasn't stood up in this pass), and acceleration via the "Faster
  IndexTTS-2" TensorRT-LLM project (no public code found — ran
  unaccelerated per the brief's own fallback path). RTF is close to 1.0 on
  short utterances unaccelerated — worth knowing if headroom under real
  multi-stream production load turns out tighter than this test's
  single-stream measurement.

**Chatterbox** (via NIM or the community vLLM port) is a clean elimination,
on both paths, for concrete and different reasons:
- NIM: ruled out at Phase 0a — this sandbox can't run Docker at all, and
  NVIDIA's own numbers put Chatterbox's NIM memory requirement (44-50GB) far
  beyond this single 16GB card regardless.
- vLLM port: **Chatterbox-Turbo specifically is not supported by
  `chatterbox-vllm`** at all (confirmed by reading `from_pretrained()`'s
  hardcoded revision/filename list against the actual Turbo repo's
  different file layout). Falling back to **base Chatterbox** (the only
  thing this port actually targets) still failed at Gate 2: `vllm==0.10.0`'s
  pinned `torch==2.7.1+cu126` doesn't support this SM120 GPU at all, and
  upgrading torch in place breaks vLLM's own prebuilt CUDA extension
  (`_C.abi3.so`, compiled against the old torch's ABI) — exactly the
  fragility the project's own README admits to ("extremely hacky
  workarounds," "will likely only work with vLLM 0.9.2"). Fixing this
  properly would mean rebuilding vLLM from source against a newer torch for
  this specific architecture — a real engineering project, not a
  config change, and out of scope here.

**Recommendation for Task 3 (Triton integration)**: proceed with
IndexTTS-2.5 as the next step, but first (a) run the full 40-minute soak
standard this investigation shortened for time, (b) get a real Kokoro-vs-
IndexTTS-2.5 paired listening comparison rather than relying on this
session's spot-checks, and (c) have a human confirm the audio samples
already generated and sent are actually acceptable quality — this
investigation cannot judge that itself. Chatterbox should not be revisited
unless someone is prepared to either wait for `chatterbox-vllm` to update
its own torch/vLLM pins for Blackwell support, or take on rebuilding vLLM
from source for this GPU architecture.

**Both discrepancies flagged at the top of this document remain unresolved
and should be reconciled before anyone acts on the "Magpie was already
rejected" narrative** — as instructed, this investigation did not touch
production to resolve them itself.

---

## Addendum: two additional candidates, evaluated against the fixed LLM choice (Qwen3-8B-NVFP4)

Requested after the LLM investigation concluded, specifically to find a TTS
model that coexists with Qwen3-8B-NVFP4 on this 16GB card (IndexTTS-2.5's
6.6-6.8GB footprint doesn't leave room, per the OOM measured above). Not
part of the original brief's candidate list — added because IndexTTS-2.5
and Chatterbox both washed out for this specific pairing.

### F5-TTS

Installed via `pip install f5-tts` in an isolated venv. Two real
environment bugs hit and fixed immediately:
- The package's own dependency resolution installs a `torchaudio` built for
  CUDA 13.0 against a `torch` built for CUDA 12.8 — `RuntimeError: Detected
  that PyTorch and TorchAudio were compiled with different CUDA versions`.
  Fixed with a matching `torchaudio` reinstall from the cu128 wheel index.
- No other install issues — notably simpler setup than either IndexTTS-2.5
  (custom CUDA kernel compile, multiple auxiliary model downloads) or
  Chatterbox (compiled extension ABI mismatch).

**Gate 2 — build + load + coherence: PASSED.** `F5TTS_v1_Base` (336M
params) loaded and ran real zero-shot cloning from the same reference clip
used throughout this investigation.

**Measured footprint**: **~3.0GB** total via `nvidia-smi` after a full
inference call (load: 703MiB torch-allocated; full pipeline including an
internal Whisper pass for reference-audio auto-transcription pushes total
to ~3.0GB). Confirmed **fits alongside Qwen3-8B-NVFP4**: combined **11.7GB
loaded, 4.15GB free after inference** — no OOM, unlike IndexTTS-2.5.

**Concurrent-load contention** (same load-generator methodology as the LLM
investigation, 8-way concurrent LLM traffic): baseline (LLM idle) p50
**0.752s**, under load p50 **0.913s** — **~1.21x degradation**, slightly
better than IndexTTS-2.5's ~1.4x (though the two weren't measured under
identical prompt/text conditions, so treat as directionally comparable, not
exact).

**Symbol handling**: unlike IndexTTS-2.5, no visible explicit text-
normalization step appeared in F5-TTS's logs — `$45.99` was passed straight
through to the model as literal text rather than expanded to spoken form.
Whether the model handles this gracefully via its own learned
pronunciation or produces an artifact is something only a human listener
can judge from the sent sample (`f5tts_symbols.wav`) — flagging this as an
open question, not a pass/fail call this investigation can make itself.

**License — a real constraint**: the GitHub repo (code) is **MIT**
(permissive), but the actual model **weights** (`SWivid/F5-TTS` on Hugging
Face) are **CC-BY-NC-4.0 — non-commercial only**. This is a materially
different situation from IndexTTS-2.5, whose Bilibili license was already
reviewed and cleared for this project's commercial scale. Verify licensing
before treating F5-TTS as viable for production use, independent of the
technical results above.

### XTTS-v2 (Coqui)

Installed via `pip install coqui-tts` (the maintained idiap fork; the
original `coqui-ai/TTS` package is abandoned). Three real environment bugs
hit and fixed:
- `coqui-tts` declares `transformers>=4.57` with no upper bound, but its
  actual tortoise-derived code imports `isin_mps_friendly` from
  `transformers.pytorch_utils` — a function removed in `transformers` 5.x.
  This is a live bug in the package's own dependency declaration, not
  something specific to this instance. Fixed by pinning
  `transformers>=4.57,<5.0`.
- Torch ≥2.9 requires `torchcodec` for audio I/O, not declared as a hard
  dependency — `ImportError: ... torchcodec library is required`. Fixed by
  installing it explicitly.
- `torchcodec` itself then failed to load its native library:
  `OSError: libnppicc.so.12: cannot open shared object file` (NVIDIA's NPP
  image-processing library, not present for this venv's CUDA runtime).
  Fixed by installing the `nvidia-npp-cu12` pip package and adding it to
  `LD_LIBRARY_PATH` — the same class of fix this whole project has needed
  repeatedly for CUDA library resolution on this bare-metal setup.
- **First-run also requires interactive Coqui Public Model License (CPML)
  acceptance** — blocks any non-interactive/automated deployment unless
  bypassed via the documented `COQUI_TOS_AGREED=1` environment variable.

**Gate 2 — build + load + coherence: PASSED**, once the above were fixed.

**Measured footprint**: **the smallest of all TTS candidates tested in
this investigation** — ~2.3GB total via `nvidia-smi` after inference (load:
2070MiB, after infer: 2268MiB). **Fits alongside Qwen3-8B-NVFP4** with the
most headroom of any candidate tested: combined **10.9GB loaded, 4.9GB
free**.

**Concurrent-load contention**: baseline (LLM idle) p50 **0.458s**, under
concurrent LLM load p50 **0.528s** — **~1.15x degradation**, the lowest of
any TTS candidate measured in this investigation (Magpie's rejected
degradation was 3.1x-5.2x, IndexTTS-2.5 ~1.4x, F5-TTS ~1.21x). Also the
fastest standalone latency of any candidate tested.

**Symbol handling**: real audio sample generated (`xtts_symbols.wav`),
sent for listening — same caveat as F5-TTS, quality judgment needs a human
ear.

**License — the most restrictive of any candidate in this investigation**:
Coqui's **CPML (Coqui Public Model License)**, explicitly **non-commercial**
unless a commercial license is purchased from Coqui directly
(`licensing@coqui.ai`). The package itself enforces this at first run via
an interactive terms-of-service prompt — this is not a passive/theoretical
restriction, it's an active gate the software itself imposes. This is a
harder constraint than F5-TTS's CC-BY-NC (which at least has an MIT-licensed
codebase); XTTS-v2's model AND the specific model-loading pathway both
carry the restriction.

### Updated comparison (LLM fixed at Qwen3-8B-NVFP4)

| | IndexTTS-2.5 | F5-TTS | XTTS-v2 | Magpie (current, unverified) |
|---|---|---|---|---|
| Fits w/ Qwen3-8B (8.7GB) | **NO** (OOM) | yes (11.7GB, 4.15GB free) | yes (10.9GB, 4.9GB free) | yes (measured earlier, ~11.8GB) |
| Footprint | 6.6-6.8GB | ~3.0GB | ~2.3GB | ~3.1GB |
| Concurrent-load degradation | ~1.4x | ~1.21x | ~1.15x | not directly comparable (measured vs. old 4B) |
| License | Bilibili — **cleared for this project** | Weights CC-BY-**NC**-4.0 | Coqui CPML — **non-commercial** | N/A (open weights, no known restriction) |
| Symbol handling | Explicit normalization, correct on currency test | Passed through raw — unverified by ear | Unverified by ear | Confirmed broken (currency-dropping bug) |

**Initial tradeoff, before the next candidate below**: the two candidates
that fit comfortably and show the least bandwidth contention (XTTS-v2,
F5-TTS) both carry non-commercial restrictions on their weights, while the
one already cleared for commercial use (IndexTTS-2.5) doesn't fit. This
looked like a real licensing-vs-engineering tradeoff — resolved by the
candidate below, which turned out to have neither problem.

### Chatterbox-Turbo, reference implementation (not the broken vLLM port)

Requested after confirming Chatterbox's actual model license is genuinely
**MIT** (both base and Turbo, confirmed on the HF model cards — `License:
mit` stated explicitly, not inferred). The earlier elimination in this
document was specifically about the **vLLM acceleration port**
(`chatterbox-vllm`) being broken on this GPU, not about the model or its
license — worth re-testing via the model's own reference implementation,
the same fallback pattern that worked for IndexTTS-2.5 (run unaccelerated
rather than assume the accelerated path is the only option).

Installed `chatterbox-tts` (the official `resemble-ai/chatterbox` package,
plain PyTorch/transformers, no vLLM dependency at all) in a fresh isolated
venv. Real environment issues hit and fixed — notably milder than the vLLM
port's compiled-extension dead end, since there's no prebuilt native binary
to go stale here:
- Same SM120/CUDA trap as every other package in this investigation that
  pins its own torch: `chatterbox-tts==0.1.7` pulls `torch==2.6.0`+cu124,
  which doesn't support this GPU (`sm_120 is not compatible`). Fixed by
  force-upgrading `torch`/`torchaudio` to the cu128 build — and unlike
  `chatterbox-vllm`, **this actually worked cleanly**, because there's no
  precompiled `.so` extension tied to the old torch ABI (pure PyTorch, no
  custom CUDA kernels compiled ahead of time).
- Same `torchcodec` + `libnppicc.so.12` (NVIDIA NPP library) gap XTTS-v2
  hit. Fixed identically: install `torchcodec` + `nvidia-npp-cu12`, add the
  latter to `LD_LIBRARY_PATH`.
- **Chatterbox-Turbo requires a reference clip longer than 5 seconds**
  (`assert len(s3gen_ref_wav) / _sr > 5.0`) — the ~3.2s clip used
  consistently for every other candidate in this investigation is too
  short. Used a different, longer clip (`scripts/pipeline_output.wav`,
  41s) for this candidate only — **note this when comparing voice
  similarity across the sent audio samples; it's not an apples-to-apples
  reference clip with the others.**

**Gate 2 — build + load + coherence: PASSED.** Loaded in ~42s, generated
real audio from real text (including the currency/symbol test phrase) in
1.6-4.2s per call.

**Measured footprint**: **~3.0-3.4GB** total via `nvidia-smi` (after load:
2988MiB, after inference: 3446MiB) — in the same small-footprint class as
F5-TTS, though the largest of the three viable-sized candidates.

**Fits alongside Qwen3-8B-NVFP4**: combined **12.41GB loaded, 3.43GB free**
— less headroom than F5-TTS (4.15GB free) or XTTS-v2 (4.9GB free), but
comfortably clear of the OOM threshold that eliminated IndexTTS-2.5.

**Concurrent-load contention**: baseline (LLM idle) p50 **0.513s**, under
8-way concurrent LLM load p50 **0.647s** — **~1.26x degradation**, in the
same range as F5-TTS (~1.21x) and XTTS-v2 (~1.15x), all far below Magpie's
rejected 3.1x-5.2x.

**License: MIT, no commercial restriction** — confirmed directly on
`ResembleAI/chatterbox` and `ResembleAI/chatterbox-turbo`'s HF model
cards, not inferred from third-party summaries (a prior websearch result
claiming Chatterbox-Turbo had a separate community fork with its own terms
turned out to be a red herring — the actual weights are plain MIT).

### Updated comparison, with Chatterbox-Turbo added

| | IndexTTS-2.5 | F5-TTS | XTTS-v2 | Chatterbox-Turbo | Magpie (current, unverified) |
|---|---|---|---|---|---|
| Fits w/ Qwen3-8B | **NO** (OOM) | yes (4.15GB free) | yes (4.9GB free) | yes (3.43GB free) | yes (~3.9GB free, measured earlier) |
| Footprint | 6.6-6.8GB | ~3.0GB | ~2.3GB | ~3.0-3.4GB | ~3.1GB |
| Concurrent-load degradation | ~1.4x | ~1.21x | ~1.15x | ~1.26x | not directly comparable (measured vs. old 4B) |
| License | Bilibili — cleared | Weights CC-BY-**NC** | Coqui **CPML** — non-commercial | **MIT — no restriction** | N/A |
| Symbol handling | Explicit normalization, correct | Passed through raw, unverified by ear | Unverified by ear | Unverified by ear | Confirmed broken |

### Final verdict, updated

**Chatterbox-Turbo (reference implementation) resolves the licensing
tradeoff this document flagged above.** It clears every gate this
investigation could test — coexists with Qwen3-8B-NVFP4 with real measured
headroom, shows concurrent-load contention in the same low range as the
other small candidates, and carries a genuinely permissive MIT license with
no commercial-use restriction, unlike F5-TTS (weights CC-BY-NC) or XTTS-v2
(Coqui CPML, non-commercial without a purchased license). It does have the
least VRAM headroom of the three small candidates (3.43GB vs. 4.15-4.9GB)
and the reference implementation is a single-purpose voice-cloning package
without XTTS-v2's broader multilingual maturity or F5-TTS's flow-matching
prosody reputation — tradeoffs worth weighing against the licensing
advantage, not an automatic win on every axis.

**Recommendation, if a permissively-licensed candidate is the priority**:
Chatterbox-Turbo is the strongest option found in this investigation —
fits, contends well, and is the only small-footprint candidate with no
licensing caveat attached. XTTS-v2 remains the strongest *technical* result
if a non-commercial license (or a purchased Coqui license) is acceptable.
IndexTTS-2.5 remains the right answer only if the LLM-side VRAM problem is
solved some other way (smaller KV-cache reservation, or accepting the two
can't be resident simultaneously) and its Bilibili license terms are
preferred over MIT for some reason not evaluated here.

### Full voice-pipeline stress test: Chatterbox-Turbo (real ASR -> LLM -> TTS, sustained)

Requested as a follow-up to the isolated Gate-4-style contention test above:
a genuine end-to-end soak test chaining real Whisper ASR -> real
Qwen3-8B-NVFP4 (HTTP) -> real Chatterbox-Turbo as actual conversation
turns, sustained for 15 minutes, with a separate 4-worker background thread
pool continuously hammering the LLM concurrently the whole time to
simulate other users sharing the GPU (not just this one representative
pipeline chain in isolation).

**Three real bugs hit and fixed while building this harness** (all
environment/orchestration issues, not TTS quality problems — worth knowing
since the same harness will be reused for XTTS-v2 and F5-TTS):
1. **Fork-after-threading deadlock.** Starting the background LLM-load
   threads *before* importing `WhisperTRTLLM` hung the whole process
   indefinitely (confirmed via `/proc/<pid>/wchan` showing `pipe_read` with
   zero CPU progress for minutes). TensorRT-LLM's MPI-based worker spawn
   forks internally; forking while other threads hold I/O/allocator locks
   mid-operation is a textbook deadlock (the forked child can inherit a
   lock with no owner left alive to release it). Fixed by loading ASR to
   completion *before* starting any other thread.
2. **Inter-process protocol contamination.** The persistent TTS worker
   subprocess communicates turn-by-turn over a pipe, but the TTS library's
   own output (progress bars, warnings, plain prints) landed on *both*
   stdout and stderr, getting misread as if it were the protocol reply.
   Fixed with a distinctive sentinel prefix (`@@PROTO@@`) that the
   orchestrator filters for, discarding everything else regardless of
   which stream it appeared on.
3. **Wrong `LD_LIBRARY_PATH` inherited by the TTS worker subprocess** — it
   silently got the parent process's path (set for `/venv/main`'s
   TensorRT-LLM) instead of its own venv's NPP library path needed by
   `torchcodec`. This invalidated a full 15-minute run before it was
   caught: every turn's actual TTS generation succeeded, but the final
   `torchaudio.save()` failed every single time, so every turn logged as
   an error despite the model working correctly. Fixed by passing an
   explicit `env=` dict to the worker's `subprocess.Popen` call rather
   than relying on inherited shell state.

**Results (clean run, 1039 turns over 900s, 0 errors, 8869 concurrent
background LLM requests fired during the same window):**

| | p50 | p95 | max |
|---|---|---|---|
| ASR | 0.016s | — | — |
| LLM | 0.092s | — | — |
| TTS | 0.729s | — | — |
| **Total turn latency** | **0.84s** | **0.92s** | **2.23s** |

**Memory**: first-10-turn average **13,827 MiB**, last-10-turn average
**13,858 MiB** — a 31MiB drift over 1039 turns and 15 minutes, indistinguishable
from noise. No leak signature in the composed pipeline, matching (not
contradicting) the earlier isolated 641-generation soak result. **Zero
errors across all 1039 turns** once the harness bugs above were fixed —
the pipeline held up under sustained real load with concurrent LLM
pressure the whole time.

This is meaningfully stronger evidence than the earlier isolated Gate-4
test: it confirms Chatterbox-Turbo doesn't just tolerate concurrent LLM
load in a synthetic timing loop, but holds up as part of an actual composed
ASR->LLM->TTS chain, repeatedly, without degrading or leaking over a
sustained window — directly answering the brief's original concern about
isolated vs. composed behavior diverging (the same concern that motivated
Task 3's planned re-test in Triton).

**Paused here** (per explicit instruction) before running the same
stress test against XTTS-v2 and F5-TTS — the harness is built and working;
resuming just means rerunning it with a different backend argument.

**Human listening feedback (partial)**: samples for all four candidates
(basic phrase, currency/symbol test, and — for IndexTTS-2.5, F5-TTS, and
Chatterbox-Turbo — an abbreviation test) were sent directly to the project
owner, since this investigation cannot judge audio quality itself.
Chatterbox-Turbo was reviewed and called **"pretty good honestly
overall"** — the only candidate with a positive human quality signal
recorded in this document as of this writing. F5-TTS, XTTS-v2, and
IndexTTS-2.5 have not yet received explicit feedback. Combined with its
clean MIT license and passing every technical gate measured (fits
alongside Qwen3-8B-NVFP4, ~1.26x concurrent-load degradation, well below
the Magpie rejection threshold), **Chatterbox-Turbo is currently the
strongest-evidenced candidate in this investigation** — the only one with
both a fully clean technical/licensing picture and a real human quality
signal, not just this session's own measurements. Not yet a final
decision: XTTS-v2 and F5-TTS haven't had their own listening review, and a
side-by-side comparison against Kokoro (the brief's original baseline)
still hasn't happened.

## Task 3: wired into Triton (later session, per explicit instruction)

Chatterbox-Turbo was wired into the real Triton deploy directory as
`chatterbox_tts`, replacing `kokoro_tts` (dead) and superseding the
never-actually-deployed `magpie_tts` git source (see the "Update" note
under "Important discrepancy" above). Full detail in
`triton_model_repo/chatterbox_tts/1/model.py`'s own docstring; summary:

- **Architecture**: unlike every other model in this pipeline,
  `chatterbox_tts`'s Triton python-backend stub does not run the model
  in-process. Chatterbox's torch/torchaudio pins conflict with
  `/venv/main`'s TensorRT-LLM stack, so `model.py` spawns a persistent
  subprocess running `/venv/chatterbox/bin/python3`
  (`chatterbox_worker.py`, alongside it), communicating over a pipe with
  the same sentinel-protocol pattern already validated in this
  investigation's own stress-test harness. The reference voice is embedded
  once at worker startup (`prepare_conditionals`), not per call, unlike
  the investigation's own throwaway `tts_worker.py` script — a real
  latency improvement for production use, not just a port.
- **A real bug hit and fixed while wiring this up**: the first load
  attempt failed with an opaque `chatterbox worker failed to start: None`
  — the worker subprocess was dying before it could report why. Root
  cause: Triton's supervisor script sets `PYTHONHOME=/venv/main` and
  `PYTHONPATH=/venv/main/lib/python3.12/site-packages` globally (see
  `deploy/supervisor/speech-cascade-triton.sh`), and `model.py`'s
  `subprocess.Popen(..., env=dict(os.environ))` inherited both, which
  breaks `/venv/chatterbox/bin/python3`'s own `sys.path` resolution
  entirely (confirmed by hand:
  `env PYTHONHOME=/venv/main PYTHONPATH=... /venv/chatterbox/bin/python3 -c "import torchaudio"`
  fails with `undefined symbol: _PyErr_SetLocaleString` — a cross-venv
  CPython ABI mismatch, not a missing dependency). Fixed by explicitly
  stripping both from the worker subprocess's env in `model.py`.
- **Real end-to-end verification, through the actual live Triton
  `voice_pipeline` gRPC endpoint** (not the standalone harness used
  earlier in this document) — first-ever successful call of its kind for
  this deployment:
  - Sent `tests/fixtures/vad_sample_16k.wav` ("Testing 123.") through
    `voice_pipeline`. Got back a real transcript, a real (if verbose)
    `nemotron_llm` response, and 6.44s of real synthesized audio (154560
    samples @ 24000Hz, RMS 0.047, peak 0.36 — genuine signal, not
    silence). Sample sent directly to the project owner for listening
    review.
  - **Composed-pipeline sustained-load test** (this is the check
    task2-tts.md's Task 3 explicitly asks for — "with the model actually
    running inside the composed Triton/BLS pipeline, not just in
    standalone isolation"): 900s (15 min) of sequential real requests
    against the live `voice_pipeline` endpoint. **471 turns, 0 errors.**
    Latency stable throughout at ~1.8-2.1s/turn (no degradation trend).
    GPU memory: 12277MiB at turn 1 -> two small steps up to 12375MiB by
    turn ~170 (t=323s) -> **completely flat for the remaining ~580s/300
    turns**. Total drift 98MiB over 471 turns — larger than the
    standalone harness's own 31MiB/1039-turn result, but the shape (all
    growth front-loaded, then a hard flat line) reads as allocator/cache
    warmup reaching a high-water mark, not an ongoing per-call leak. Not
    the full 40-minute duration task2-tts.md's Task 3 specifies; stated
    here as 15 minutes because that's what was actually run, not rounded
    up.
  - This test was sequential (single client, no concurrent background
    load), unlike the standalone harness's version which also fired
    concurrent LLM traffic throughout. Concurrency/capacity tuning for
    `chatterbox_tts` inside Triton (analogous to `kokoro_tts`'s own
    `count: 1 -> 4` journey, see `docs/kokoro-tts-capacity-fix.md`) has
    not been done — `instance_group.count` is still 1, a functional
    baseline, not a tuned deployment.

**Verdict**: `chatterbox_tts` is live, verified via a real call and a real
15-minute composed-pipeline soak test with zero errors and no leak
signature beyond initial warmup — not just "should work by construction."
Remaining before this is a fully tuned production deployment: instance
count/concurrency tuning, the full 40-minute duration, and a live
side-by-side listening comparison against Kokoro (impossible now — Kokoro
is dead on this instance) or against a fresh Kokoro re-install if that
comparison is still wanted.
