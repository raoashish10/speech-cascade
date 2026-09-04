# Whisper from-scratch build verification, and the qwen_llm cold-start flag

Two independent items from the same session, on the same fresh (no
persistent volume, no S3 access) instance: verifying Whisper's
TensorRT-LLM engine build end-to-end for the first time, and chasing down
`qwen-llm-migration.md`'s one flagged-but-never-rechecked load-test
failure.

## Part 1: Whisper TensorRT-LLM engine build — now verified end-to-end

### Verdict

**REBUILD.md §4c's documented commands work exactly as written.** Every
step — `convert_checkpoint.py`, both `trtllm-build` calls — ran with zero
deviations from the doc on a fresh instance with no access to the
project's S3 backup, and the resulting engines pass all 15
`tests/integration` tests, including the three Whisper-specific tests and
the full `voice_pipeline` end-to-end round trip. REBUILD.md's own "Known
gaps" section is updated to close this out.

### What was actually run

```bash
git clone --depth 1 --branch v1.2.1 https://github.com/NVIDIA/TensorRT-LLM.git /workspace/tensorrt_llm_repo
cd /workspace/tensorrt_llm_repo/examples/models/core/whisper
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt

python3 convert_checkpoint.py --output_dir /workspace/whisper_base_weights --model_name base
trtllm-build --checkpoint_dir /workspace/whisper_base_weights/encoder --output_dir .../encoder \
  --moe_plugin disable --max_batch_size 8 --gemm_plugin disable --bert_attention_plugin float16 \
  --max_input_len 3000 --max_seq_len=3000
trtllm-build --checkpoint_dir /workspace/whisper_base_weights/decoder --output_dir .../decoder \
  --moe_plugin disable --max_beam_width 4 --max_batch_size 8 --max_seq_len 114 --max_input_len 14 \
  --max_encoder_input_len 3000 --gemm_plugin float16 --bert_attention_plugin float16 \
  --gpt_attention_plugin float16
```

Copied character-for-character from REBUILD.md §4c — no flags changed, no
workarounds needed.

### Results

| Step | Result |
|---|---|
| `convert_checkpoint.py` | Succeeded, <1s |
| Encoder `trtllm-build` | Succeeded, ~19s, `rank0.engine` 45.7MB |
| Decoder `trtllm-build` | Succeeded, ~2s, `rank0.engine` 165.6MB |
| `tests/integration` (15 tests) | **15/15 passed**, including `test_whisper_asr.py` (3/3) and `test_voice_pipeline.py`'s full audio-in/audio-out round trip |

No deviation from the documented flags, no missing dependency (the
example's own `requirements.txt` — `tiktoken`, `datasets`, `kaldialign`,
`openai-whisper`, `librosa`, `soundfile`, `safetensors`, `transformers`,
`janus` — was already fully covered by `deploy/requirements-main.txt`),
no path mismatch (`triton_model_repo/whisper_asr/config.pbtxt`'s
`engine_dir`/`assets_dir` line up with the build output exactly as
documented).

**This closes REBUILD.md's own "unverified, reconstructed from the
upstream example" caveat for §4c.** The values it already had
(`max_seq_len 114`, etc.) were correct; they just hadn't been run.

## Part 2: the flagged qwen_llm cold-start failure

### Verdict

**Not reproducible on the current config — and there's now a real, fixed
root cause on record for exactly the kind of instability that would
produce it.** `qwen-llm-migration.md` flagged one `qwen_llm` request
timing out at 60.096s during its first-ever load test (1/20 at
concurrency 4), hypothesized as a one-time cold-start cost (CUDA graph
capture, kernel autotuning) and never rechecked. Three separate
fresh-reload trials here — 8 sequential requests, 20 requests at
concurrency 4, 40 requests at concurrency 8, each immediately after an
`unload`+`load` with zero warmup requests in between — produced **zero
failures and no outlier** (max latency 0.212s across all three runs, TTFT
p99 0.149s).

### Why this is a meaningful negative result, not just "didn't reproduce"

This same session independently found and fixed a real bug in this exact
code path (see `deploy/REBUILD.md`'s "Known gaps" and this branch's own
Triton-startup fix): `PYTHONHOME=/venv/main`, set globally by
`speech-cascade-triton.sh`, breaks `import ctypes` in any process
TensorRT-LLM's MPI-based worker-spawn (`MpiPoolSession`) forks after that
point — which is exactly the machinery `qwen_llm`'s `_TrtLLM` construction
depends on. Before that fix, `qwen_llm` didn't intermittently time out —
it **failed to load at all**, deterministically, every time. That's a
more severe failure mode than a single slow request, so this isn't a
direct "found the same bug" claim. But it establishes that this exact
MPI/`ctypes` subsystem is fragile to environment details in ways nobody
had root-caused before this session, which is a plausible (not proven)
explanation for a rare, unexplained timeout during the original
qwen-llm-migration load test — that session was mid-migration, actively
changing the deployment's Python environment, and never got a clean
explanation for the outlier either.

### Method

1. `curl -X POST .../v2/repository/models/qwen_llm/unload` then `.../load`,
   waited for `READY` — a genuinely fresh model instance, zero prior
   inference calls.
2. Immediately (no warmup): `scripts/load_test.py --model qwen_llm` at
   three concurrency/volume combinations, timing every individual request
   rather than only reporting aggregates, so a single-request outlier
   would be directly visible, not averaged away.

| Trial | Concurrency | n | Failures | Max client latency | TTFT p99 |
|---|---|---|---|---|---|
| 1 (sequential) | 1 | 8 | 0 | 0.083s | — |
| 2 | 4 | 20 | 0 | 0.190s | 0.055s |
| 3 | 8 | 40 | 0 | 0.212s | 0.149s |

No request anywhere near 60s in any trial, on a model that had never
served a single request before each trial started.

### What this doesn't close out

- This is not the *same* environment as the original `qwen-llm-migration`
  session — different instance, rebuilt from scratch. It's not possible
  to literally replay their exact conditions to confirm the cause with
  certainty, only to confirm the failure doesn't happen on the current,
  now-more-thoroughly-debugged config.
- A genuinely one-time-in-hundreds-of-requests outlier could still exist
  and simply not have shown up in 68 total requests across three trials.
  Task 2's sustained concurrent stress campaign (much higher request
  volume, run separately) is the better instrument for catching something
  that rare if it's still there — cross-reference that doc for whether it
  turned up anything similar.
