# Qwen3-8B-NVFP4 across TensorRT-LLM, vLLM, and SGLang

Follow-up to [`qwen-llm-migration.md`](qwen-llm-migration.md): that document
covers getting `qwen_llm` running inside this project's Triton deployment.
This one answers a narrower, standalone question — raised while adding
vLLM/SGLang usage instructions to the checkpoint's Hugging Face model cards
([`Qwen3-8B-NVFP4`](https://huggingface.co/raoashish10/Qwen3-8B-NVFP4)) —
**do those documented commands actually work**, and how do the three
backends compare, serving the exact same checkpoint under the exact same
test?

## Verdict

All three load the checkpoint and produce coherent output. **Neither vLLM
nor SGLang works out of the box on this box's GPU (RTX 5070 Ti, consumer
Blackwell / SM120)** — both hit the same underlying cause (a `flashinfer`
JIT-kernel gap for SM120) via two different code paths, and each needed a
different workaround. TensorRT-LLM needed no workaround; it's what this
project already runs in production. Once each is actually working,
performance is close across all three at this concurrency.

| | TensorRT-LLM (`trtllm-serve`) | vLLM | SGLang |
|---|---|---|---|
| Version | `1.2.1` | `0.28.0` | `0.5.18` |
| Successes / n | 20/20 | 20/20 | 20/20 |
| Client latency p50 | 0.923s | 0.741s | 0.866s |
| Client latency p90 | 1.107s | 0.870s | 1.301s |
| Client latency p99 | 1.134s | 1.002s | 1.553s |
| TTFT p50 | 0.028s | 0.023s | 0.022s |
| Throughput | 4.547 req/s | 5.487 req/s | 4.693 req/s |
| Worked with the checkpoint as-published? | Yes, no changes | **No** — needed a checkpoint metadata patch + an env var | **No** — needed explicit backend flags (no checkpoint changes) |

## Method

Same test against all three, back-to-back, on the same idle GPU (checked
0 MiB used before each run) — this is a controlled, apples-to-apples
comparison, unlike `qwen-llm-migration.md`'s Triton numbers (measured
inside the full 4-model voice-pipeline, sharing the GPU with
`whisper_asr`/`chatterbox_tts`/`voice_pipeline`, over Triton's gRPC
protocol — not comparable to the isolated numbers here):

- All three served `raoashish10/Qwen3-8B-NVFP4` (this project's local copy)
  standalone, nothing else loaded on the GPU.
- All three capped at `max_seq_len`/`--max-model-len`/`--context-length` =
  4096, matching the model card's example TensorRT-LLM config.
- Same client script
  (`openai_load_test.py`, an ad hoc script written for this comparison —
  concurrency=4, n=20, 5 short rotating prompts, `max_tokens=128`,
  `temperature=0`, `enable_thinking=false`, streaming for TTFT) against
  each backend's OpenAI-compatible `/v1/chat/completions` endpoint.
- Correctness spot-checked before each load test with a single non-streamed
  request ("What is the capital of France?") — all three answered
  correctly ("The capital of France is Paris.").

This is a narrow, single-box, single-request-shape smoke test — concurrency
4, short prompts, `max_tokens=128` — not a rigorous benchmark. Treat the
relative closeness of the three latency/throughput numbers as "all three
work reasonably at this scale," not as a precise ranking.

## Why vLLM and SGLang aren't drop-in on this GPU

### Root cause: `flashinfer`'s SM120 (consumer Blackwell) support gaps

Both vLLM and SGLang use NVIDIA's `flashinfer` library for various
JIT-compiled CUDA kernels. Two separate gaps showed up:

1. **The FP4 GEMM kernel.** `flashinfer`'s CUTLASS FP4 GEMM path
   (`gen_gemm_sm120_module_cutlass_fp4`) fails at JIT-compile time with
   `RuntimeError: No supported CUDA architectures found for major versions
   [12]` — it doesn't recognize SM120 (compute capability 12.x) as a
   target it can build for.
2. **A separate, unrelated `check_cuda_arch()` bug.** `flashinfer`'s
   generic JIT-compile gate (used by, e.g., the sampling kernel and the
   attention/prefill kernel) raises `RuntimeError: FlashInfer requires
   GPUs with sm75 or higher` — on an SM120 GPU. This reads like a version-
   parsing bug (SM120's two-digit major version tripping up logic written
   for single-digit majors), not a real hardware-support gap, since SM120
   is far newer/higher than sm75.

These are `flashinfer`/vLLM/SGLang-side gaps as of the versions tested
above (2026-09-02) — not something wrong with the checkpoint, and quite
possibly already fixed in newer releases; worth re-checking before
assuming this analysis is still current.

### An additional, checkpoint-specific wrinkle for vLLM

vLLM's ModelOpt quantization loader (`vllm/model_executor/layers/
quantization/modelopt.py`) dispatches on the checkpoint's declared
`quant_algo`: a plain `"NVFP4"` routes to the full W4A4 (cutlass) path —
the one that hits gap #1 above — while a separate `"W4A16_NVFP4"` tag
routes to a weight-only path via Marlin, which doesn't need the SM120
cutlass kernel at all. **This checkpoint's `config.json` declares
`"NVFP4"`** (both in the top-level `quantization_config` block and in
`hf_quant_config.json`), so vLLM picked the full W4A4 path by default.

Testing this directly: patching a local copy's `config.json` +
`hf_quant_config.json` (`quant_algo: "NVFP4"` → `"W4A16_NVFP4"`, nothing
else changed) let vLLM load successfully via Marlin and produce correct
output — consistent with this checkpoint's own model card description
("weight-only NVFP4 / W4A16 ... activations left at native precision").

**Worth flagging as a real, unresolved inconsistency**: that same
`config.json`'s `quantization_config.config_groups.group_0
.input_activations` block explicitly declares `{"num_bits": 4, "dynamic":
false, ...}` — i.e., its own metadata also describes activations as
4-bit-quantized, which contradicts both the model card's "weight-only"
claim and the working-Marlin-path result above. TensorRT-LLM's loader
reads the same `hf_quant_config.json` and serves this checkpoint correctly
today in production (see `qwen-llm-migration.md` and the main `README.md`'s
benchmark numbers), which is strong evidence the checkpoint's actual
*weights* are genuinely weight-only-compatible — but the metadata itself
is self-contradictory across its two config surfaces
(`hf_quant_config.json` vs. `config.json`'s `config_groups`), likely an
artifact of how `modelopt.torch.export.export_hf_checkpoint()` populates
the `compressed-tensors`-style `config_groups` block for a `--qformat
nvfp4` export regardless of whether activation quantization was actually
calibrated/applied. **Not root-caused further here** — flagging for
whoever touches this checkpoint's export process next.

### Working commands (what was actually run)

**vLLM** (needs the checkpoint patch above, plus disabling flashinfer's
sampler):

```bash
# one-time: quant_algo "NVFP4" -> "W4A16_NVFP4" in both config.json and
# hf_quant_config.json (on a local copy, not the original)
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve <patched-checkpoint-dir> \
  --quantization modelopt_fp4 --max-model-len 4096
```

**SGLang** (no checkpoint changes — explicit backend flags instead):

```bash
python3 -m sglang.launch_server \
  --model-path raoashish10/Qwen3-8B-NVFP4 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend marlin \
  --attention-backend triton \
  --sampling-backend pytorch \
  --context-length 4096
```

**TensorRT-LLM** (no changes — matches the model card exactly):

```bash
trtllm-serve serve raoashish10/Qwen3-8B-NVFP4 \
  --backend pytorch --max_seq_len 4096 --max_batch_size 8 \
  --free_gpu_memory_fraction 0.7
```

The Hugging Face model cards intentionally do **not** carry the vLLM/SGLang
workarounds above (checkpoint patch, env var, backend flags) — they
document the plain upstream-documented commands and flag them as untested
by this project, on request. This document is the actual test record.
