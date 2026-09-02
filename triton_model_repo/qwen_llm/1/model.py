"""Triton Python backend for Qwen3-8B-NVFP4, served by loading the quantized
HF checkpoint directly via TensorRT-LLM's LLM API (JIT graph build, classic
TensorRT backend, not AutoDeploy).

Decoupled/streaming: each request gets its own generate_async(streaming=True)
call on a bounded thread pool, forwarding each incremental text_diff to the
client as it's produced instead of waiting for the full completion.

Admission control: ThreadPoolExecutor.submit() never rejects -- it just
queues unboundedly, so a burst of concurrent requests used to pile up
invisibly (measured: p50 7.4s at concurrency 32 vs 1.75s at concurrency 1,
pure queueing). self._admitted tracks in-flight + already-queued requests;
once it hits MAX_ADMITTED, new requests get an immediate, explicit rejection
response instead of being handed to the pool, so a caller finds out the
system is overloaded in milliseconds instead of after minutes of queueing."""

import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
except Exception as e:
    with open("/workspace/qwen/numpy_debug.log", "w") as f:
        f.write(f"sys.path={sys.path}\n")
        f.write(f"sys.prefix={sys.prefix}\n")
        f.write(f"sys.base_prefix={sys.base_prefix}\n")
        f.write(f"sys.flags={sys.flags}\n")
        traceback.print_exception(type(e), e, e.__traceback__, file=f)
        cause = e.__cause__
        while cause is not None:
            f.write("--- CAUSED BY ---\n")
            traceback.print_exception(type(cause), cause, cause.__traceback__, file=f)
            cause = cause.__cause__
    raise

import triton_python_backend_utils as pb_utils

# Matches the engine's --max_batch_size 16 build config -- bounds the stub
# process's thread pool so a burst of concurrent streaming sessions can't
# spawn unbounded threads.
# Raised from 8 -- measured (docs/nemotron-batch-size-scaling.md) to still
# be within the "nearly free" flat-latency zone: p50 0.673s (concurrency 1)
# -> 0.726s (concurrency 16), vs. max_batch_size=24/32 which already show
# 38-46% p50 growth at their own ceiling concurrency (compute/bandwidth
# crossover, not a VRAM limit -- free VRAM was ~1.6GiB and unchanged across
# 8/16/24/32 with all four models loaded).
MAX_CONCURRENT_STREAMS = 16

# Admission ceiling for in-flight + queued requests, i.e. anything already
# accepted into self._pool but not yet finished. 2x the worker count: enough
# slack to absorb a short burst beyond what's actively running (so requests
# don't get rejected the instant all 8 workers are briefly busy), but bounded
# so queueing can't run away the way it did with an unbounded
# ThreadPoolExecutor queue. Requests beyond this are rejected immediately
# rather than queued.
MAX_ADMITTED = 2 * MAX_CONCURRENT_STREAMS


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        engine_dir = params["engine_dir"]["string_value"]
        tokenizer_dir = params["tokenizer_dir"]["string_value"]

        from tensorrt_llm.llmapi.llm import _TrtLLM
        from tensorrt_llm import SamplingParams
        from transformers import AutoTokenizer

        self.SamplingParams = SamplingParams
        # Rendered ourselves via apply_chat_template() rather than passed as
        # a messages list to generate_async() -- keeps template rendering
        # explicit and independent of whatever TRT-LLM's own chat-template
        # handling does or doesn't do, and matches the checkpoint's actual
        # chat_template.jinja (Qwen3's ChatML format: <|im_start|>/<|im_end|>)
        # exactly, since it's the same HF tokenizer class reading the same
        # tokenizer_config.json.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.llm = _TrtLLM(
            model=engine_dir,
            tokenizer=tokenizer_dir,
            # Capped so the LLM doesn't greedily claim ~all free VRAM for KV cache pool —
            # this GPU is shared with the ASR and TTS stages.
            kv_cache_config={"free_gpu_memory_fraction": 0.2},
            # The checkpoint's config.json declares a 128K max_position_embeddings; with
            # no explicit cap here that gets inherited as the KV-cache sizing basis
            # (scaled further by an implicit default batch size of 8) and OOMs on this
            # 16GB GPU. 4096/8 matches the old AOT-compiled engine's hard-coded limits —
            # far more than a voice pipeline turn needs.
            max_seq_len=4096,
            max_batch_size=16,  # raised from 8 -- see docs/nemotron-batch-size-scaling.md
        )
        self._pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STREAMS)
        self._admitted = 0
        self._admitted_lock = threading.Lock()

    def execute(self, requests):
        for request in requests:
            if self._try_admit():
                self._pool.submit(self._stream_one, request)
            else:
                self._reject(request)
        return None  # decoupled: no synchronous response list

    def _try_admit(self):
        with self._admitted_lock:
            if self._admitted >= MAX_ADMITTED:
                return False
            self._admitted += 1
            return True

    def _reject(self, request):
        sender = request.get_response_sender()
        sender.send(
            pb_utils.InferenceResponse(
                error=pb_utils.TritonError(
                    f"qwen_llm is overloaded: {MAX_ADMITTED} requests already "
                    "in flight or queued. Try again shortly."
                )
            ),
            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
        )

    def _stream_one(self, request):
        sender = request.get_response_sender()
        try:
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            transcript = prompt_tensor.as_numpy().flatten()[0]
            if isinstance(transcript, bytes):
                transcript = transcript.decode("utf-8")

            # The raw ASR transcript was previously sent straight to
            # generate_async() with no chat template at all -- the model had
            # no way to tell "answer this" from "continue this sentence",
            # and treated every prompt as free-text continuation. Measured
            # (on the prior Nemotron checkpoint): 8/8 realistic
            # voice-assistant prompts hit the 256-token cap with rambling,
            # off-topic output ("What's the capital of France" -> a
            # multi-paragraph tangent that never says "Paris").
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Answer the following directly and concisely, in 1-2 short "
                        "sentences suitable for being spoken aloud. Do not show your "
                        f"reasoning.\n\n{transcript}"
                    ),
                },
            ]
            # enable_thinking=False is Qwen3's native reasoning-mode-off toggle:
            # its chat_template.jinja checks this exact kwarg (not a magic system
            # string) and, when false, appends an already-closed
            # <think>\n\n</think>\n\n itself right after the generation-prompt
            # header -- same effect the previous Nemotron checkpoint needed a
            # manual prompt-string hack to achieve, done natively here instead.
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )

            sampling_params = self.SamplingParams(
                # Lowered from 256 after measuring: at 256, this checkpoint's rambling
                # responses (see the chat-template fix's own follow-up notes) never
                # reach a natural stop even by token 256 -- truncating the same
                # deterministic (temperature=0) token stream at 64/96/128/160/192 showed
                # no case where a coherent answer got cut off earlier, since none of the
                # measured responses resolve to a coherent, complete answer within the
                # 256-token window regardless of where the cut is. So the cap is a pure
                # latency/cost bound here, not a content-completeness tradeoff. 96 gives
                # ~1.5x headroom over what a well-behaved "1-2 short spoken sentences"
                # answer needs (roughly 30-60 tokens), while cutting worst-case latency
                # by roughly 2.5x versus 256. A separate experiment strengthening the
                # user-turn instruction (single direct sentence, no hedging, no lists)
                # did NOT help -- it made responses more meta/confused (asking for
                # clarification on plain factual questions) while still hitting the cap
                # 8/8 -- so that change was reverted; only max_tokens changed here.
                max_tokens=96,
                temperature=0,
                # temperature=0 (exact greedy) with no repetition penalty degenerates into
                # repeated phrases specifically under this engine's classic TensorRT backend
                # (not reproduced on vLLM or TensorRT-LLM's PyTorch backend on the same
                # checkpoint/settings). This keeps decoding deterministic while discouraging
                # the loop. Re-verified in docs/nemotron-token-cap-investigation.md: dropping
                # to 1.0 reintroduced the loop on 10/69 eval prompts while barely moving the
                # 96-token cap-hit rate (67/69 -> 65/69, noise) -- not the cause of the cap
                # problem, so left at 1.15.
                repetition_penalty=1.15,
                # EXPERIMENT B (docs/nemotron-token-cap-investigation.md): the model
                # reliably forms one short, complete first sentence (median ~16-21
                # tokens across a 69-prompt eval, well inside the "1-2 short sentences"
                # target) then drifts into unrelated step-by-step/meta-commentary
                # rambling after a paragraph break, almost never producing a real EOS
                # before the 96-token cap. "\n\n" reliably marks that transition in
                # measured output (a single short spoken answer has no reason to contain
                # a paragraph break at all). "</think>" additionally guards against a
                # rarer artifact where the model hallucinates a stray, unopened </think>
                # tag mid-response (a side effect of the forced-empty-think-block fix --
                # see docs/nemotron-response-quality.md new artifact class section).
                # "<|im_end|>" is Qwen3's chat template end-of-turn token (its
                # eos_token per tokenizer_config.json), kept as belt-and-suspenders
                # alongside whatever EOS handling generate_async() does on its own,
                # given this engine backend has already needed workarounds for
                # decoding quirks the PyTorch/vLLM backends do not reproduce (see
                # repetition_penalty above).
                stop=["<|im_end|>", "\n\n", "</think>"],
            )
            result = self.llm.generate_async(prompt, sampling_params, streaming=True)
            for output in result:  # blocking sync iteration -- fine on a pool thread
                diff = output.outputs[0].text_diff
                if diff:
                    out_tensor = pb_utils.Tensor(
                        "GENERATED_TEXT", np.array([diff.encode("utf-8")], dtype=np.object_)
                    )
                    sender.send(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            sender.send(None, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        except Exception as e:
            sender.send(
                pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e))),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            )
        finally:
            # Always release the admission slot, even on error/exception above,
            # so a failure can't leak slots and permanently wedge admission.
            with self._admitted_lock:
                self._admitted -= 1

    def finalize(self):
        self._pool.shutdown(wait=False)
        self.llm = None
