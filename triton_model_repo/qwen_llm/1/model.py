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
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

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

import torch
import triton_python_backend_utils as pb_utils

# Marks each generation call in nsys's timeline (see deploy/PROFILING.md) --
# torch.cuda.nvtx push/pop is a cheap call into nvToolsExt, effectively free
# when no profiler is attached to the process. Without this, nsys shows an
# unlabeled wall of CUDA kernels with no indication of which pipeline stage
# produced which one. torch.profiler (used for chatterbox_tts instead --
# see chatterbox_worker.py) can't see inside this model's execution: the
# actual generation happens inside TensorRT-LLM's own compiled engine/MPI
# worker, not as Python-visible ATen ops, so nsys is the only tool here that
# shows real per-kernel timing for qwen_llm. Set NSYS_NVTX=0 to disable.
_NVTX_ENABLED = os.environ.get("NSYS_NVTX", "1") != "0"


@contextmanager
def nvtx_range(name):
    if _NVTX_ENABLED and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


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

        # PYTHONHOME=/venv/main (set by speech-cascade-triton.sh so this
        # stub process resolves numpy/site-packages correctly at startup --
        # see README.md quirk #3) is fatal to any FRESH python process
        # spawned later: importing tensorrt_llm below transitively imports
        # mpi4py.MPI, which -- as an import side effect -- spawns an `orted`
        # MPI singleton daemon that inherits os.environ AT IMPORT TIME and
        # keeps that copy for its own lifetime (spawning it earlier, e.g.
        # only before _TrtLLM's constructor runs, is too late: the daemon
        # is already forked with the poisoned environment baked in by
        # then). Every worker orted spawns after that inherits PYTHONHOME
        # from it, and on a venv created the plain `python3.12 -m venv` way
        # (REBUILD.md step 3), that breaks `import ctypes` outright
        # (`undefined symbol: _PyErr_SetLocaleString` in _ctypes...so) in
        # those workers -- which mpi4py needs transitively, so the MPI
        # spawn dies and this model never loads. Must run before the
        # tensorrt_llm import line below, not just before _TrtLLM(...).
        # Confirmed empirically: identical `python3.12 -c "import ctypes"`
        # fails with PYTHONHOME=/venv/main set, succeeds without it; this
        # process's own already-running interpreter is unaffected by
        # popping it now (PYTHONHOME only matters at process startup).
        os.environ.pop("PYTHONHOME", None)

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
                    # See docs/fake-confirmation-fix.md: without this,
                    # every one of 8/8 "unfulfillable command" prompts
                    # ("set a timer", "turn off the lights") produced a
                    # confident, fabricated "I've done it" claim instead of
                    # an honest refusal -- this pipeline has zero real
                    # action/tool-calling capability anywhere downstream of
                    # qwen_llm, so there is no legitimate case where
                    # claiming success is correct for that category. Root-
                    # caused empirically (not guessed): reproduces on the
                    # unquantized BF16 checkpoint too, just less severely
                    # (3/8 honest refusals vs 0/8 on this NVFP4 checkpoint),
                    # is largely resistant to temperature>0 resampling
                    # (7/8 prompts fabricate in 5/5 samples), and
                    # enable_thinking=True only helps 2/8 even with an
                    # unbounded token budget -- so this explicit instruction
                    # is the actual fix, not a workaround for a decoding
                    # artifact. Cuts the failure rate to 2/8 (still
                    # disclosed as unresolved for those two).
                    "role": "system",
                    "content": (
                        "You are a voice assistant with no ability to take "
                        "real-world actions: you cannot set timers or "
                        "alarms, control smart-home devices, send texts or "
                        "make calls, place orders, or perform any task "
                        "outside of generating a spoken response. If asked "
                        "to do something you cannot actually do, say so "
                        "honestly in one short sentence instead of "
                        "claiming you did it."
                    ),
                },
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
            with nvtx_range("qwen_llm.generate"):
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
