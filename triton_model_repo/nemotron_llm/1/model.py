"""Triton Python backend for Llama-3.1-Nemotron-Nano-4B-v1.1 (NVFP4), served by
loading the quantized HF checkpoint directly via TensorRT-LLM's LLM API (JIT
graph build, classic TensorRT backend, not AutoDeploy).

Decoupled/streaming: each request gets its own generate_async(streaming=True)
call on a bounded thread pool, forwarding each incremental text_diff to the
client as it's produced instead of waiting for the full completion."""

import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
except Exception as e:
    with open("/workspace/nemotron/numpy_debug.log", "w") as f:
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

# Matches the engine's --max_batch_size 8 build config -- bounds the stub
# process's thread pool so a burst of concurrent streaming sessions can't
# spawn unbounded threads.
MAX_CONCURRENT_STREAMS = 8


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        engine_dir = params["engine_dir"]["string_value"]
        tokenizer_dir = params["tokenizer_dir"]["string_value"]

        from tensorrt_llm.llmapi.llm import _TrtLLM
        from tensorrt_llm import SamplingParams

        self.SamplingParams = SamplingParams
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
            max_batch_size=8,
        )
        self._pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STREAMS)

    def execute(self, requests):
        for request in requests:
            self._pool.submit(self._stream_one, request)
        return None  # decoupled: no synchronous response list

    def _stream_one(self, request):
        sender = request.get_response_sender()
        try:
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            prompt = prompt_tensor.as_numpy().flatten()[0]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")

            sampling_params = self.SamplingParams(
                max_tokens=256,
                temperature=0,
                # temperature=0 (exact greedy) with no repetition penalty degenerates into
                # repeated phrases specifically under this engine's classic TensorRT backend
                # (not reproduced on vLLM or TensorRT-LLM's PyTorch backend on the same
                # checkpoint/settings). This keeps decoding deterministic while discouraging
                # the loop.
                repetition_penalty=1.15,
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

    def finalize(self):
        self._pool.shutdown(wait=False)
        self.llm = None
