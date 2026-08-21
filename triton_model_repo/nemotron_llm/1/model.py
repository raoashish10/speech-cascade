"""Triton Python backend for Llama-3.1-Nemotron-Nano-4B-v1.1 (FP8), served via a
TensorRT-LLM AOT-compiled engine (classic TensorRT backend, not AutoDeploy)."""

import json
import sys
import traceback

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
        )

    def execute(self, requests):
        prompts = []
        for request in requests:
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            prompt = prompt_tensor.as_numpy().flatten()[0]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            prompts.append(prompt)

        outputs = self.llm.generate(prompts, self.SamplingParams(max_tokens=256, temperature=0))

        responses = []
        for output in outputs:
            text = output.outputs[0].text
            out_tensor = pb_utils.Tensor(
                "GENERATED_TEXT", np.array([text.encode("utf-8")], dtype=np.object_)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        self.llm = None
