"""Triton Python backend for Chatterbox-Turbo TTS (ResembleAI, MIT license),
served via the reference implementation's own PyTorch runtime, not the
community vLLM port -- see docs/tts-replacement-investigation.md for why:
the vLLM port only accelerates the T3 (Llama-backbone) half, the S3Gen
vocoder stage still runs unaccelerated either way, and this reference-impl
path is what was actually validated end-to-end (1039-turn, 0-error, 15-min
composed ASR->LLM->TTS soak test with concurrent LLM load).

Replaces kokoro_tts (dead as of this deploy: its `kokoro-onnx` package was
uninstalled from /venv/main and its weight files no longer exist on this
instance) and supersedes the never-actually-deployed magpie_tts source in
the git repo (magpie-tts was investigated and rejected -- see
docs/magpie-tts-investigation.md referenced by task2-tts.md; its own
Triton wiring was never completed or run).

Unlike Kokoro/Magpie, Chatterbox's torch/torchaudio pins are incompatible
with the TensorRT-LLM stack installed in /venv/main (the environment
Triton's python backend stub otherwise uses for every model in this repo).
Rather than risk that conflict -- or an unverified Triton
EXECUTION_ENV_PATH/conda-pack integration -- this model spawns Chatterbox
as a persistent subprocess in its own already-built, already-validated
/venv/chatterbox, communicating over a pipe. Same architecture already
proven in this project's own stress-test harness
(scripts/pipeline_stress_test.py + chatterbox_worker.py's shared ancestor);
see chatterbox_worker.py alongside this file for the worker side.

Interface kept identical to kokoro_tts/magpie_tts so voice_pipeline and the
streaming gateway need only a name change: TEXT (+ optional VOICE, currently
accepted but unused -- see note below) in, AUDIO_SAMPLES (+ SAMPLE_RATE)
out.

Known limitation, stated plainly rather than silently ignored: this
investigation validated exactly ONE reference voice (a single ~41s
reference clip, config parameter `ref_audio_path`); VOICE is accepted for
interface compatibility but does not yet select between multiple voices
the way Kokoro's/Magpie's did. Multi-voice support would mean embedding
multiple reference clips at startup and switching cached conditioning per
request -- not built here.

instance_group.count is 1 (a single persistent worker subprocess, single
CUDA context) -- replicating instances the way Kokoro's count:4 did
(docs/kokoro-tts-capacity-fix.md) isn't viable here: Chatterbox alone needs
~3GB VRAM and the full 4-model pipeline already uses ~14GB of this GPU's
16GB, leaving no headroom for a second copy. Instead, concurrent requests
are batched through this ONE instance: config.pbtxt enables dynamic_batching
with max_batch_size > 1, so Triton itself groups multiple pending requests
into a single execute() call below, and chatterbox_worker.py's
generate_batch() runs them through ONE shared T3 decode pass (see its
module docstring and chatterbox_profiling/ for why -- the autoregressive
decode loop was profiled as kernel-launch-overhead-bound, not compute-bound,
so batching amortizes that overhead across requests rather than shaving it
per-request).
"""

import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
import triton_python_backend_utils as pb_utils

# Backend switch: CHATTERBOX_BACKEND=vllm (default) runs T3's decode through
# vLLM (see chatterbox_profiling/vllm/ -- ~5-5.4x faster than the batched+
# CUDA-graph PyTorch path, kernel fusion via torch.compile, not just launch-
# overhead amortization). CHATTERBOX_BACKEND=pytorch falls back to the
# original eager/CUDA-graph PyTorch worker (chatterbox_worker.py) -- kept
# available, not deleted, in case vLLM regresses under real production load
# in a way this investigation's profiling runs didn't surface.
_BACKEND = os.environ.get("CHATTERBOX_BACKEND", "vllm")
if _BACKEND == "vllm":
    CHATTERBOX_PYTHON = "/venv/vllm/bin/python3"
    WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "chatterbox_worker_vllm.py")
    CHATTERBOX_NPP_LIB = None  # vLLM's own nvidia-*-cu12 wheels are self-contained; no torchcodec dependency here
else:
    CHATTERBOX_PYTHON = "/venv/chatterbox/bin/python3"
    WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "chatterbox_worker.py")
    # See chatterbox_worker.py's own docstring and
    # docs/tts-replacement-investigation.md's harness-bug #3: the worker
    # subprocess must NOT inherit Triton's own LD_LIBRARY_PATH (set for
    # /venv/main's TensorRT-LLM libs) -- it needs its own venv's NPP library
    # path for torchcodec's audio-save dependency. Every turn's generation
    # would succeed while the final save silently failed if this were wrong.
    CHATTERBOX_NPP_LIB = "/venv/chatterbox/lib/python3.12/site-packages/nvidia/npp/lib"
PROTO = "@@PROTO@@"


def _read_proto_line(pipe, proc):
    while True:
        line = pipe.readline()
        if not line:
            return None
        line = line.strip()
        if line.startswith(PROTO):
            return line[len(PROTO):].strip()
        if proc.poll() is not None:
            return None


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        ref_audio_path = params["ref_audio_path"]["string_value"]

        # Strip PYTHONHOME/PYTHONPATH: Triton's supervisor script sets these
        # globally for /venv/main (see deploy/supervisor/speech-cascade-triton.sh),
        # and they leak into any subprocess via os.environ inheritance. Left
        # in place, /venv/chatterbox/bin/python3 resolves sys.path against
        # /venv/main's stdlib instead of its own, crashing on the first
        # import (confirmed: "undefined symbol: _PyErr_SetLocaleString" from
        # a cross-build-mismatched _ctypes) well before it can print
        # anything -- which is why this failure came back as a bare `None`
        # rather than a real error message the first time this was wired up.
        worker_env = dict(os.environ)
        worker_env.pop("PYTHONHOME", None)
        worker_env.pop("PYTHONPATH", None)
        if CHATTERBOX_NPP_LIB is not None:
            worker_env["LD_LIBRARY_PATH"] = CHATTERBOX_NPP_LIB
        else:
            worker_env.pop("LD_LIBRARY_PATH", None)

        worker_args = [CHATTERBOX_PYTHON, WORKER_SCRIPT, ref_audio_path]
        if _BACKEND == "vllm":
            # Registering our out-of-tree T3 model class only happens in
            # whichever process imports chatterbox_worker_vllm.py -- vLLM's
            # V1 engine normally spawns a SEPARATE subprocess for the actual
            # model executor (via 'spawn', since CUDA is already
            # initialized), which would never see that registration and
            # fail with "architecture not supported". Keeping everything
            # in-process sidesteps that (see chatterbox_profiling/vllm/).
            worker_env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            # Explicit config.pbtxt parameter, not left to inherit from
            # whatever launched tritonserver -- see the parameter's own
            # comment in config.pbtxt for why this needs to be lower when
            # chatterbox_tts shares the GPU with the rest of the pipeline.
            worker_env["CHATTERBOX_VLLM_GPU_MEM_UTIL"] = params.get("vllm_gpu_mem_util", {}).get(
                "string_value", "0.3"
            )
            worker_args.append(params.get("vllm_t3_model_dir", {}).get(
                "string_value", "/workspace/speech-cascade-inference/vllm_t3_model_dir"
            ))

        self.proc = subprocess.Popen(
            worker_args,
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=worker_env,
        )
        ready = _read_proto_line(self.proc.stderr, self.proc)
        if ready != "WORKER_READY":
            raise pb_utils.TritonModelException(
                f"chatterbox worker failed to start: {ready!r}"
            )

    def execute(self, requests):
        # Triton's dynamic_batching (config.pbtxt) may hand us anywhere from
        # 1 to max_batch_size requests here in one call -- collect all their
        # texts and send them to the worker as ONE batch, rather than looping
        # a round-trip per request the way this used to (see model.py's own
        # docstring and chatterbox_worker.py's for why that left the GPU
        # mostly idle between tiny per-request kernel launches).
        texts = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            text = text_tensor.as_numpy().flatten()[0]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            texts.append(text.replace("\n", " "))

        self.proc.stdin.write(json.dumps(texts) + "\n")
        self.proc.stdin.flush()
        result = _read_proto_line(self.proc.stderr, self.proc)

        if result is None or not result.startswith("OK "):
            # Whole batch failed (e.g. the worker died, or something outside
            # any single item's generation raised) -- every request in this
            # call gets the same error, since we can't attribute it to one.
            err = result or "worker process died"
            return [
                pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"chatterbox_tts failed: {err}")
                )
                for _ in requests
            ]

        payload = json.loads(result[len("OK "):])
        items = payload["results"]
        if len(items) != len(requests):
            return [
                pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"chatterbox_tts protocol error: sent {len(requests)} texts, "
                        f"got {len(items)} results back"
                    )
                )
                for _ in requests
            ]

        responses = []
        for item in items:
            if not item.get("ok"):
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            f"chatterbox_tts failed: {item.get('error', 'unknown error')}"
                        )
                    )
                )
                continue

            wav_path, sr = item["path"], item["sr"]
            samples, sr = sf.read(wav_path, dtype="float32")
            os.remove(wav_path)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)

            audio_out = pb_utils.Tensor("AUDIO_SAMPLES", samples.astype(np.float32))
            sr_out = pb_utils.Tensor("SAMPLE_RATE", np.array([int(sr)], dtype=np.int32))
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[audio_out, sr_out])
            )
        return responses

    def finalize(self):
        if getattr(self, "proc", None) is not None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
            except Exception:
                pass
            self.proc.terminate()
