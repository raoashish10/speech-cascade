"""Production Chatterbox-Turbo TTS worker, T3 decode powered by vLLM instead
of eager/CUDA-graph PyTorch (see chatterbox_profiling/ for the full
investigation: batching -41% self-CUDA time, CUDA graphs +13.5% more,
vLLM ~5-5.4x faster than the batched+CUDA-graph PyTorch path -- vLLM's
torch.compile backend does real kernel fusion, not just launch-overhead
amortization).

Runs under /venv/vllm (not /venv/chatterbox) -- chatterbox-tts is installed
there too (--no-deps, alongside its non-torch dependencies only, so vLLM's
own torch/tokenizers/etc. stay at the versions IT needs; see deploy notes).
Conditioning, tokenization, and S3Gen vocoding all reuse the real, unmodified
ChatterboxTurboTTS class exactly as the PyTorch worker does -- only the T3
decode step itself (previously t3.inference_turbo) is replaced by a vLLM
LLM.generate() call against T3's GPT2 backbone, registered as a vLLM
out-of-tree model (vllm_t3/chatterbox_t3_model.py). model.t3.tfmr (the GPT2
backbone) is deleted right after conditioning setup, since vLLM loads its
own copy of those same weights and running two would waste GPU memory for
no benefit -- prepare_input_embeds() only needs cond_enc/text_emb/speech_emb,
never self.tfmr.

Same JSON-array-in/JSON-array-out protocol as chatterbox_worker.py (see that
file's docstring) -- model.py needs no changes, just point it at this script
and /venv/vllm/bin/python3 instead.
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vllm_t3"))

import soundfile as sf
import torch

REF_AUDIO = sys.argv[1]
VLLM_MODEL_DIR = sys.argv[2] if len(sys.argv) > 2 else "/workspace/speech-cascade-inference/vllm_t3_model_dir"

PROTO = "@@PROTO@@"


def reply(msg):
    print(f"{PROTO} {msg}", file=sys.stderr, flush=True)


# --- Register the vLLM out-of-tree T3 model (must happen before LLM(...)) ---
from transformers import AutoConfig
from configuration_chatterbox import ChatterboxTurboConfig
AutoConfig.register("chatterbox_turbo", ChatterboxTurboConfig)

from vllm import LLM, ModelRegistry, SamplingParams
from chatterbox_t3_model import ChatterboxTurboT3ForGeneration
ModelRegistry.register_model("ChatterboxTurboT3ForGeneration", ChatterboxTurboT3ForGeneration)

# --- Conditioning, tokenization, S3Gen vocoding: the real, unmodified class ---
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.models.s3gen.const import S3GEN_SIL

model = ChatterboxTurboTTS.from_pretrained(device="cuda")
model.prepare_conditionals(REF_AUDIO)

# Free the redundant GPT2 backbone -- vLLM loads its own copy of the same
# weights below, and prepare_input_embeds() only touches cond_enc/text_emb/
# speech_emb, never self.tfmr.
del model.t3.tfmr
torch.cuda.empty_cache()

llm = LLM(
    model=VLLM_MODEL_DIR,
    trust_remote_code=True,
    dtype="float32",
    max_model_len=2048,
    gpu_memory_utilization=float(os.environ.get("CHATTERBOX_VLLM_GPU_MEM_UTIL", "0.3")),
    enforce_eager=False,
    skip_tokenizer_init=True,
    enable_prompt_embeds=True,
)

SAMPLING_PARAMS = SamplingParams(
    temperature=0.8,
    top_k=1000,
    top_p=0.95,
    repetition_penalty=1.2,
    max_tokens=1000,
    stop_token_ids=[model.t3.hp.stop_speech_token],
    detokenize=False,
)

reply("WORKER_READY")

for line in sys.stdin:
    line = line.strip()
    if not line or line == "QUIT":
        break
    try:
        texts = json.loads(line)
        if not isinstance(texts, list) or not texts:
            raise ValueError(f"expected a non-empty JSON array of texts, got {line!r}")

        t0 = time.time()

        # Build each text's own prompt_embeds via the exact, unmodified
        # T3.prepare_input_embeds() the PyTorch path already uses -- no
        # left-padding needed, vLLM's own scheduler handles variable-length
        # prompts natively (see chatterbox_profiling/vllm/README.md).
        prompts = []
        with torch.inference_mode():
            for text in texts:
                text_tokens = model.tokenizer(text, return_tensors="pt").input_ids.to(model.device)
                speech_start_token = model.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
                embeds, _len_cond = model.t3.prepare_input_embeds(
                    t3_cond=model.conds.t3, text_tokens=text_tokens,
                    speech_tokens=speech_start_token, cfg_weight=0.0,
                )
                prompts.append({"prompt_embeds": embeds.squeeze(0).detach()})

        outputs = llm.generate(prompts, sampling_params=SAMPLING_PARAMS, use_tqdm=False)

        results = []
        for out in outputs:
            try:
                speech_tokens = torch.tensor(list(out.outputs[0].token_ids), dtype=torch.long, device=model.device)
                speech_tokens = speech_tokens[speech_tokens < 6561]
                silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(model.device)
                speech_tokens = torch.cat([speech_tokens, silence])

                with torch.inference_mode():
                    wav, _ = model.s3gen.inference(
                        speech_tokens=speech_tokens, ref_dict=model.conds.gen, n_cfm_timesteps=2,
                    )
                wav = wav.squeeze(0).detach().cpu().numpy()
                # Host-RAM leak fix (found via this soak test -- see
                # docs/accelerated-chatterbox-vllm-fresh-deploy-and-soak-test.md):
                # PerthImplicitWatermarker.apply_watermark() defaults to
                # device="cpu" and runs a real nn.Module forward pass
                # (self.perth_net.encoder(...)) with no torch.no_grad()/
                # inference_mode() of its own. Called here outside any such
                # context (unlike every other tensor op in this file), it
                # built a full CPU-resident autograd graph on every single
                # request -- reclaimable only by Python's cyclic GC, not
                # plain refcounting, which under sustained load couldn't
                # keep pace with the allocation rate. Measured: ~5.4-5.5 MB
                # of host RAM leaked per request, constant across
                # concurrency 1 and 4, that drove this container from a
                # healthy baseline to its actual ~45GiB cgroup ceiling
                # within about 25 minutes of sustained load. GPU memory
                # stayed completely flat throughout -- consistent with this
                # being a host-only (CPU autograd), not GPU, leak.
                with torch.inference_mode():
                    watermarked_wav = model.watermarker.apply_watermark(wav, sample_rate=model.sr)

                fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="chatterbox_")
                os.close(fd)
                sf.write(out_path, watermarked_wav.astype("float32"), model.sr)
                results.append({"ok": True, "path": out_path, "sr": model.sr})
            except Exception as item_e:
                results.append({"ok": False, "error": str(item_e)})

        elapsed = time.time() - t0
        reply("OK " + json.dumps({"elapsed": elapsed, "results": results}))
    except Exception as e:
        reply(f"ERR {e}")
