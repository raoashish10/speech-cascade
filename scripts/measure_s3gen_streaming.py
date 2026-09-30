"""Measure what S3Gen streaming would actually buy, before building it.

Runs inside /venv/vllm on a GPU pod, reusing exactly the load path
chatterbox_worker_vllm.py uses, and answers three things:

  1. THE SPLIT. tts_first_ms (1686ms measured end to end) covers T3 decode
     plus S3Gen plus watermarking plus a temp-wav round trip. Nobody has
     measured the proportions. If T3 dominates, chunking S3Gen buys nothing,
     because S3Gen cannot start until T3 has produced the tokens.

  2. TIME TO FIRST CHUNK. Run S3Gen on a prefix of the speech tokens rather
     than all of them, at several chunk sizes, and time it. This is the
     ceiling on what streaming can deliver -- the real thing has to be at
     least this slow.

  3. WHETHER THE AUDIO SURVIVES IT. Each chunk is an independent S3Gen call
     here, with no cross-chunk flow state, which is the crude version. Audio
     is written out so the joins can be listened to, and the RMS
     discontinuity at each join is reported as a cheap numeric proxy.

HOW TO RUN IT. The GPU pod's own SSH (both the Runpod proxy and direct) was
unusable during this work, so the reliable way is to make the measurement the
container's entrypoint and read the results out of the pod logs:

    entrypoint: ["bash", "-c",
        "bash /opt/speech-cascade/scripts/build_models_from_scratch.sh && "
        "<write this file to /tmp/m.py> && "
        "LD_LIBRARY_PATH=/venv/vllm/lib/python3.12/site-packages/nvidia/cudnn/lib:"
        "/venv/vllm/lib/python3.12/site-packages/nvidia/cublas/lib "
        "/venv/vllm/bin/python3 -u /tmp/m.py; sleep infinity"]

That LD_LIBRARY_PATH is not optional and is not a detail of this script: it
is the same one chatterbox_tts/1/model.py sets for the worker subprocess,
and without it the linker resolves the base image's older system libcudnn
instead of vLLM's bundled one and the voice encoder's LSTM fails.

Note also that the entrypoint must NOT leave tritonserver running alongside
this: the two would each take their own vLLM allocation of the GPU and the
timings would measure contention. Do not try to free the GPU by killing
tritonserver on a normally-started pod either -- the image's entrypoint ends
in `wait "${TRITON_PID}"`, so killing it takes the container with it.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/workspace/speech-cascade-inference/triton_model_repo/chatterbox_tts/1/vllm_t3")

import numpy as np
import soundfile as sf
import torch

OUT = "/workspace/s3gen_measure"
os.makedirs(OUT, exist_ok=True)

TEXTS = [
    "The sky appears blue because shorter wavelengths of light (blue) scatter "
    "more easily in Earth's atmosphere.",
    "A neural network is a model made of layers of simple units that learn "
    "patterns from data by adjusting the strength of their connections.",
    "The capital of France is Paris.",
]
CHUNK_SIZES = [25, 50, 100]   # speech tokens; 25Hz token rate -> 1s, 2s, 4s
REPEATS = 3

from transformers import AutoConfig  # noqa: E402
from configuration_chatterbox import ChatterboxTurboConfig  # noqa: E402
AutoConfig.register("chatterbox_turbo", ChatterboxTurboConfig)

from vllm import LLM, ModelRegistry, SamplingParams  # noqa: E402
from chatterbox_t3_model import ChatterboxTurboT3ForGeneration  # noqa: E402
ModelRegistry.register_model("ChatterboxTurboT3ForGeneration", ChatterboxTurboT3ForGeneration)

from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402
from chatterbox.models.s3gen.const import S3GEN_SIL  # noqa: E402

print("loading chatterbox...", flush=True)
model = ChatterboxTurboTTS.from_pretrained(device="cuda")
del model.t3.tfmr
torch.cuda.empty_cache()

llm = LLM(
    model="/workspace/speech-cascade-inference/vllm_t3_model_dir",
    trust_remote_code=True, dtype="float32", max_model_len=2048,
    gpu_memory_utilization=0.3, enforce_eager=False,
    skip_tokenizer_init=True, enable_prompt_embeds=True,
)
SP = SamplingParams(temperature=0.8, top_k=1000, top_p=0.95,
                    repetition_penalty=1.2, max_tokens=1000,
                    stop_token_ids=[model.t3.hp.stop_speech_token],
                    detokenize=False)


def t3_tokens(text):
    """T3 decode via vLLM -- exactly the worker's path. Returns (tokens, sec)."""
    t0 = time.perf_counter()
    with torch.inference_mode():
        tt = model.tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        start = model.t3.hp.start_speech_token * torch.ones_like(tt[:, :1])
        embeds, _ = model.t3.prepare_input_embeds(
            t3_cond=model.conds.t3, text_tokens=tt, speech_tokens=start, cfg_weight=0.0)
    out = llm.generate([{"prompt_embeds": embeds.squeeze(0).detach()}],
                       sampling_params=SP, use_tqdm=False)
    toks = torch.tensor(list(out[0].outputs[0].token_ids), dtype=torch.long,
                        device=model.device)
    toks = toks[toks < 6561]
    return toks, time.perf_counter() - t0


def s3gen(tokens, add_silence=True):
    """One S3Gen call. Returns (waveform, sec)."""
    t0 = time.perf_counter()
    tk = tokens
    if add_silence:
        sil = torch.tensor([S3GEN_SIL] * 3).long().to(model.device)
        tk = torch.cat([tk, sil])
    with torch.inference_mode():
        wav, _ = model.s3gen.inference(speech_tokens=tk, ref_dict=model.conds.gen,
                                       n_cfm_timesteps=2)
    wav = wav.squeeze(0).detach().cpu().numpy()
    return wav, time.perf_counter() - t0


def watermark(wav):
    t0 = time.perf_counter()
    with torch.inference_mode():
        w = model.watermarker.apply_watermark(wav, sample_rate=model.sr)
    return w, time.perf_counter() - t0


def join_discontinuity(wav, joins, win=240):
    """RMS ratio across each concatenation point. 1.0 = seamless; a large
    value means the waveform jumps where two chunks were stitched. A crude
    proxy for 'can you hear the join', not a substitute for listening."""
    out = []
    for j in joins:
        if j < win or j + win > len(wav):
            continue
        before = np.sqrt(np.mean(wav[j - win:j] ** 2)) + 1e-9
        after = np.sqrt(np.mean(wav[j:j + win] ** 2)) + 1e-9
        out.append(float(max(before, after) / min(before, after)))
    return out


results = []
for text in TEXTS:
    print(f"\n{'=' * 78}\n{text[:70]}...\n{'=' * 78}", flush=True)
    # Warm up so the first timed run isn't paying compile/alloc costs.
    toks, _ = t3_tokens(text)
    s3gen(toks)

    rec = {"text": text, "n_tokens": int(len(toks))}

    t3_times, full_times, wm_times = [], [], []
    for _ in range(REPEATS):
        toks, dt = t3_tokens(text)
        t3_times.append(dt)
        wav, ds = s3gen(toks)
        full_times.append(ds)
        _, dw = watermark(wav)
        wm_times.append(dw)
    rec["t3_ms"] = float(np.median(t3_times) * 1000)
    rec["s3gen_full_ms"] = float(np.median(full_times) * 1000)
    rec["watermark_ms"] = float(np.median(wm_times) * 1000)
    rec["audio_sec"] = len(wav) / model.sr
    sf.write(f"{OUT}/full_{len(results)}.wav", wav, model.sr)

    print(f"  tokens={rec['n_tokens']}  audio={rec['audio_sec']:.2f}s")
    print(f"  T3 (vLLM)     {rec['t3_ms']:8.1f} ms")
    print(f"  S3Gen (full)  {rec['s3gen_full_ms']:8.1f} ms")
    print(f"  watermark     {rec['watermark_ms']:8.1f} ms", flush=True)

    rec["chunked"] = {}
    for cs in CHUNK_SIZES:
        if len(toks) <= cs:
            continue
        pieces, times = [], []
        for i in range(0, len(toks), cs):
            window = toks[i:i + cs]
            last = (i + cs) >= len(toks)
            w, dt = s3gen(window, add_silence=last)
            pieces.append(w)
            times.append(dt * 1000)
        joins = list(np.cumsum([len(p) for p in pieces])[:-1])
        cat = np.concatenate(pieces)
        sf.write(f"{OUT}/chunked_{len(results)}_cs{cs}.wav", cat, model.sr)
        disc = join_discontinuity(cat, joins)
        rec["chunked"][cs] = {
            "n_chunks": len(pieces),
            "first_chunk_ms": times[0],
            "total_ms": float(sum(times)),
            "per_chunk_ms": [round(t, 1) for t in times],
            "audio_sec": len(cat) / model.sr,
            "join_discontinuity": [round(d, 2) for d in disc],
        }
        print(f"  chunk={cs:3d} tok -> {len(pieces)} chunks  "
              f"first={times[0]:7.1f} ms  total={sum(times):7.1f} ms  "
              f"joins={[round(d, 2) for d in disc]}", flush=True)
    results.append(rec)

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {OUT}/results.json")
