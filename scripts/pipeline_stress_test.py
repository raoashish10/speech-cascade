"""End-to-end voice-pipeline stress test: real Whisper ASR -> real LLM
(HTTP, OpenAI-compatible) -> real TTS, chained as actual conversation
turns, sustained over DURATION_S, with all three stages loaded
concurrently -- plus a separate concurrent LLM load generator running in
parallel to simulate realistic multi-user pressure on the shared GPU while
this one representative pipeline chain is measured.

This is the harness behind the composed-pipeline soak-test results in
docs/tts-replacement-investigation.md for Chatterbox-Turbo, F5-TTS, and
XTTS-v2. Each TTS candidate needs its own isolated venv (torch/torchaudio
pins conflict across chatterbox/xtts/f5tts) -- see tts_worker.py, run under
that venv's own python as a subprocess, and
docs/tts-replacement-investigation.md's "Resumed" section for how each
venv was built (XTTS-v2 in particular needed a Python 3.11 interpreter via
`uv` plus two compatibility patches -- see that doc before rerunning this
against XTTS-v2 on a fresh instance).

Usage: python3 pipeline_stress_test.py <tts_python_bin> <backend> <ref_audio> <duration_s> <tag> [tts_ld_library_path]
  backend: chatterbox | xtts | f5tts (passed straight through to tts_worker.py)
  tts_ld_library_path: only needed for backends whose venv's torchcodec
    dependency can't find its own NPP libs on the default LD_LIBRARY_PATH
    (chatterbox needed this; xtts/f5tts didn't) -- see tts_worker.py's own
    venv-specific notes and docs/tts-replacement-investigation.md.

Requires: a real Triton whisper_asr-compatible engine at ASR_ENGINE_DIR/
ASR_ASSETS_DIR (adjust the paths below for your deploy layout), and an
OpenAI-compatible LLM server already running at LLM_URL (e.g.
`trtllm-serve serve <checkpoint> --backend pytorch ...` on the port set
below).
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import numpy as np
import torch

sys.path.insert(0, "/workspace/speech-cascade/triton_model_repo/whisper_asr/1")
from trtllm_whisper.whisper_model import WhisperTRTLLM
from trtllm_whisper.whisper_utils import log_mel_spectrogram

TTS_PYTHON = sys.argv[1]
BACKEND = sys.argv[2]
REF_AUDIO = sys.argv[3]
DURATION_S = float(sys.argv[4])
TAG = sys.argv[5]
# The TTS worker subprocess must NOT inherit this parent process's own
# LD_LIBRARY_PATH (set for /venv/main's tensorrt_llm) -- some candidates'
# venvs need their own venv's NPP library path instead (torchcodec's
# audio-save dependency). Inheriting the wrong one silently breaks only the
# final save step while generation itself keeps succeeding -- exactly the
# bug that invalidated the first full run of this script against
# Chatterbox-Turbo.
TTS_LD_LIBRARY_PATH = sys.argv[6] if len(sys.argv) > 6 else ""

ASR_ENGINE_DIR = "/workspace/speech-cascade-inference/models/whisper-base-trtllm"
ASR_ASSETS_DIR = "/workspace/speech-cascade-inference/models/whisper-base-trtllm/assets"
ASR_AUDIO = "/workspace/speech-cascade/tests/fixtures/vad_sample_16k.wav"
LLM_URL = "http://localhost:8123/v1/chat/completions"
LLM_MODEL = "Qwen3-8B-NVFP4"
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses short and "
    "conversational, as if spoken aloud. Avoid lists, markdown, or headers."
)
LOG_PATH = os.environ.get(
    "PIPELINE_STRESS_LOG_PATH",
    os.path.join(tempfile.gettempdir(), f"stress_{TAG}.jsonl"),
)
TTS_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_worker.py")

# --- load Whisper ASR FIRST, before starting any threads ---
# NOTE: WhisperTRTLLM's import/init triggers TensorRT-LLM's MPI-based worker
# spawn (fork+exec). Starting background httpx threads before this is a
# classic fork-after-threading deadlock (a forked child can inherit a
# thread's I/O/allocator lock held mid-operation, with no owner left alive
# to release it) -- hit this exact hang on the first attempt. ASR must load
# to completion before any other thread starts.
asr = WhisperTRTLLM(ASR_ENGINE_DIR, assets_dir=ASR_ASSETS_DIR, batch_size=8,
                     use_py_session=False, num_beams=1, kv_cache_free_gpu_memory_fraction=0.05)
audio, sr = __import__("librosa").load(ASR_AUDIO, sr=16000)
audio = audio[:16000 * 30] if len(audio) > 16000 * 30 else np.pad(audio, (0, 16000 * 30 - len(audio)))
mel = [log_mel_spectrogram(audio, asr.n_mels, device="cuda", mel_filters_dir=ASR_ASSETS_DIR).unsqueeze(0)]
mel_lens = torch.tensor([m.shape[2] for m in mel], dtype=torch.int32, device="cuda")
print(f"[{TAG}] whisper ASR loaded", flush=True)

# --- background LLM load generator (simulates other concurrent users) ---
# Safe to start now: ASR's MPI worker spawn already completed above.
stop_flag = threading.Event()
bg_count = [0]


def bg_worker():
    with httpx.Client(timeout=30) as client:
        while not stop_flag.is_set():
            try:
                client.post(LLM_URL, json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": "Tell me a short fact about the ocean."}],
                    "max_tokens": 96, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                bg_count[0] += 1
            except Exception:
                pass


bg_threads = [threading.Thread(target=bg_worker, daemon=True) for _ in range(4)]
for t in bg_threads:
    t.start()
print(f"[{TAG}] background LLM load generator started (4 workers)", flush=True)

PROTO = "@@PROTO@@"


def read_proto_line(pipe, proc):
    """Read lines until one carries our sentinel prefix, discarding any
    library noise (progress bars, warnings, prints) in between. Returns
    None if the process's stream closed without ever sending one."""
    while True:
        line = pipe.readline()
        if not line:
            return None
        line = line.strip()
        if line.startswith(PROTO):
            return line[len(PROTO):].strip()
        if not proc.poll() is None:
            return None


# --- launch persistent TTS worker subprocess ---
# Worker's protocol replies are tagged with PROTO and sent over stderr (see
# tts_worker.py) -- library noise (progress bars, warnings, prints) can
# land on either stream, so filtering by sentinel prefix rather than by
# stream choice is what actually isolates the channel. Drain stdout to
# devnull so its pipe buffer never fills and blocks the worker.
tts_env = dict(os.environ)
if TTS_LD_LIBRARY_PATH:
    tts_env["LD_LIBRARY_PATH"] = TTS_LD_LIBRARY_PATH
else:
    tts_env.pop("LD_LIBRARY_PATH", None)
tts_proc = subprocess.Popen(
    [TTS_PYTHON, TTS_WORKER_SCRIPT, BACKEND, REF_AUDIO],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1,
    env=tts_env,
)
ready = read_proto_line(tts_proc.stderr, tts_proc)
print(f"[{TAG}] tts worker: {ready}", flush=True)


def gpu_mem_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True)
    return int(out.stdout.strip().split("\n")[0])


start = time.time()
turn = 0
errors = 0
log_f = open(LOG_PATH, "w")

while time.time() - start < DURATION_S:
    turn += 1
    row = {"turn": turn, "t": round(time.time() - start, 1)}
    try:
        # ASR
        t0 = time.time()
        out_ids = asr.process_batch(mel, mel_lens)
        row["asr_s"] = round(time.time() - t0, 3)
        transcript = out_ids[0] if out_ids else "Hello, how can I help?"

        # LLM
        t0 = time.time()
        resp = httpx.post(LLM_URL, json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript or "Hello, how can I help?"},
            ],
            "max_tokens": 96, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=30)
        row["llm_s"] = round(time.time() - t0, 3)
        llm_text = resp.json()["choices"][0]["message"]["content"].strip() or "Okay."

        # TTS
        t0 = time.time()
        tts_proc.stdin.write(llm_text.replace("\n", " ") + "\n")
        tts_proc.stdin.flush()
        tts_result = read_proto_line(tts_proc.stderr, tts_proc) or "ERR no reply (process died)"
        row["tts_s"] = round(time.time() - t0, 3)
        row["tts_result"] = tts_result

        row["total_s"] = round(row["asr_s"] + row["llm_s"] + row["tts_s"], 3)
        row["gpu_mem_mb"] = gpu_mem_mb()
        row["ok"] = tts_result.startswith("OK")
        if not row["ok"]:
            errors += 1
    except Exception as e:
        row["error"] = str(e)
        errors += 1

    log_f.write(json.dumps(row) + "\n")
    log_f.flush()
    if turn % 10 == 0 or turn == 1:
        print(f"[{TAG}] turn {turn} t={row.get('t')}s asr={row.get('asr_s')} llm={row.get('llm_s')} "
              f"tts={row.get('tts_s')} total={row.get('total_s')} mem={row.get('gpu_mem_mb')}MiB "
              f"errors={errors} bg_llm_reqs={bg_count[0]}", flush=True)

stop_flag.set()
tts_proc.stdin.write("QUIT\n")
tts_proc.stdin.flush()
tts_proc.terminate()
log_f.close()

print(f"\n[{TAG}] === STRESS TEST SUMMARY ===", flush=True)
print(f"[{TAG}] total turns: {turn}, errors: {errors}, background LLM reqs: {bg_count[0]}", flush=True)
print(f"[{TAG}] log written to {LOG_PATH}", flush=True)
print(f"[{TAG}] STRESS_TEST_DONE", flush=True)
