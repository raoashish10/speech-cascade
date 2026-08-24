"""Streams a wav file's audio in real-time-simulated chunks to the streaming
gateway over WebSocket, printing/saving LLM text and TTS audio as they arrive.

Usage (local, gateway's internal port -- no auth needed):
    python3 test_streaming_client.py --wav test_tts_output.wav

Usage (from outside the box, through the Caddy-authed external port -- see
streaming_gateway/README.md for the full external-access writeup):
    python3 test_streaming_client.py --wav clip.wav \\
        --gateway-url ws://<PUBLIC_IPADDR>:<VAST_TCP_PORT_10100>/ws/stream \\
        --token "$OPEN_BUTTON_TOKEN" --voice af_heart

The token is sent as a `?token=` query param by default -- that's the only
method a browser's native WebSocket API can use (it can't set custom
headers on the upgrade request), and it's one of the methods Caddy's edge
accepts. Pass --auth-mode header to send `Authorization: Bearer <token>`
instead (Caddy accepts that too; only useful for non-browser clients).
"""
import argparse
import asyncio
import base64
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import soundfile as sf
import websockets

CHUNK_MS = 20  # simulate mic frames arriving every 20ms
SAMPLE_RATE = 16000


async def send_audio(ws, wav_path: str, realtime: bool):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples].astype(np.float32)
        await ws.send(json.dumps({
            "type": "audio_chunk",
            "audio": base64.b64encode(chunk.tobytes()).decode(),
        }))
        if realtime:
            await asyncio.sleep(CHUNK_MS / 1000)
    # A little trailing silence so VAD's min_silence_duration_ms has room to fire.
    silence_chunk = np.zeros(chunk_samples, dtype=np.float32)
    for _ in range(60):  # ~1.2s of silence
        await ws.send(json.dumps({
            "type": "audio_chunk",
            "audio": base64.b64encode(silence_chunk.tobytes()).decode(),
        }))
        if realtime:
            await asyncio.sleep(CHUNK_MS / 1000)
    await ws.send(json.dumps({"type": "end"}))


async def receive_loop(ws, out_dir: Path):
    t_turn_start = None
    sentence_idx = 0
    llm_text = []
    async for raw in ws:
        msg = json.loads(raw)
        t = msg["type"]
        if t == "speech_start":
            print("\n[speech_start]")
            t_turn_start = time.monotonic()
        elif t == "transcript":
            print(f"[transcript] {msg['text']}")
        elif t == "llm_delta":
            print(msg["text"], end="", flush=True)
            llm_text.append(msg["text"])
        elif t == "tts_chunk":
            sentence_idx += 1
            audio = np.frombuffer(base64.b64decode(msg["audio"]), dtype=np.float32)
            out_path = out_dir / f"sentence_{sentence_idx:03d}.wav"
            sf.write(out_path, audio, msg["sample_rate"])
            elapsed = time.monotonic() - t_turn_start if t_turn_start else float("nan")
            print(f"\n[tts_chunk #{sentence_idx}] '{msg['sentence'][:60]}' -> {out_path} "
                  f"(t+{elapsed:.2f}s since speech_start)")
        elif t == "turn_end":
            print(f"\n[turn_end] full response: {''.join(llm_text)!r}")
            break
        elif t == "error":
            print(f"\n[error] {msg['message']}")
            break


async def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query = {"voice": args.voice}
    extra_headers = {}
    if args.token:
        if args.auth_mode == "query":
            query["token"] = args.token
        else:
            extra_headers["Authorization"] = f"Bearer {args.token}"
    url = f"{args.gateway_url}?{urlencode(query)}"

    # Default 1MB WS frame limit is too small for base64-encoded float32 TTS
    # audio chunks (a several-second sentence alone can exceed it).
    async with websockets.connect(
        url, max_size=20 * 1024 * 1024, additional_headers=extra_headers or None,
    ) as ws:
        await asyncio.gather(
            send_audio(ws, args.wav, realtime=not args.no_realtime),
            receive_loop(ws, out_dir),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--gateway-url", default="ws://localhost:18010/ws/stream")
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--out-dir", default="./stream_test_output")
    p.add_argument("--no-realtime", action="store_true",
                    help="blast the whole file instantly instead of real-time-paced")
    p.add_argument("--token", default=None,
                    help="instance auth token ($OPEN_BUTTON_TOKEN / $WEB_PASSWORD), "
                         "required when --gateway-url points at the Caddy-authed "
                         "external port instead of localhost:18010")
    p.add_argument("--auth-mode", choices=["query", "header"], default="query",
                    help="how to send --token: as a ?token= query param (default, "
                         "matches what a browser WebSocket client can do) or as an "
                         "Authorization: Bearer header")
    asyncio.run(main(p.parse_args()))
