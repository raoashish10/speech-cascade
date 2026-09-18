"""Streams a wav file's audio in real-time-simulated chunks to the streaming
gateway over WebSocket, printing/saving LLM text and TTS audio as they arrive.

Usage (local, gateway's internal port -- no auth needed):
    python3 test_streaming_client.py --wav test_tts_output.wav

Usage (from outside the box, through the Caddy-authed external port -- see
streaming_gateway/README.md for the full external-access writeup):
    python3 test_streaming_client.py --wav clip.wav \\
        --gateway-url ws://<PUBLIC_IPADDR>:<VAST_TCP_PORT_10100>/ws/stream \\
        --token "$OPEN_BUTTON_TOKEN" --voice Sofia

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
import statistics
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import soundfile as sf
import websockets

CHUNK_MS = 20  # simulate mic frames arriving every 20ms
SAMPLE_RATE = 16000


async def send_audio(ws, wav_path: str, realtime: bool, send_end: bool = True):
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
    if send_end:
        await ws.send(json.dumps({"type": "end"}))


async def receive_loop(ws, out_dir: Path, turn: int = 1):
    """Consume one turn's messages. Returns the turn's `timings` breakdown
    (see streaming_gateway/timings.py) or None if the turn failed."""
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
            out_path = out_dir / f"turn_{turn:03d}_sentence_{sentence_idx:03d}.wav"
            sf.write(out_path, audio, msg["sample_rate"])
            elapsed = time.monotonic() - t_turn_start if t_turn_start else float("nan")
            print(f"\n[tts_chunk #{sentence_idx}] '{msg['sentence'][:60]}' -> {out_path} "
                  f"(t+{elapsed:.2f}s since speech_start)")
        elif t == "turn_end":
            print(f"\n[turn_end] full response: {''.join(llm_text)!r}")
            timings = msg.get("timings") or {}
            if timings:
                print("[timings] " + " ".join(f"{k}={v}" for k, v in timings.items()))
            return timings
        elif t == "error":
            print(f"\n[error] {msg['message']}")
            return None
    return None


# The metrics worth summarising across turns, in pipeline order. Anything the
# gateway reports but that isn't listed here still lands in --timings-json.
SUMMARY_KEYS = [
    "vad_silence_ms", "asr_ms", "llm_ttft_ms",
    "llm_deltas", "llm_decode_ms",
    "llm_tbt_mean_ms", "llm_tbt_p50_ms", "llm_tbt_p95_ms", "llm_tbt_max_ms",
    "llm_to_sentence_ms", "tts_first_ms",
    "ttfa_from_vad_end_ms", "ttfa_from_speech_end_ms", "turn_total_ms",
]


def print_summary(runs: list[dict]):
    """Per-metric distribution across turns.

    min/median/max rather than a single mean: with a handful of turns a mean
    is dominated by whichever one happened to contend, and the point of this
    run is to find out which STAGE dominates, which the spread answers and an
    average does not."""
    if not runs:
        print("\nNo successful turns -- nothing to summarise.")
        return
    print(f"\n{'=' * 72}\n{len(runs)} successful turn(s)\n{'=' * 72}")
    print(f"{'metric':<24}{'min':>10}{'median':>10}{'max':>10}{'n':>6}")
    for key in SUMMARY_KEYS:
        vals = [r[key] for r in runs if key in r]
        if not vals:
            continue
        print(f"{key:<24}{min(vals):>10.1f}{statistics.median(vals):>10.1f}"
              f"{max(vals):>10.1f}{len(vals):>6}")

    # The question this whole run exists to answer. Every latency argument so
    # far has been made on a split obtained by subtracting one measured
    # number from another; these two are now both measured directly.
    asr = [r["asr_ms"] for r in runs if "asr_ms" in r]
    ttft = [r["llm_ttft_ms"] for r in runs if "llm_ttft_ms" in r]
    if asr and ttft:
        a, t = statistics.median(asr), statistics.median(ttft)
        print(f"\nASR vs LLM prefill (medians): asr={a:.0f}ms  llm_ttft={t:.0f}ms"
              f"  ->  ASR is {a / t:.1f}x the LLM's time-to-first-token"
              if t else "")


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
    runs = []
    async with websockets.connect(
        url, max_size=20 * 1024 * 1024, additional_headers=extra_headers or None,
    ) as ws:
        for turn in range(1, args.turns + 1):
            if args.turns > 1:
                print(f"\n----- turn {turn}/{args.turns} -----")
            # "end" only on the last turn: the server treats it as "no more
            # mic audio" and closes the session after draining, so sending it
            # earlier would cost a reconnect per turn (and re-pay the
            # per-connection setup this run is trying to measure around).
            _, timings = await asyncio.gather(
                send_audio(ws, args.wav, realtime=not args.no_realtime,
                           send_end=(turn == args.turns)),
                receive_loop(ws, out_dir, turn=turn),
            )
            if timings:
                runs.append(timings)

    if args.turns > 1 or args.timings_json:
        print_summary(runs)
    if args.timings_json:
        Path(args.timings_json).write_text(json.dumps(runs, indent=2))
        print(f"\nPer-turn timings written to {args.timings_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--gateway-url", default="ws://localhost:18010/ws/stream")
    p.add_argument("--voice", default="Sofia")
    p.add_argument("--out-dir", default="./stream_test_output")
    p.add_argument("--turns", type=int, default=1,
                    help="replay the wav this many times over ONE connection, then "
                         "print a per-metric distribution. One turn tells you almost "
                         "nothing: the first turn after a cold model load is not "
                         "representative, and stage times vary turn to turn.")
    p.add_argument("--timings-json", default=None,
                    help="write the per-turn timings breakdowns to this file as JSON")
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
