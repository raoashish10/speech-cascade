"""FastAPI WebSocket gateway fronting the Triton voice pipeline. Ingests raw
mic audio, runs VAD-driven utterance segmentation, and streams back
transcript / LLM text / TTS audio chunks as they're produced.

Run with: uvicorn streaming_gateway.server:app --host 127.0.0.1 --port 18010

External access
----------------
This process only binds 127.0.0.1:18010 -- it is never reachable directly.
On the Vast.ai instance it's put behind the Caddy auth edge (see
`/etc/portal.yaml`'s "Streaming Gateway" entry, external_port 10100), which
terminates TLS/auth and forwards WebSocket upgrades to this port. From
outside the box the endpoint is:

    ws://<PUBLIC_IPADDR>:<VAST_TCP_PORT_10100>/ws/stream?token=<instance token>

(`ws://` unless the instance has `ENABLE_HTTPS=true`, in which case `wss://`).
The token is the instance's `$OPEN_BUTTON_TOKEN` / `$WEB_PASSWORD` -- Caddy
checks it on every request, WebSocket upgrades included, and rejects with
HTTP 401 before the upgrade completes if it's missing or wrong. Pass it as
the `?token=` query param (the only option a browser's native WebSocket API
supports, since it can't set custom headers) or, for non-browser clients,
as an `Authorization: Bearer <token>` header -- Caddy accepts either. See
streaming_gateway/README.md for a full worked example and
scripts/test_streaming_client.py for a client that speaks both.

This is a single shared instance token, not a per-user credential, so
anyone holding it can open a session -- see MAX_CONCURRENT_SESSIONS below
for the (deliberately modest) protection against that.

Triton's own HTTP/gRPC/metrics ports stay localhost-only and are not
exposed externally; this gateway is the sanctioned external surface, and
proxying raw Triton access would just widen the attack surface without
adding a capability this endpoint doesn't already provide end-to-end.

Docker deployment (docker/, docker-compose.yml)
-------------------------------------------------
There's no Caddy edge in that deployment shape, so this process checks
GATEWAY_AUTH_TOKEN itself (see _check_auth below) instead of trusting an
upstream proxy to have done it -- same `?token=`/`Authorization: Bearer`
convention as above, just enforced here rather than by Caddy. Leave
GATEWAY_AUTH_TOKEN unset to skip the check (e.g. bare-metal-behind-Caddy,
or your own reverse proxy already handles it).
"""

import asyncio
import base64
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .session import StreamingSession

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

# The gateway is now reachable by anyone holding the instance's shared
# auth token (Caddy authenticates the connection, not the "user" -- there
# isn't a per-user identity here). The concrete risk on this box isn't
# abuse of the gateway process itself, it's every session queuing work
# onto the *same* GPU-resident Triton models (whisper_asr / qwen_llm /
# chatterbox_tts) with only ~5.5GB VRAM headroom at steady state (that
# figure was measured against kokoro_tts; not yet re-measured against
# chatterbox_tts's own ~3.0-3.4GB footprint) -- a handful
# of concurrent utterances easily starves that. A hard concurrency cap is
# the cheapest guard against that specific failure mode, so that's what's
# here; see the module docstring above for the reachability side of this,
# and streaming_gateway/README.md for the full "what we didn't add" list.
MAX_CONCURRENT_SESSIONS = int(os.environ.get("GATEWAY_MAX_SESSIONS", "4"))
_active_sessions = 0
_sessions_lock = asyncio.Lock()

# Shared-secret check, only meaningful when nothing upstream (Vast.ai's
# Caddy edge, or your own reverse proxy) already does it -- see the module
# docstring's "Docker deployment" section. Empty/unset disables the check
# entirely, preserving today's bare-metal-behind-Caddy behavior.
GATEWAY_AUTH_TOKEN = os.environ.get("GATEWAY_AUTH_TOKEN", "")


def _check_auth(websocket: WebSocket) -> bool:
    if not GATEWAY_AUTH_TOKEN:
        return True
    token = websocket.query_params.get("token")
    if not token:
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[len("bearer "):]
    return token == GATEWAY_AUTH_TOKEN


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    global _active_sessions

    if not _check_auth(websocket):
        # Reject before accept() -- same as Caddy rejecting with 401 before
        # the upgrade completes, not a WS-level close after the fact.
        await websocket.close(code=1008, reason="unauthorized")
        return

    await websocket.accept()

    async with _sessions_lock:
        if _active_sessions >= MAX_CONCURRENT_SESSIONS:
            log.warning(
                "rejecting session: %d/%d concurrent sessions already active",
                _active_sessions, MAX_CONCURRENT_SESSIONS,
            )
            # 1013 = "Try Again Later" (RFC 6455 registry); real close codes
            # so a client can tell "server full" apart from any other error.
            await websocket.close(code=1013, reason="gateway at capacity, try again shortly")
            return
        _active_sessions += 1

    try:
        voice = websocket.query_params.get("voice", "Sofia")

        async def send_json(payload: dict):
            await websocket.send_json(payload)

        session = StreamingSession(send_json, voice=voice)
        graceful_end = False
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "audio_chunk":
                    await session.handle_audio_chunk(base64.b64decode(msg["audio"]))
                elif msg.get("type") == "end":
                    graceful_end = True
                    break
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("streaming session error")
        finally:
            if session.turn_task and not session.turn_task.done():
                if graceful_end:
                    # Client said "end" (no more mic audio coming), not "hang up
                    # on me" -- let any in-flight ASR/LLM/TTS turn finish and
                    # keep streaming its responses before the connection closes.
                    try:
                        await session.turn_task
                    except Exception:
                        log.exception("turn task failed while draining on graceful end")
                else:
                    session.turn_task.cancel()
    finally:
        async with _sessions_lock:
            _active_sessions -= 1
