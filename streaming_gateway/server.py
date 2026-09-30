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
credentials itself instead of trusting an upstream proxy to have done it --
same `?token=`/`Authorization: Bearer` convention as above, just enforced
here. See streaming_gateway/auth.py; in short:

  - GATEWAY_API_KEYS holds named, individually revocable keys (preferred).
    Only their SHA-256 is stored, so the environment never holds a usable
    credential, and sessions get an identity -- which is what makes the
    per-key session cap below possible at all.
  - GATEWAY_AUTH_TOKEN, the original single shared token, still works.
  - With neither set the process REFUSES TO START. It used to accept every
    connection in that case, so one missing environment variable published
    an open endpoint silently. Genuinely unauthenticated operation (correct
    only when something upstream authenticates) now has to be stated with
    GATEWAY_ALLOW_ANONYMOUS=1.
"""

import asyncio
import base64
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .auth import Authenticator
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

# Per-client cap, on top of the global one above. The global cap protects the
# GPU from the fleet as a whole but cannot stop ONE client taking every slot
# and starving the others -- which is only expressible once connections carry
# an identity, i.e. with API keys rather than one shared token. 0 disables it.
MAX_SESSIONS_PER_KEY = int(os.environ.get("GATEWAY_MAX_SESSIONS_PER_KEY", "2"))
_sessions_by_key: dict[str, int] = {}

# Raises at import if no credentials are configured and GATEWAY_ALLOW_ANONYMOUS
# is not set -- a missing env var should stop the process, not quietly publish
# an open endpoint. See streaming_gateway/auth.py.
_auth = Authenticator()


def _presented_credential(websocket: WebSocket) -> str | None:
    """Pull the credential from the Authorization header or ?token=.

    Header first: a credential in a query string ends up in proxy access logs,
    so ?token= is only supported because a browser's WebSocket API cannot set
    headers on the upgrade. Service-to-service clients should use the header.
    """
    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[len("bearer "):].strip()
    return websocket.query_params.get("token")


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    global _active_sessions

    principal = _auth.authenticate(_presented_credential(websocket))
    if principal is None:
        # Reject before accept() -- same as Caddy rejecting with 401 before
        # the upgrade completes, not a WS-level close after the fact.
        # Deliberately does not distinguish "unknown key" from "wrong secret";
        # that difference is only useful to someone guessing.
        log.warning("rejecting unauthorized connection from %s", websocket.client)
        await websocket.close(code=1008, reason="unauthorized")
        return

    await websocket.accept()

    async with _sessions_lock:
        if _active_sessions >= MAX_CONCURRENT_SESSIONS:
            log.warning(
                "rejecting session for %s: %d/%d concurrent sessions already active",
                principal, _active_sessions, MAX_CONCURRENT_SESSIONS,
            )
            # 1013 = "Try Again Later" (RFC 6455 registry); real close codes
            # so a client can tell "server full" apart from any other error.
            await websocket.close(code=1013, reason="gateway at capacity, try again shortly")
            return
        in_use = _sessions_by_key.get(principal.key_id, 0)
        if MAX_SESSIONS_PER_KEY and in_use >= MAX_SESSIONS_PER_KEY:
            log.warning(
                "rejecting session for %s: %d/%d sessions already held by this key",
                principal, in_use, MAX_SESSIONS_PER_KEY,
            )
            await websocket.close(
                code=1013, reason="per-key session limit reached, try again shortly"
            )
            return
        _active_sessions += 1
        _sessions_by_key[principal.key_id] = in_use + 1

    log.info("session opened for %s (%d/%d active)", principal,
             _active_sessions, MAX_CONCURRENT_SESSIONS)

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
            # Drop the key's entry at zero rather than leaving a 0 behind, so
            # this dict tracks live sessions and not every key ever seen.
            remaining = _sessions_by_key.get(principal.key_id, 1) - 1
            if remaining > 0:
                _sessions_by_key[principal.key_id] = remaining
            else:
                _sessions_by_key.pop(principal.key_id, None)
