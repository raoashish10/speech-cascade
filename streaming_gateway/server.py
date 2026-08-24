"""FastAPI WebSocket gateway fronting the Triton voice pipeline. Ingests raw
mic audio, runs VAD-driven utterance segmentation, and streams back
transcript / LLM text / TTS audio chunks as they're produced.

Run with: uvicorn streaming_gateway.server:app --host 127.0.0.1 --port 18010
"""

import base64
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .session import StreamingSession

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    voice = websocket.query_params.get("voice", "af_heart")

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
