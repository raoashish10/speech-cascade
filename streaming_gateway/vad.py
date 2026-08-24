"""Silero VAD wrapper for utterance-boundary detection.

Ported from LocalVox's voice_agent/vad.py (BargeInVAD), stripped to the core
speech-start/speech-end detection this project needs -- no barge-in, no
partial-transcript-specific tuning. One UtteranceVAD instance per WebSocket
session.

The frontend streams raw float32 PCM frames (16kHz mono) continuously over
the WS while a session is open. process_bytes() internally rebuffers into
Silero's native 512-sample/32ms chunk size and runs the model on each frame,
returning "start" the instant speech-onset is confirmed, or "end" after
min_silence_duration_ms of trailing silence.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # 32ms @ 16kHz -- Silero VAD's native chunk size

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from silero_vad import load_silero_vad
                log.info("Loading Silero VAD (onnx)...")
                _model = load_silero_vad(onnx=True)
                log.info("Silero VAD ready")
    return _model


class UtteranceVAD:
    """Streaming speech-start/speech-end detector for one connection."""

    def __init__(self, threshold: float = 0.5, min_silence_duration_ms: int = 800):
        from silero_vad import VADIterator
        self._iterator = VADIterator(
            _get_model(),
            threshold=threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=min_silence_duration_ms,
        )
        self._buffer = np.zeros(0, dtype=np.float32)

    def process_bytes(self, pcm_f32_bytes: bytes) -> str | None:
        """Feed raw float32 PCM bytes; internally chunked to FRAME_SAMPLES.
        Returns 'start', 'end', or None. If more than one frame's worth of
        audio arrives at once, only the most significant event is returned
        (start takes priority over end within the same call)."""
        chunk = np.frombuffer(pcm_f32_bytes, dtype=np.float32)
        self._buffer = np.concatenate([self._buffer, chunk])

        event = None
        while len(self._buffer) >= FRAME_SAMPLES:
            frame, self._buffer = self._buffer[:FRAME_SAMPLES], self._buffer[FRAME_SAMPLES:]
            result = self._iterator(frame)
            if result is not None:
                if "start" in result:
                    event = "start"
                elif "end" in result and event != "start":
                    event = "end"
        return event

    def reset(self):
        self._iterator.reset_states()
        self._buffer = np.zeros(0, dtype=np.float32)
