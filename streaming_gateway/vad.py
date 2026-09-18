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
import os
import threading

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # 32ms @ 16kHz -- Silero VAD's native chunk size

# Trailing silence required before end-of-speech is declared.
#
# This is pure dead time on the user's clock: nothing in the pipeline may
# start until it elapses, so it lands on perceived time-to-first-audio in
# full. After clause chunking took TTS from 1686ms to 561ms (see
# streaming_gateway/README.md) it became the largest single remaining item in
# the budget, at 49% of the 1643ms a user waits -- larger than ASR, LLM
# prefill and the entire decode combined, several times over.
#
# 800 -> 400 buys 400ms directly and arithmetically; there is no measurement
# needed to know what it saves. THE COST IS THE PART THAT NEEDS MEASURING,
# and it is not measured: a speaker who pauses mid-sentence for longer than
# this gets endpointed early, and their turn is sent to ASR truncated. 400ms
# is a normal within-sentence pause for a hesitant speaker, someone
# thinking, or anyone saying a phone number or address with natural breaks.
# The synthetic clips this was measured against have no such pauses, so they
# cannot show the regression.
#
# Env-overridable for that reason: the right value is a property of the
# speakers, not of the code, and finding it should not need a rebuild. Raise
# it back toward 800 if turns start arriving truncated.
DEFAULT_MIN_SILENCE_MS = int(os.environ.get("GATEWAY_VAD_SILENCE_MS", "400"))

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

    def __init__(self, threshold: float = 0.5, min_silence_duration_ms: int | None = None):
        # Kept as an attribute because it is not just a VAD tuning knob: it is
        # dead time on the user's clock. End-of-speech cannot be declared
        # until this much trailing silence has elapsed, so every turn's
        # perceived time-to-first-audio includes it in full. The turn timing
        # breakdown reports it separately for exactly that reason -- see
        # streaming_gateway/timings.py.
        if min_silence_duration_ms is None:
            min_silence_duration_ms = DEFAULT_MIN_SILENCE_MS
        self.min_silence_duration_ms = min_silence_duration_ms
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
