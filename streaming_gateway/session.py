"""Per-connection state machine: buffers incoming mic audio, uses VAD to
find utterance boundaries, and on each finalized utterance runs
ASR -> streamed LLM -> per-sentence TTS, forwarding partial results out as
they arrive.

Ported from LocalVox's voice_agent/server.py handle_mic_frame/
_finalize_barge_in, stripped to the core pattern only: pre-roll buffer,
VAD start/end, minimum-speech-duration confirmation gate, finalize-on-
silence-or-timeout. No barge-in/playback-interruption, no partial rolling
transcripts, no speaker verification -- see LocalVox for those if this
project grows into them later. The seam to add barge-in is noted at
self.turn_task below.
"""

import asyncio
import logging
import time

import numpy as np

from . import triton_client as tc
from .sentence import SentenceAccumulator
from .timings import (
    ASR_DONE,
    FIRST_AUDIO,
    FIRST_SENTENCE,
    LLM_FIRST_DELTA,
    TURN_END,
    TURN_START,
    TurnTimings,
)
from .vad import SAMPLE_RATE, UtteranceVAD

log = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 4  # float32

PREROLL_MS = 400
PREROLL_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * PREROLL_MS / 1000)

MIN_SPEECH_MS = 350

# LocalVox uses different caps for barge-in (8s) vs. dictation (15s); this
# project has neither distinction, so one sane cap covers both cases.
MAX_UTTERANCE_SEC = 15.0


class StreamingSession:
    def __init__(self, send_json, voice: str = "Sofia"):
        self._send_json = send_json
        self.voice = voice
        self.vad = UtteranceVAD()

        self.preroll = bytearray()
        self.speech_buf: bytearray | None = None
        self.buffering = False
        self.speech_confirmed = False
        self.utterance_started_at = 0.0

        # Seam to extend for barge-in later: on a new VAD "start" while
        # turn_task is running, cancel it and abort the in-flight
        # stream_infer call. Not built in this pass.
        self.turn_task: asyncio.Task | None = None

    async def handle_audio_chunk(self, pcm_bytes: bytes):
        event = self.vad.process_bytes(pcm_bytes)

        if event == "start" and not self.buffering:
            self.buffering = True
            self.speech_confirmed = False
            # Seed with pre-roll FIRST (audio strictly before this frame) --
            # Silero confirms onset a few frames late; without this the
            # first word's attack gets clipped.
            self.speech_buf = bytearray(self.preroll)
            self.utterance_started_at = time.monotonic()

        self.preroll.extend(pcm_bytes)
        if len(self.preroll) > PREROLL_BYTES:
            del self.preroll[: -PREROLL_BYTES]

        if not self.buffering:
            return
        self.speech_buf.extend(pcm_bytes)

        if not self.speech_confirmed:
            speech_ms = len(self.speech_buf) / (SAMPLE_RATE * BYTES_PER_SAMPLE) * 1000
            if speech_ms >= MIN_SPEECH_MS:
                self.speech_confirmed = True
                await self._send_json({"type": "speech_start"})

        timed_out = (time.monotonic() - self.utterance_started_at) > MAX_UTTERANCE_SEC
        if event == "end" or timed_out:
            pcm = bytes(self.speech_buf)
            self.buffering = False
            self.speech_buf = None
            if self.speech_confirmed:
                # Backgrounded so the WS receive loop stays free to buffer
                # the NEXT utterance's mic audio while this one's
                # ASR/LLM/TTS runs.
                self.turn_task = asyncio.create_task(self._run_turn(pcm))

    async def _run_turn(self, pcm_bytes: bytes):
        # Started here, not inside the try, so the clock covers everything
        # this turn does -- including a failure.
        timings = TurnTimings(vad_silence_ms=self.vad.min_silence_duration_ms)
        timings.mark(TURN_START)
        try:
            audio = np.frombuffer(pcm_bytes, dtype=np.float32)
            transcript = await tc.transcribe(audio, sample_rate=SAMPLE_RATE)
            timings.mark(ASR_DONE)
            if not transcript.strip():
                return
            await self._send_json({"type": "transcript", "text": transcript})

            accumulator = SentenceAccumulator()
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

            async def tts_worker():
                while True:
                    sentence = await tts_queue.get()
                    if sentence is None:
                        break
                    audio_out, sr = await tc.synthesize(sentence, voice=self.voice)
                    # First write wins inside this loop, so this stays
                    # time-to-FIRST-audio rather than time-to-last.
                    timings.mark(FIRST_AUDIO)
                    await self._send_json({
                        "type": "tts_chunk",
                        "sentence": sentence,
                        "audio": _b64(audio_out),
                        "sample_rate": sr,
                    })

            worker = asyncio.create_task(tts_worker())
            async for delta in tc.generate_stream(transcript):
                timings.mark(LLM_FIRST_DELTA)
                await self._send_json({"type": "llm_delta", "text": delta})
                for sentence in accumulator.push(delta):
                    timings.mark(FIRST_SENTENCE)
                    await tts_queue.put(sentence)
            trailing = accumulator.flush()
            if trailing:
                # A short reply with no sentence punctuation reaches TTS only
                # here, via the flush -- mark it so those turns still report a
                # sentence boundary instead of a hole in the breakdown.
                timings.mark(FIRST_SENTENCE)
                await tts_queue.put(trailing)
            await tts_queue.put(None)
            await worker
            timings.mark(TURN_END)
            log.info("turn timings: %s", timings.summary())
            await self._send_json({"type": "turn_end", "timings": timings.breakdown()})
        except Exception as e:
            timings.mark(TURN_END)
            # Log the partial breakdown too: knowing how far a failing turn
            # got, and how long it took to get there, is most of diagnosing it.
            log.warning("turn failed after %s: %s", timings.summary(), e)
            await self._send_json({"type": "error", "message": str(e)})


def _b64(audio: np.ndarray) -> str:
    import base64
    return base64.b64encode(audio.astype(np.float32).tobytes()).decode("ascii")
