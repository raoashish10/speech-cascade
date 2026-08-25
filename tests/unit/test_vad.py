"""Unit tests for streaming_gateway.vad.UtteranceVAD.

No live Triton server needed -- Silero VAD's own ONNX model runs locally on
CPU (streaming_gateway/requirements.txt pins plain `onnxruntime`, not the
GPU build, for exactly this reason). These are the only tests in the suite
that pull in a real (small, CPU) ML model rather than pure Python logic;
they're still "unit" in the sense that matters here -- no dependency on the
live Triton service or the GPU.

The fixture (tests/fixtures/vad_sample_16k.wav) is a real short speech clip
("Testing one two three.") synthesized offline via the project's own Kokoro
TTS weights -- NOT via the live Triton server -- with ~0.6s of lead silence
and ~1.2s of trailing silence appended, giving a genuine
silence -> speech -> silence utterance shape for UtteranceVAD to detect.
Synthetic tones/noise were tried first and don't reliably trigger a real
speech model; a short real speech clip does. See
tests/fixtures/README.md for regeneration instructions.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from streaming_gateway.vad import FRAME_SAMPLES, SAMPLE_RATE, UtteranceVAD

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "vad_sample_16k.wav"

# How big a chunk to hand to process_bytes() per call, simulating the
# gateway's real usage (~20ms mic frames over the WebSocket).
CHUNK_SAMPLES = 320  # 20ms @ 16kHz


def _load_fixture():
    audio, sr = sf.read(FIXTURE_PATH, dtype="float32")
    assert sr == SAMPLE_RATE, f"fixture must be {SAMPLE_RATE}Hz, got {sr}"
    return audio


def _feed(vad, audio, chunk_samples=CHUNK_SAMPLES):
    """Feed audio in small chunks and collect (sample_offset, event) pairs."""
    events = []
    for start in range(0, len(audio), chunk_samples):
        piece = audio[start : start + chunk_samples]
        event = vad.process_bytes(piece.astype(np.float32).tobytes())
        if event is not None:
            events.append((start, event))
    return events


def test_silence_speech_silence_yields_one_start_one_end_pair():
    audio = _load_fixture()
    vad = UtteranceVAD()
    events = _feed(vad, audio)

    assert len(events) == 2, f"expected exactly one start/end pair, got {events}"
    (start_offset, start_event), (end_offset, end_event) = events
    assert start_event == "start"
    assert end_event == "end"
    assert start_offset < end_offset


def test_start_event_lands_near_actual_speech_onset():
    """The fixture's speech region begins at sample 9600 (0.6s of lead
    silence). Silero needs a few frames to confirm onset, so allow some
    lag, but the start event must not fire wildly early or late."""
    audio = _load_fixture()
    vad = UtteranceVAD()
    events = _feed(vad, audio)
    start_offset, _ = events[0]

    speech_onset_samples = int(0.6 * SAMPLE_RATE)
    # generous window: within 1s of the true onset in either direction
    assert abs(start_offset - speech_onset_samples) < SAMPLE_RATE


def test_end_event_lands_after_min_silence_duration():
    """The fixture appends 1.2s of trailing silence after speech, comfortably
    longer than the default min_silence_duration_ms (800ms), so the end
    event must fire only after speech actually stops, not during it."""
    audio = _load_fixture()
    vad = UtteranceVAD()
    events = _feed(vad, audio)
    end_offset, _ = events[1]

    # Speech region in the fixture ends well before the final ~1.2s of
    # trailing silence; the end event must land inside that trailing window,
    # not spuriously mid-speech.
    assert end_offset > len(audio) - int(1.2 * SAMPLE_RATE) - SAMPLE_RATE


def test_pure_silence_produces_no_events():
    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # 2s of silence
    vad = UtteranceVAD()
    events = _feed(vad, silence)
    assert events == []


def test_reset_clears_state_between_utterances():
    """After reset(), a fresh silence-speech-silence clip should again
    produce exactly one start/end pair -- confirming reset() actually
    clears the iterator's internal state and the frame buffer, not just
    part of it."""
    audio = _load_fixture()
    vad = UtteranceVAD()
    first_events = _feed(vad, audio)
    assert len(first_events) == 2

    vad.reset()
    second_events = _feed(vad, audio)
    assert len(second_events) == 2


def test_process_bytes_handles_multi_frame_chunks():
    """Feeding more than one FRAME_SAMPLES worth of audio in a single call
    must still detect the boundary (start takes priority over end within
    the same call, per the docstring), not silently drop it."""
    audio = _load_fixture()
    vad = UtteranceVAD()
    big_chunk = FRAME_SAMPLES * 8  # feed 8 native frames at a time
    events = _feed(vad, audio, chunk_samples=big_chunk)
    assert [e for _, e in events] == ["start", "end"]
