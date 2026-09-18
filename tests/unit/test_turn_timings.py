"""Unit tests for the per-turn latency breakdown.

No server, no GPU -- TurnTimings is deliberately just arithmetic over
monotonic marks so the thing we intend to make optimisation decisions from is
itself testable.

The cases below are the ones that would quietly corrupt the numbers:

  - FIRST_AUDIO and FIRST_SENTENCE are marked inside loops that run once per
    sentence. If a later mark overwrote an earlier one, "time to first audio"
    would silently become "time to last audio" -- a much larger number that
    still looks plausible in a log line.
  - a turn that fails or ends early must still report what it reached, since
    how far a failing turn got is most of diagnosing it.
  - ttfa_from_speech_end must include the VAD silence window. Omitting it
    under-reports perceived latency by 800ms, which is the largest single
    tunable in the stack.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from streaming_gateway.timings import (  # noqa: E402
    ASR_DONE,
    FIRST_AUDIO,
    FIRST_SENTENCE,
    LLM_FIRST_DELTA,
    TURN_END,
    TURN_START,
    TurnTimings,
)


def _timings_at(vad_silence_ms=800.0, clock=None):
    """A TurnTimings whose marks come from a scripted clock, in seconds."""
    t = TurnTimings(vad_silence_ms=vad_silence_ms)
    if clock:
        for name, when in clock:
            t._marks[name] = when
    return t


FULL_TURN = [
    (TURN_START, 100.00),
    (ASR_DONE, 100.40),        # 400ms ASR
    (LLM_FIRST_DELTA, 100.65),  # 250ms LLM TTFT
    (FIRST_SENTENCE, 101.05),   # 400ms to a complete sentence
    (FIRST_AUDIO, 101.31),      # 260ms TTS
    (TURN_END, 102.50),
]


def test_stage_durations():
    b = _timings_at(clock=FULL_TURN).breakdown()
    assert b["asr_ms"] == 400.0
    assert b["llm_ttft_ms"] == 250.0
    assert b["llm_to_sentence_ms"] == 400.0
    assert b["tts_first_ms"] == 260.0
    assert b["turn_total_ms"] == 2500.0


def test_the_two_ttfa_clocks_differ_by_the_vad_window():
    """The pipeline's number and the user's number are not the same."""
    b = _timings_at(vad_silence_ms=800.0, clock=FULL_TURN).breakdown()
    assert b["ttfa_from_vad_end_ms"] == 1310.0
    assert b["ttfa_from_speech_end_ms"] == 2110.0
    assert (b["ttfa_from_speech_end_ms"] - b["ttfa_from_vad_end_ms"]) == 800.0


def test_vad_window_is_reported_even_though_it_is_not_pipeline_time():
    b = _timings_at(vad_silence_ms=800.0, clock=FULL_TURN).breakdown()
    assert b["vad_silence_ms"] == 800.0


def test_lowering_the_vad_window_moves_perceived_ttfa_one_for_one():
    """The cheapest lever in the stack, and this is what it buys."""
    slow = _timings_at(vad_silence_ms=800.0, clock=FULL_TURN).breakdown()
    fast = _timings_at(vad_silence_ms=400.0, clock=FULL_TURN).breakdown()
    assert slow["ttfa_from_vad_end_ms"] == fast["ttfa_from_vad_end_ms"]
    assert slow["ttfa_from_speech_end_ms"] - fast["ttfa_from_speech_end_ms"] == 400.0


def test_first_mark_wins_so_ttfa_is_not_silently_time_to_last_audio():
    t = TurnTimings(vad_silence_ms=0.0)
    t._marks[TURN_START] = 100.0
    t.mark(FIRST_AUDIO)
    first = t._marks[FIRST_AUDIO]
    t.mark(FIRST_AUDIO)  # second sentence's audio
    t.mark(FIRST_AUDIO)  # third
    assert t._marks[FIRST_AUDIO] == first


def test_a_turn_that_ends_early_still_reports_what_it_reached():
    """Empty transcript: ASR ran, nothing else did."""
    b = _timings_at(clock=[(TURN_START, 100.0), (ASR_DONE, 100.4)]).breakdown()
    assert b["asr_ms"] == 400.0
    assert "llm_ttft_ms" not in b
    assert "ttfa_from_vad_end_ms" not in b
    assert "ttfa_from_speech_end_ms" not in b


def test_a_failed_turn_reports_how_far_it_got():
    b = _timings_at(clock=[
        (TURN_START, 100.0), (ASR_DONE, 100.4), (LLM_FIRST_DELTA, 100.65),
        (TURN_END, 100.9),
    ]).breakdown()
    assert b["asr_ms"] == 400.0
    assert b["llm_ttft_ms"] == 250.0
    assert b["turn_total_ms"] == 900.0
    assert "tts_first_ms" not in b


def test_summary_reads_in_pipeline_order():
    s = _timings_at(clock=FULL_TURN).summary()
    for earlier, later in [
        ("vad_silence", "asr"), ("asr", "llm_ttft"),
        ("llm_ttft", "llm_to_sentence"), ("llm_to_sentence", "tts_first"),
        ("tts_first", "ttfa_from_vad_end"),
    ]:
        assert s.index(earlier + "=") < s.index(later + "="), s


def test_summary_omits_stages_a_turn_never_reached():
    s = _timings_at(clock=[(TURN_START, 100.0), (ASR_DONE, 100.4)]).summary()
    assert "asr=400ms" in s
    assert "tts_first" not in s


def test_marks_use_a_monotonic_clock():
    """Durations must not be able to go negative on a wall-clock adjustment."""
    t = TurnTimings()
    t.mark(TURN_START)
    t.mark(TURN_END)
    assert t.breakdown()["turn_total_ms"] >= 0.0
