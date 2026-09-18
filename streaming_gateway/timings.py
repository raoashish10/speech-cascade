"""Per-turn latency breakdown for the voice pipeline.

Exists because every latency discussion about this project so far has been
conducted on inferred numbers. The TTS stage is measured (0.26s per sentence
with CHATTERBOX_BACKEND=vllm, 0.81s with pytorch, both on an RTX PRO 4500),
and the whole warm `voice_pipeline` round-trip is measured (~3.25s), but the
ASR and LLM halves of the remainder have only ever been estimated by
subtracting one from the other. That is not good enough to decide what to
optimise: it is entirely possible that ASR is 2s and the LLM is 0.4s, or the
reverse, and those point at completely different work.

WHAT TTFA MEANS HERE. Two different clocks matter and they differ by most of
a second:

  ttfa_from_vad_end   - from the moment VAD reports end-of-speech. This is
                        what the pipeline can actually influence.
  ttfa_from_speech_end - from the moment the user actually stopped talking.
                        This is what the user perceives, and it is larger by
                        UtteranceVAD's min_silence_duration_ms (800ms by
                        default), because that much trailing silence has to
                        elapse before end-of-speech can be declared at all.

Reporting only the first would flatter the system by ~800ms and hide the
single largest tunable in the stack. Reporting only the second would make
pipeline work look pointless. Both are emitted.

The marks are deliberately coarse -- one per pipeline stage boundary, not
per token. Fine-grained timing belongs in deploy/PROFILING.md's nsys/
torch.profiler flow; this is the always-on breakdown that answers "where did
this turn's latency go" for every turn in production.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Ordered pipeline stage boundaries. Each is recorded at most once per turn;
# a turn that ends early (empty transcript, error) simply has fewer.
TURN_START = "turn_start"            # VAD declared end-of-speech
ASR_DONE = "asr_done"                # transcript in hand
LLM_FIRST_DELTA = "llm_first_delta"  # LLM's first streamed token
FIRST_SENTENCE = "first_sentence"    # first complete sentence for TTS
FIRST_AUDIO = "first_audio"          # first audio chunk sent -> TTFA
TURN_END = "turn_end"


@dataclass
class TurnTimings:
    """Records stage boundaries for one turn and reports the breakdown.

    Uses time.monotonic() rather than time.time(): these are durations, and a
    clock adjustment mid-turn should not be able to produce a negative one.
    """

    vad_silence_ms: float = 0.0
    _marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        """Record a stage boundary. First write wins.

        First-write-wins matters for FIRST_SENTENCE and FIRST_AUDIO, which sit
        inside loops that fire once per sentence -- without it, "time to FIRST
        audio" would silently become "time to LAST audio", which is a very
        different and much larger number.
        """
        self._marks.setdefault(name, time.monotonic())

    def has(self, name: str) -> bool:
        return name in self._marks

    def _delta_ms(self, start: str, end: str) -> float | None:
        if start in self._marks and end in self._marks:
            return (self._marks[end] - self._marks[start]) * 1000.0
        return None

    def breakdown(self) -> dict[str, float]:
        """Stage durations in ms, omitting stages this turn never reached."""
        stages = {
            # Dead time before the pipeline is even allowed to start. Not a
            # pipeline cost, but it is on the user's clock and it is one
            # config value (UtteranceVAD.min_silence_duration_ms).
            "vad_silence_ms": self.vad_silence_ms,
            "asr_ms": self._delta_ms(TURN_START, ASR_DONE),
            # LLM time-to-first-token, the part that gates everything after.
            "llm_ttft_ms": self._delta_ms(ASR_DONE, LLM_FIRST_DELTA),
            # How long after the first token a full sentence exists. Driven by
            # SentenceAccumulator's boundary rules, not by model speed -- if
            # this is large, emitting on clause boundaries would help more
            # than a faster model.
            "llm_to_sentence_ms": self._delta_ms(LLM_FIRST_DELTA, FIRST_SENTENCE),
            # Synthesis of that first sentence only.
            "tts_first_ms": self._delta_ms(FIRST_SENTENCE, FIRST_AUDIO),
            # What the pipeline controls.
            "ttfa_from_vad_end_ms": self._delta_ms(TURN_START, FIRST_AUDIO),
            "turn_total_ms": self._delta_ms(TURN_START, TURN_END),
        }
        out = {k: round(v, 1) for k, v in stages.items() if v is not None}

        # What the user experiences: everything above plus the trailing
        # silence VAD had to observe before it could declare end-of-speech.
        ttfa = out.get("ttfa_from_vad_end_ms")
        if ttfa is not None:
            out["ttfa_from_speech_end_ms"] = round(ttfa + self.vad_silence_ms, 1)
        return out

    def summary(self) -> str:
        """One log line, ordered so the pipeline reads left to right."""
        b = self.breakdown()
        order = [
            "vad_silence_ms", "asr_ms", "llm_ttft_ms", "llm_to_sentence_ms",
            "tts_first_ms", "ttfa_from_vad_end_ms", "ttfa_from_speech_end_ms",
            "turn_total_ms",
        ]
        return " ".join(f"{k.removesuffix('_ms')}={b[k]:.0f}ms"
                        for k in order if k in b)
