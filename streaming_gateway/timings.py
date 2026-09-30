"""Per-turn latency breakdown for the voice pipeline.

Exists because every latency discussion about this project was conducted on
inferred numbers -- the ASR and LLM halves of the budget had only ever been
estimated by subtracting one measured figure from another.

It has since been run against the two-pod deployment, and the inference was
wrong in a way worth recording here so nobody re-derives it: ASR is 88ms and
LLM prefill is 18ms, together under 5% of time-to-first-audio, while TTS is
1812ms, or 82%. Time between deltas is 9.7ms with an 11.6ms worst case over
30 turns. streaming_gateway/README.md has the full table and what it implies
about what to optimise; the short version is that everything except TTS (and
the VAD silence window) is inside the noise.

WHAT TTFA MEANS HERE. Two different clocks matter and they differ by most of
a second:

  ttfa_from_vad_end   - from the moment VAD reports end-of-speech. This is
                        what the pipeline can actually influence.
  ttfa_from_speech_end - from the moment the user actually stopped talking.
                        This is what the user perceives, and it is larger by
                        UtteranceVAD's min_silence_duration_ms (400ms by
                        default, env GATEWAY_VAD_SILENCE_MS), because that
                        much trailing silence has to elapse before
                        end-of-speech can be declared at all.

Reporting only the first would flatter the system by ~800ms and hide the
single largest tunable in the stack. Reporting only the second would make
pipeline work look pointless. Both are emitted.

Stage marks are deliberately coarse -- one per pipeline stage boundary. The
one exception is the LLM's output stream, which is timed per delta, because
time-between-tokens is a separate user-visible property from TTFT and neither
predicts the other: a fast first token followed by a stall still sounds
broken. See record_llm_delta() for what a "delta" is and is not.

Finer-grained timing than this belongs in deploy/PROFILING.md's nsys/
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


def _percentile(ordered: list[float], pct: float) -> float:
    """Nearest-rank percentile over an already-sorted list.

    Nearest-rank rather than interpolating: these samples are counted in the
    tens per turn, and an interpolated p95 of 12 samples invents a value that
    never occurred. Every number reported here is one that actually happened.
    """
    if not ordered:
        raise ValueError("percentile of an empty sample")
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil, integer-only
    return ordered[int(rank) - 1]


@dataclass
class TurnTimings:
    """Records stage boundaries for one turn and reports the breakdown.

    Uses time.monotonic() rather than time.time(): these are durations, and a
    clock adjustment mid-turn should not be able to produce a negative one.
    """

    vad_silence_ms: float = 0.0
    _marks: dict[str, float] = field(default_factory=dict)
    _llm_deltas: list[float] = field(default_factory=list)

    def record_llm_delta(self) -> None:
        """Record the arrival of one streamed LLM delta, for TBT.

        WHAT A DELTA IS. qwen_llm sends one InferenceResponse per TensorRT-LLM
        streaming step, carrying that step's `text_diff` -- but only `if diff`,
        so steps whose token adds no decodable text (byte-level BPE
        continuation bytes, for instance) are dropped before they reach the
        wire. A delta is therefore *at least* one token and occasionally more,
        which makes these figures a slight over-estimate of true
        time-between-tokens. `llm_deltas` is reported alongside so the gap
        between deltas and tokens stays visible rather than assumed away.

        Also marks LLM_FIRST_DELTA, so a caller cannot record the stream
        without also establishing TTFT.
        """
        self.mark(LLM_FIRST_DELTA)
        self._llm_deltas.append(time.monotonic())

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

    def _tbt_gaps_ms(self) -> list[float]:
        """Inter-delta gaps. N deltas give N-1 gaps; the wait for the FIRST
        delta is TTFT and is deliberately not one of them -- folding it in
        would let a slow prefill masquerade as slow decode."""
        d = self._llm_deltas
        return [(b - a) * 1000.0 for a, b in zip(d, d[1:])]

    def _llm_stream_stats(self) -> dict[str, float]:
        gaps = self._tbt_gaps_ms()
        if not gaps:
            # One delta (or none): a count is still worth reporting, but there
            # is no interval to characterise.
            return {"llm_deltas": float(len(self._llm_deltas))} if self._llm_deltas else {}
        ordered = sorted(gaps)
        decode_ms = (self._llm_deltas[-1] - self._llm_deltas[0]) * 1000.0
        return {
            "llm_deltas": float(len(self._llm_deltas)),
            # Decode wall time, first delta to last -- the span the gaps
            # partition. TTFT is excluded by construction.
            "llm_decode_ms": decode_ms,
            "llm_tbt_mean_ms": sum(gaps) / len(gaps),
            "llm_tbt_p50_ms": _percentile(ordered, 50),
            # p95 and max are the ones that matter for perceived smoothness:
            # a mean of 30ms hides a single 900ms stall, and the stall is what
            # a listener hears. Reported separately for exactly that reason.
            "llm_tbt_p95_ms": _percentile(ordered, 95),
            "llm_tbt_max_ms": ordered[-1],
        }

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
        stages.update(self._llm_stream_stats())
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
            "vad_silence_ms", "asr_ms", "llm_ttft_ms",
            "llm_tbt_p50_ms", "llm_tbt_p95_ms", "llm_tbt_max_ms",
            "llm_to_sentence_ms",
            "tts_first_ms", "ttfa_from_vad_end_ms", "ttfa_from_speech_end_ms",
            "turn_total_ms",
        ]
        parts = [f"{k.removesuffix('_ms')}={b[k]:.0f}ms" for k in order if k in b]
        if "llm_deltas" in b:
            # Not a duration, so it does not get the ms formatting -- but the
            # TBT figures are uninterpretable without knowing how many
            # intervals they were computed over.
            parts.insert(3, f"llm_deltas={b['llm_deltas']:.0f}")
        return " ".join(parts)
