"""Accumulates incrementally-arriving LLM text and yields complete sentences
as soon as a boundary is crossed, so TTS can start on sentence N while the
LLM is still generating sentence N+1.

kokoro_onnx.Kokoro.create() already does its own sentence/clause chunking,
splice-avoidance, and pause insertion for whatever complete text it's given
-- so feeding it one already-complete sentence at a time is a safe fit (its
internal chunking becomes a no-op most of the time, a safety net for
unusually long sentences the rest of the time). This module only needs to
find the boundaries, not do any of kokoro's own smoothing work.

Abbreviations ("Dr. Smith", "e.g.", single-letter initials like "J. Smith")
are special-cased below so the naive punctuation-based boundary doesn't
split on them -- see _is_abbreviation(). Not a general-purpose sentence
tokenizer (no ML, no exhaustive abbreviation list): just enough to cover
the realistic voice-assistant-answer cases this pipeline actually produces.
"""

import re

# Matches kokoro_onnx.chunker.SENTENCE_MARKS ('.!?…') -- duplicated as a
# plain string rather than importing kokoro_onnx here, since the gateway
# runs in its own isolated venv without kokoro-onnx installed.
SENTENCE_MARKS = ".!?…"
_SENTENCE_END_RE = re.compile(rf"(?<=[{re.escape(SENTENCE_MARKS)}])\s+")

# Fallback flush for a long run of text with no sentence punctuation at all
# (e.g. a comma-separated list), so TTS isn't stalled waiting forever.
MAX_ACCUM_CHARS = 400

# Common abbreviations a naive "punctuation + whitespace" boundary would
# otherwise mistake for a sentence end. Titles/Latin abbreviations/units
# that plausibly show up in a spoken, 1-2-sentence voice-assistant answer --
# not an attempt at an exhaustive list.
_ABBREVIATIONS = frozenset({
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "st.",
    "vs.", "e.g.", "i.e.", "etc.", "approx.", "inc.", "ltd.", "co.",
    "no.", "fig.", "vol.", "pp.",
})
_LAST_WORD_RE = re.compile(r"(\S+)$")


def _is_abbreviation(text_before_boundary: str) -> bool:
    """True if the word immediately before a candidate boundary (the text
    up to and including the punctuation mark that triggered the match) is a
    known abbreviation or a single-letter initial ("J." in "J. Smith") --
    i.e. this candidate boundary is a false positive, not a real sentence
    end."""
    m = _LAST_WORD_RE.search(text_before_boundary)
    if not m:
        return False
    word = m.group(1).lower()
    if word in _ABBREVIATIONS:
        return True
    # A lone letter + "." ("J.", "A.") -- real sentences essentially never
    # end on a single initial.
    return len(word) == 2 and word[1] == "." and word[0].isalpha()


class SentenceAccumulator:
    """Feed LLM text_diff pieces in; get back zero or more complete
    sentences ready to hand to kokoro_tts, as soon as each boundary is
    crossed."""

    def __init__(self):
        self._buf = ""

    def push(self, text_diff: str) -> list[str]:
        self._buf += text_diff
        out = []
        while True:
            boundary = None
            for m in _SENTENCE_END_RE.finditer(self._buf):
                if _is_abbreviation(self._buf[: m.start()]):
                    continue
                boundary = m
                break
            if boundary:
                sentence, self._buf = self._buf[: boundary.start()], self._buf[boundary.end() :]
                sentence = sentence.strip()
                if sentence:
                    out.append(sentence)
                continue
            if len(self._buf) >= MAX_ACCUM_CHARS:
                cut = self._buf.rfind(" ", 0, MAX_ACCUM_CHARS)
                cut = cut if cut > 0 else MAX_ACCUM_CHARS
                piece = self._buf[:cut].strip()
                self._buf = self._buf[cut:].lstrip()
                if piece:
                    out.append(piece)
                continue
            break
        return out

    def flush(self) -> str | None:
        """Call once the LLM stream is done; returns any trailing partial
        sentence still buffered."""
        rest = self._buf.strip()
        self._buf = ""
        return rest or None
