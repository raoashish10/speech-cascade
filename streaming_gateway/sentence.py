"""Accumulates incrementally-arriving LLM text and yields complete sentences
as soon as a boundary is crossed, so TTS can start on sentence N while the
LLM is still generating sentence N+1.

kokoro_onnx.Kokoro.create() already does its own sentence/clause chunking,
splice-avoidance, and pause insertion for whatever complete text it's given
-- so feeding it one already-complete sentence at a time is a safe fit (its
internal chunking becomes a no-op most of the time, a safety net for
unusually long sentences the rest of the time). This module only needs to
find the boundaries, not do any of kokoro's own smoothing work.

Known limitation, not fixed here: the naive punctuation-based boundary
regex will prematurely split on abbreviations ("Dr. Smith", "e.g."). Fine
for the "core pattern" scope this was built for.
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
            m = _SENTENCE_END_RE.search(self._buf)
            if m:
                sentence, self._buf = self._buf[: m.start()], self._buf[m.end() :]
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
