"""Accumulates incrementally-arriving LLM text and yields complete sentences
as soon as a boundary is crossed, so TTS can start on sentence N while the
LLM is still generating sentence N+1.

kokoro_onnx.Kokoro.create() used to do its own sentence/clause chunking,
splice-avoidance, and pause insertion for whatever complete text it was
given, making it a safe fit for one already-complete sentence at a time.
chatterbox_tts (ChatterboxTurboTTS.generate(), the current backend -- see
docs/tts-replacement-investigation.md) has no documented equivalent internal
chunking, and this project's own soak test only ever exercised short
LLM-generated turns (max_tokens=96, comfortably sub-sentence-cap length in
every one of 1039 turns) -- so a hard per-call length ceiling for this
backend, if one exists, is untested territory, not confirmed absent. This
module still only finds sentence boundaries, on the same unverified
assumption as before: that a single accumulated "sentence" never approaches
whatever that ceiling turns out to be.

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


# ---------------------------------------------------------------------------
# Sub-sentence chunking for TTS
# ---------------------------------------------------------------------------
#
# WHY. Measured on the two-pod deployment (see streaming_gateway/README.md):
# synthesising one 6.0s sentence costs 1812ms and is 82% of time-to-first-
# audio, because the pipeline waits for ALL of the audio to exist before
# sending any of it. Synthesis itself is not slow -- it fits
#
#     tts_ms  ~=  144ms  +  278ms per second of output audio
#
# a real-time factor of ~0.30 with only ~144ms of fixed cost. Handing TTS a
# shorter first piece therefore converts almost linearly into lower TTFA.
#
# WHERE WE SPLIT, AND WHY ONLY THERE. Only at genuine clause boundaries:
# after clause punctuation, or before a clause-introducing word. This is the
# whole safety argument for the feature, and it has two parts:
#
#   1. Each chunk is synthesised as an independent utterance, so it gets its
#      own sentence-final intonation contour. At a clause boundary that reads
#      as a speaker pausing mid-thought, which is how people actually talk.
#      Mid-phrase ("...shorter | wavelengths of light...") it reads as broken.
#   2. Chunks are synthesised sequentially, so a chunk whose synthesis takes
#      longer than the previous chunk's playback leaves a gap. At a clause
#      boundary a gap is a natural pause; mid-phrase it is a glitch. Splitting
#      only at clause boundaries means we never have to reason about chunk
#      size ratios to stay safe -- the worst case degrades to a pause a human
#      would also have made.
#
# So this never splits mid-phrase, and when a sentence offers no clause
# boundary big enough to be worth it, it is passed through whole -- exactly
# the behaviour that existed before. The feature can only add split points
# that a speaker would also have paused at.

# Estimated from the measured run: a 107-character reply produced 6.00s of
# audio. Used only to decide whether a sentence is long enough to be worth
# splitting and whether a piece is big enough to stand alone, so an error
# here changes chunk sizing, never correctness.
CHARS_PER_SECOND = 17.8

# Don't touch a sentence that already synthesises quickly: below this, the
# ~144ms fixed cost per chunk is a meaningful fraction of what there is to
# save, and a split buys little.
MIN_SENTENCE_SEC_TO_SPLIT = 2.5
MIN_SENTENCE_CHARS_TO_SPLIT = int(MIN_SENTENCE_SEC_TO_SPLIT * CHARS_PER_SECOND)

# A chunk shorter than this is not worth its own synthesis call (fixed cost,
# and a very short standalone utterance is where independent-intonation
# artefacts are most audible).
MIN_CHUNK_SEC = 1.0
MIN_CHUNK_CHARS = int(MIN_CHUNK_SEC * CHARS_PER_SECOND)

# Split AFTER these. Ordinary clause punctuation; sentence-final marks are
# already handled by SentenceAccumulator upstream.
_CLAUSE_PUNCT_RE = re.compile(r"[,;:]\s+|\s+[-–—]+\s+")

# Split BEFORE these. Words that introduce a new clause, i.e. places a
# speaker can draw breath without the result sounding cut off. English-only,
# and deliberately short: every entry here is a word that reliably starts a
# clause rather than merely appearing inside one.
_CLAUSE_WORDS = frozenset({
    "because", "but", "so", "which", "while", "although", "though",
    "since", "unless", "however", "therefore", "whereas", "yet",
})
_WORD_RE = re.compile(r"\b\w+\b")


def _clause_split_points(sentence: str) -> list[int]:
    """Character offsets where the sentence may be cut, in order."""
    points = set()
    for m in _CLAUSE_PUNCT_RE.finditer(sentence):
        points.add(m.end())
    for m in _WORD_RE.finditer(sentence):
        if m.group(0).lower() in _CLAUSE_WORDS and m.start() > 0:
            points.add(m.start())
    return sorted(points)


def split_for_tts(sentence: str) -> list[str]:
    """Split one sentence into clause-sized pieces for TTS.

    Returns [sentence] unchanged whenever splitting would not clearly help:
    a short sentence, or one with no clause boundary that leaves both sides
    big enough to stand alone. Pieces are returned in speaking order and
    concatenate back to the original text (whitespace normalised)."""
    sentence = sentence.strip()
    if len(sentence) < MIN_SENTENCE_CHARS_TO_SPLIT:
        return [sentence]

    chunks = []
    start = 0
    for point in _clause_split_points(sentence):
        piece = sentence[start:point].strip()
        remainder = sentence[point:].strip()
        # Both sides must be worth synthesising on their own. Checking the
        # remainder too is what stops a boundary near the end from stranding
        # a two-word fragment as its own utterance.
        if len(piece) >= MIN_CHUNK_CHARS and len(remainder) >= MIN_CHUNK_CHARS:
            chunks.append(piece)
            start = point
    tail = sentence[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks or [sentence]


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
