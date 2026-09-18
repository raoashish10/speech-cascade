"""Unit tests for sub-sentence TTS chunking.

The measured breakdown (streaming_gateway/README.md) put whole-sentence TTS
at 82% of time-to-first-audio, because the pipeline waited for all of a
sentence's audio to exist before sending any. split_for_tts() cuts a sentence
at clause boundaries so the first piece comes back sooner.

The risk is entirely in WHERE it cuts. Each chunk is synthesised as an
independent utterance with its own intonation contour, and chunks are
synthesised sequentially so a gap can open between them. Both are acceptable
at a clause boundary -- that is a pause a speaker would also make -- and both
are damage anywhere else. So the tests below are mostly about refusing to
split rather than about splitting.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from streaming_gateway.sentence import (  # noqa: E402
    MIN_CHUNK_CHARS,
    MIN_SENTENCE_CHARS_TO_SPLIT,
    split_for_tts,
)

# The exact reply measured at 1812ms of synthesis for 6.00s of audio.
MEASURED = ("The sky appears blue because shorter wavelengths of light (blue) "
            "scatter more easily in Earth's atmosphere.")


def test_the_measured_sentence_splits_at_its_one_clause_boundary():
    """The case this feature was built for: 1812ms of synthesis becomes a
    ~1.1s first chunk, which the fitted model puts at ~450ms."""
    chunks = split_for_tts(MEASURED)
    assert chunks == [
        "The sky appears blue",
        "because shorter wavelengths of light (blue) scatter more easily in "
        "Earth's atmosphere.",
    ]


def test_chunks_reassemble_into_the_original_sentence():
    """Nothing may be dropped or duplicated -- this is spoken output."""
    for text in [MEASURED,
                 "It rains, then it stops, and later the sun comes out again.",
                 "We can go now, but only if everyone is ready to leave."]:
        assert " ".join(split_for_tts(text)) == " ".join(text.split())


def test_a_short_sentence_is_never_split():
    """Below the threshold there is little to save and a split costs an
    intonation reset for it."""
    assert split_for_tts("Paris.") == ["Paris."]
    assert split_for_tts("The capital of France is Paris.") == \
        ["The capital of France is Paris."]


def test_a_long_sentence_with_no_clause_boundary_is_passed_through_whole():
    """Refusing to split is the correct outcome, not a missed opportunity:
    the alternative is a mid-phrase cut, which is what this must never do."""
    text = "The quick brown fox jumped over the extremely lazy sleeping dog again"
    assert len(text) > MIN_SENTENCE_CHARS_TO_SPLIT
    assert split_for_tts(text) == [text]


def test_a_boundary_near_the_end_does_not_strand_a_fragment():
    """Splitting here would leave a two-word utterance to be synthesised on
    its own, which is where independent-intonation artefacts are worst."""
    text = "I looked everywhere in the house for the missing keys but failed"
    chunks = split_for_tts(text)
    assert all(len(c) >= MIN_CHUNK_CHARS for c in chunks), chunks


def test_a_boundary_too_early_does_not_produce_a_tiny_first_chunk():
    text = "So the answer to your question is that the meeting was moved to Thursday"
    chunks = split_for_tts(text)
    assert all(len(c) >= MIN_CHUNK_CHARS for c in chunks), chunks


def test_splits_on_clause_punctuation_after_the_comma():
    """The comma belongs to the clause it closes, not to the next one."""
    chunks = split_for_tts(
        "When the meeting finally ended, everyone went straight back to work.")
    assert len(chunks) == 2
    assert chunks[0].endswith(","), chunks
    assert chunks[1].startswith("everyone"), chunks


def test_a_multi_clause_sentence_splits_more_than_once():
    chunks = split_for_tts(
        "We finished the report on Monday, but the client asked for changes, "
        "so we spent all of Tuesday revising it.")
    assert len(chunks) >= 3, chunks
    assert all(len(c) >= MIN_CHUNK_CHARS for c in chunks), chunks


def test_a_clause_word_inside_a_word_is_not_a_boundary():
    """'so' must not match inside 'absolutely', 'but' inside 'butter'."""
    text = "The butter was absolutely essential to the recipe we were following"
    assert split_for_tts(text) == [text]


def test_whitespace_is_normalised_and_no_chunk_is_blank():
    chunks = split_for_tts(
        "  It was already late,   but we decided to keep walking anyway.  ")
    assert all(c == c.strip() and c for c in chunks), chunks


def test_every_chunk_is_a_prefix_boundary_of_the_original_order():
    """Chunks must come back in speaking order."""
    text = ("We finished the report on Monday, but the client asked for "
            "changes, so we spent all of Tuesday revising it.")
    pos = 0
    for chunk in split_for_tts(text):
        found = text.find(chunk.split()[0], pos)
        assert found >= pos, (chunk, pos)
        pos = found
