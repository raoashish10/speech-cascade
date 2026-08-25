"""Unit tests for streaming_gateway.sentence.SentenceAccumulator.

No live server needed -- SentenceAccumulator is pure string logic. These
formalize the standalone spot-checks done by hand during the streaming
gateway session into a real, repeatable suite.
"""

import pytest

from streaming_gateway.sentence import MAX_ACCUM_CHARS, SentenceAccumulator


def test_single_push_single_sentence():
    acc = SentenceAccumulator()
    # Trailing space after the terminator is what completes the boundary
    # match (`(?<=[.!?...])\s+`) -- see
    # test_boundary_without_trailing_whitespace_waits_for_flush below for
    # the no-trailing-space case.
    assert acc.push("Hello there. ") == ["Hello there."]
    assert acc.flush() is None


def test_multiple_sentences_in_one_push():
    acc = SentenceAccumulator()
    out = acc.push("First one. Second one! Third one? ")
    assert out == ["First one.", "Second one!", "Third one?"]
    assert acc.flush() is None


def test_boundary_without_trailing_whitespace_waits_for_flush():
    """A sentence-ending punctuation mark with nothing after it yet isn't a
    confirmed boundary (the regex requires trailing whitespace, since more
    LLM text -- or an abbreviation like "Mr." -- could still be coming) --
    it stays buffered until either more text or flush() resolves it."""
    acc = SentenceAccumulator()
    assert acc.push("Hello there.") == []
    assert acc.flush() == "Hello there."


def test_piecewise_arrival_splits_at_boundary():
    """LLM tokens arrive as small diffs -- a boundary crossed mid-push must
    be detected the moment the punctuation + following whitespace lands,
    not before."""
    acc = SentenceAccumulator()
    pieces = ["Hel", "lo there", ". How are", " you", "?", " I'm", " fine."]
    out = []
    for p in pieces:
        out.extend(acc.push(p))
    remainder = acc.flush()
    if remainder:
        out.append(remainder)
    assert out == ["Hello there.", "How are you?", "I'm fine."]


def test_flush_returns_trailing_partial_sentence():
    acc = SentenceAccumulator()
    out = acc.push("Complete sentence. Trailing partial")
    assert out == ["Complete sentence."]
    assert acc.flush() == "Trailing partial"
    # buffer is cleared after flush
    assert acc.flush() is None


def test_flush_on_empty_buffer_returns_none():
    acc = SentenceAccumulator()
    assert acc.flush() is None


def test_no_punctuation_at_all_stays_buffered_until_flush():
    acc = SentenceAccumulator()
    text = "no punctuation in this whole utterance at all"
    assert acc.push(text) == []
    assert acc.flush() == text


def test_fallback_flush_fires_on_long_punctuation_free_run():
    """A comma-separated list or otherwise punctuation-free run past
    MAX_ACCUM_CHARS must force a flush at the last space boundary, so TTS
    isn't stalled waiting forever for a sentence terminator that never
    arrives."""
    acc = SentenceAccumulator()
    # Build a long run of space-separated words with no sentence punctuation.
    words = [f"word{i}" for i in range(200)]
    long_text = " ".join(words)
    assert len(long_text) > MAX_ACCUM_CHARS

    out = acc.push(long_text)
    assert len(out) >= 1, "fallback flush should have fired at least once"
    # Every emitted piece must be at or under the cutoff length.
    for piece in out:
        assert len(piece) <= MAX_ACCUM_CHARS
    # Whatever remains is retrievable via flush(), and no words were lost or
    # duplicated across the emitted pieces + the final flush.
    remainder = acc.flush()
    reconstructed = " ".join(out + ([remainder] if remainder else []))
    assert reconstructed.split() == long_text.split()


def test_fallback_cuts_at_word_boundary_not_mid_word():
    acc = SentenceAccumulator()
    long_text = " ".join(f"token{i}" for i in range(100))
    out = acc.push(long_text)
    for piece in out:
        assert not piece.endswith(" ")
        # every piece is made of whole tokens from the source text
        for tok in piece.split():
            assert tok in long_text.split()


def test_exact_reconstruction_across_many_piecewise_pushes():
    """Feed a realistic multi-sentence LLM stream one character at a time
    and confirm the emitted sentences + final flush reconstruct the
    original text exactly (mod the boundary whitespace the accumulator
    itself consumes as a separator)."""
    source = (
        "The quick brown fox jumps over the lazy dog. "
        "It was a bright cold day in April, and the clocks were striking "
        "thirteen! Is this real life, or is this just fantasy? "
        "Trailing thought with no terminator"
    )
    acc = SentenceAccumulator()
    out = []
    for ch in source:
        out.extend(acc.push(ch))
    remainder = acc.flush()
    if remainder:
        out.append(remainder)

    # Rejoin with single spaces (the accumulator strips the separating
    # whitespace at each boundary) and compare against the source rejoined
    # the same way, so the check is exact rather than approximate.
    assert " ".join(out) == " ".join(source.split(" "))


def test_empty_push_is_a_noop():
    acc = SentenceAccumulator()
    assert acc.push("") == []
    assert acc.flush() is None


@pytest.mark.parametrize("mark", [".", "!", "?", "…"])
def test_each_sentence_mark_is_a_boundary(mark):
    acc = SentenceAccumulator()
    out = acc.push(f"Sentence one{mark} Sentence two.")
    assert out[0] == f"Sentence one{mark}"


@pytest.mark.parametrize(
    "abbrev",
    ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "St.", "vs.",
     "e.g.", "i.e.", "etc.", "approx.", "Inc.", "Ltd.", "Co.", "No.",
     "Fig.", "vol.", "pp."],
)
def test_abbreviation_does_not_split(abbrev):
    acc = SentenceAccumulator()
    out = acc.push(f"See {abbrev} Smith for details. That's everything. ")
    assert out == [f"See {abbrev} Smith for details.", "That's everything."]


def test_single_letter_initial_does_not_split():
    acc = SentenceAccumulator()
    out = acc.push("Ask J. Smith about it. Thanks. ")
    assert out == ["Ask J. Smith about it.", "Thanks."]


def test_abbreviation_at_end_of_utterance_resolves_via_flush():
    """An abbreviation with nothing recognizable as a real sentence end
    after it should still surface via flush() rather than being lost."""
    acc = SentenceAccumulator()
    assert acc.push("Please see Dr. Smith") == []
    assert acc.flush() == "Please see Dr. Smith"


def test_abbreviation_piecewise_arrival():
    """The abbreviation check must still work when 'Dr.' and the following
    word arrive in separate pushes, matching how LLM tokens actually
    stream in."""
    acc = SentenceAccumulator()
    pieces = ["See Dr", ".", " Smith", " now", ". Done", "."]
    out = []
    for p in pieces:
        out.extend(acc.push(p))
    remainder = acc.flush()
    if remainder:
        out.append(remainder)
    assert out == ["See Dr. Smith now.", "Done."]
