"""Unit tests for voice_pipeline's add_batch_dim().

No live server, no GPU, no Triton needed -- this is pure numpy reshaping
(see triton_model_repo/voice_pipeline/1/batch_shape.py's own docstring for
why it was split out of model.py specifically to make this possible).

Written to cover the exact class of bug README.md warns about by name: the
three models voice_pipeline calls all expect an implicit leading batch
dimension, but voice_pipeline's own inputs don't have one -- getting this
reshape wrong produces either "batch size does not match other inputs"
(dim missing) or silently wrong data, e.g. a whole audio array collapsed
to its first sample (an extra dim stripped that was never meant to be
there). These tests check the shape and, critically, that no data is lost
or reordered by the reshape -- only that a dimension was added.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "triton_model_repo", "voice_pipeline", "1"
    ),
)
from batch_shape import add_batch_dim  # noqa: E402


def test_adds_a_leading_dim_of_one():
    audio = np.zeros(16000, dtype=np.float32)  # 1s of 16kHz audio, voice_pipeline's real shape
    out = add_batch_dim(audio)
    assert out.shape == (1, 16000)


def test_preserves_every_sample_in_order():
    # A whole audio array silently collapsing to its first sample (README's
    # own example of what goes wrong here) would still pass a shape-only
    # check -- this confirms the actual values survive the reshape intact.
    audio = np.arange(10, dtype=np.float32)
    out = add_batch_dim(audio)
    assert np.array_equal(out[0], audio)


def test_works_on_a_scalar_shaped_array():
    # SAMPLE_RATE is sent as a 1-element array, not a full audio buffer --
    # confirms this doesn't assume audio-sized input specifically.
    sample_rate = np.array([24000], dtype=np.int32)
    out = add_batch_dim(sample_rate)
    assert out.shape == (1, 1)
    assert out[0, 0] == 24000


def test_does_not_mutate_the_input_array():
    # reshape() can return a view -- confirm callers don't get a surprise if
    # they still hold a reference to the original array afterward.
    original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    copy_before = original.copy()
    add_batch_dim(original)
    assert np.array_equal(original, copy_before)
