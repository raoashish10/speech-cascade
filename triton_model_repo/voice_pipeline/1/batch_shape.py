"""Pure array-reshaping helper, split out of model.py so it's unit-testable
without triton_python_backend_utils (which only exists inside a running
Triton python-backend process) -- see tests/unit/test_batch_shape.py.

Exists because of the one non-obvious wrinkle in this whole model (see
README.md's own section on it): the three models voice_pipeline calls all
declare max_batch_size > 0 (an implicit leading batch dimension), but
voice_pipeline itself is unbatched (max_batch_size: 0). Every tensor
forwarded to a callee needs this leading batch dim of 1 added before the
call -- Triton doesn't do it automatically for BLS-constructed requests.
The asymmetric part (not this module's concern, since there's no code to
run on the way back): tensors coming *back* from a BLS call do NOT carry
that batch dimension, so nothing here should ever be applied to a response.
Getting either direction backwards produces the exact confusing failures
README.md describes -- "batch size does not match other inputs" if this is
skipped on the way in, silently wrong data (e.g. a whole audio array
collapsed to its first sample) if an equivalent strip is mistakenly applied
on the way out.
"""

import numpy as np


def add_batch_dim(arr: np.ndarray) -> np.ndarray:
    """Add a leading batch dim of 1, e.g. shape (N,) -> (1, N)."""
    return arr.reshape(1, *arr.shape)
