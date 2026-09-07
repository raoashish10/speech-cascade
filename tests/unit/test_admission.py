"""Unit tests for qwen_llm's AdmissionGate.

No live server, no GPU, no Triton needed -- AdmissionGate is pure counting
logic (see triton_model_repo/qwen_llm/1/admission.py's own docstring for
why it was split out of model.py specifically to make this possible).
Written to cover the exact failure this class exists to prevent: an
earlier version let concurrent requests queue unboundedly instead of being
rejected once a real limit was hit (measured regression: p50 7.4s at
concurrency 32 vs 1.75s at concurrency 1).
"""

import os
import sys
import threading

# admission.py lives alongside qwen_llm's model.py, not as an installed
# package -- add that directory to sys.path the same way model.py adds its
# own directory for sibling imports.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "triton_model_repo", "qwen_llm", "1"
    ),
)
from admission import AdmissionGate  # noqa: E402


def test_admits_up_to_the_limit():
    gate = AdmissionGate(max_admitted=3)
    assert gate.try_admit() is True
    assert gate.try_admit() is True
    assert gate.try_admit() is True
    assert gate.admitted == 3


def test_rejects_past_the_limit():
    gate = AdmissionGate(max_admitted=2)
    assert gate.try_admit() is True
    assert gate.try_admit() is True
    # The 3rd request is the one that used to queue unboundedly instead of
    # being told "no" immediately -- this is the actual regression case.
    assert gate.try_admit() is False
    assert gate.admitted == 2


def test_rejecting_does_not_increment_the_counter():
    gate = AdmissionGate(max_admitted=1)
    gate.try_admit()
    for _ in range(5):
        gate.try_admit()
    # Repeated rejections must not silently corrupt the count -- a bug here
    # would let it drift and eventually admit past the real limit, or never
    # admit again.
    assert gate.admitted == 1


def test_release_frees_a_slot_for_the_next_request():
    gate = AdmissionGate(max_admitted=1)
    assert gate.try_admit() is True
    assert gate.try_admit() is False  # full
    gate.release()
    assert gate.try_admit() is True  # room again


def test_release_after_a_failed_request_still_frees_the_slot():
    """Mirrors model.py's actual usage: release() is called from a `finally`
    block so a request that raises can't leak its slot and permanently
    wedge admission -- the exact bug class the module docstring warns about."""
    gate = AdmissionGate(max_admitted=1)
    gate.try_admit()
    try:
        raise RuntimeError("simulated request failure")
    except RuntimeError:
        pass
    finally:
        gate.release()
    assert gate.admitted == 0
    assert gate.try_admit() is True


def test_concurrent_try_admit_respects_the_limit():
    """The actual risk this class's lock guards against: without it, two
    threads could both read self._admitted below the limit and both
    increment, over-admitting past MAX_ADMITTED -- exactly the kind of race
    that would silently reintroduce the unbounded-queueing regression this
    class was written to fix. Fires many concurrent try_admit() calls from
    real threads (model.py's actual usage -- one call per ThreadPoolExecutor
    worker) and checks exactly max_admitted succeed, never more."""
    gate = AdmissionGate(max_admitted=10)
    results = []
    results_lock = threading.Lock()

    def worker():
        admitted = gate.try_admit()
        with results_lock:
            results.append(admitted)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 10
    assert results.count(False) == 90
