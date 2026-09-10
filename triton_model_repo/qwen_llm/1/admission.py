"""Pure admission-control counter, split out of model.py so it's unit-
testable without triton_python_backend_utils (which only exists inside a
running Triton python-backend process, not in a plain pytest environment)
-- see tests/unit/test_admission.py.

Exists because an earlier version let ThreadPoolExecutor.submit() queue
unboundedly: a burst of concurrent requests piled up invisibly (measured:
p50 7.4s at concurrency 32 vs 1.75s at concurrency 1, pure queueing). This
bounds in-flight-or-queued work and rejects immediately past the limit
instead of queueing -- see model.py's own module docstring for the full
story and MAX_ADMITTED's derivation.
"""

import threading


class AdmissionGate:
    """Thread-safe counter bounding concurrent admitted work to `max_admitted`.

    try_admit() returns True and increments the count if there's room, False
    (without incrementing) if not. release() must be called exactly once for
    every True returned by try_admit(), regardless of whether the admitted
    work succeeded or raised -- model.py calls it from a `finally` block so a
    failure can't leak a slot and permanently wedge admission.
    """

    def __init__(self, max_admitted: int):
        self.max_admitted = max_admitted
        self._admitted = 0
        self._lock = threading.Lock()

    def try_admit(self) -> bool:
        with self._lock:
            if self._admitted >= self.max_admitted:
                return False
            self._admitted += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._admitted -= 1

    @property
    def admitted(self) -> int:
        with self._lock:
            return self._admitted
