import time


class CancelToken:
    __slots__ = ("deadline",)

    def __init__(self, timeout_s: float | None):
        self.deadline = (time.time() + float(timeout_s)) if (timeout_s and timeout_s > 0) else None

    def expired(self) -> bool:
        return self.deadline is not None and time.time() > self.deadline

    def remaining(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())



