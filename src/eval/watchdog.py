import time


class Timeout(Exception):
    pass


def check_time(start: float, limit_s: float, every_n: int, i: int):
    """Raise Timeout when now-start > limit_s. Call inside loops every_n iterations."""
    if limit_s is None or limit_s <= 0:
        return
    if i % max(1, every_n) == 0:
        if time.time() - start > limit_s:
            raise Timeout(f"time limit {limit_s}s exceeded")



