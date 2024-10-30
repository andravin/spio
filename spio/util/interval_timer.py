"""Classes for timing code execution and measuring intervals."""

from time import perf_counter
from .logger import log_level

TIMER_OVERHEAD = 3e-6  # 3 microseconds


class IntervalTimer:
    """A class for measuring intervals of time."""

    def __init__(self, skip=0):
        """Initialize the timer and start it.

        Args:
            skip (int): Number of initial records to skip in the timer statistics.
        """
        self.total = 0
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.start_time = None
        self.skip = skip
        self.start()

    def start(self):
        """Set the start time to now."""
        self.start_time = perf_counter()

    def elapsed(self) -> float:
        """Return the elapsed time since the timer was last started."""
        return perf_counter() - self.start_time

    def record(self):
        """Record the time elapsed since the timer was last started.

        Updates the timer statistics, skipping the first `skip` records.
        """
        self.count += 1
        if self.count > self.skip:
            dt = self.elapsed()
            self.total += dt
            self.min = min(dt, self.min)
            self.max = max(dt, self.max)

    def average(self) -> float:
        """Return the average time, excluding skipped records."""
        n = self.count - self.skip
        return self.total / n if n > 0 else 0

    def reset(self):
        """Reset the timer statistics."""
        self.total = 0
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.start()

    def report(self, message=""):
        """Print a report of the timer statistics."""
        print(
            f"{message:40s}: avg: {self.average()*1e6:8.1f}us "
            f"min: {self.min*1e6:8.1f}us max: {self.max*1e6:8.1f}us"
        )


class Timer:
    """A context manager for timing code blocks."""

    def __init__(self, message: str = "", timer_log_level: int = 0):
        """Initialize the timer.

        Args:
            message (str, optional): A message to display when the timer starts. Defaults to "".

        Example:
            with Timer("Compiling kernel"):
                kernel.compile()

        Output:
            Compiling kernel in progress .. finished in 0.123 seconds.
        """
        self.message = message
        self.log_level = timer_log_level
        self.start = None
        self.elapsed = None
        self.end = None

    def __enter__(self):
        self.start = perf_counter()
        if log_level >= self.log_level:
            print(f"{self.message} in progress ..", end="")
        return self

    def __exit__(self, *args):
        self.end = perf_counter()
        self.elapsed = self.end - self.start
        if log_level >= self.log_level:
            print(f" finished in {self.elapsed:.6f} seconds.")


def time_function(message="", timer_log_level=0):
    """A decorator for timing function calls.

    Args:
        message (str, optional): A message to display when the timer starts. Defaults to "".
        timer_log_level (int, optional): The log level at which to display the message. Default 0.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with Timer(message=message, timer_log_level=timer_log_level):
                return func(*args, **kwargs)

        return wrapper

    return decorator
