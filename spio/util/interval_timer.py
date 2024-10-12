from time import perf_counter
from .logger import log_level

TIMER_OVERHEAD = 3e-6  # 3 microseconds


class IntervalTimer:
    def __init__(self, skip=0):
        self.reset()
        self.skip = skip

    def start(self):
        self.start_time = perf_counter()

    def elapsed(self):
        return perf_counter() - self.start_time

    def record(self):
        self.count += 1
        if self.count > self.skip:
            dt = self.elapsed()
            self.total += dt
            if dt < self.min:
                self.min = dt
            if dt > self.max:
                self.max = dt

    def average(self):
        n = self.count - self.skip
        return self.total / n if n > 0 else 0

    def reset(self):
        self.total = 0
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")
        self.start()

    def report(self, message=""):
        print(
            f"{message:40s}: avg: {self.average()*1e6:8.1f}us min: {self.min*1e6:8.1f}us max: {self.max*1e6:8.1f}us"
        )


class Timer:
    """A context manager for timing code blocks.

    Args:
        message (str, optional): A message to display when the timer starts. Defaults to "".

    Example:
        with Timer("Compiling kernel"):
            kernel.compile()

    Output:
        Compiling kernel in progress .. finished in 0.123 seconds.
    """

    def __init__(self, message="", log_level=0):
        self.message = message
        self.log_level = log_level

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


def time_function(message="", log_level=0):
    """A decorator for timing function calls.
    
    Args:
        message (str, optional): A message to display when the timer starts. Defaults to "".
        log_level (int, optional): The log level at which to display the message. Defaults to 0.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with Timer(message=message, log_level=log_level):
                return func(*args, **kwargs)
        return wrapper
    return decorator
