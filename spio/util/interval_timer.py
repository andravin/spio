from time import perf_counter

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
