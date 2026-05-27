import time

# Đo thời gian cho từng bước pipeline
class Timer:
    def __init__(self):
        self.records = {}
        self._t0 = None
        self._label = None

    def start(self, label: str):
        self._t0 = time.time()
        self._label = label
        return self

    def stop(self):
        if self._t0 is None:
            return 0.0
        elapsed = time.time() - self._t0
        self.records[self._label] = round(elapsed, 3)
        self._t0 = None
        return elapsed

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def report(self):
        return dict(self.records)
