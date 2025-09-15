import sys


class Logger:
    def __init__(self):
        self.tracker = [] 

    def add_to_track(self, name, reference):
        self.tracker.append((name, reference))

    def print_all(self):
        for name, reference in self.tracker:
            size = sys.getsizeof(reference) / 1000000
            print(f"{name}: ~{size:.2f} MB")
            # for deep size use: asizeof.asizeof(reference) / (1024*1024)
