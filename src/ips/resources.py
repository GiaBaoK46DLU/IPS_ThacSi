"""RAM toàn tiến trình lấy mẫu, VRAM allocator PyTorch, đồng bộ thời gian CUDA."""
import threading
import time

import psutil
import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class MemorySampler:
    def __init__(self, interval_seconds=0.01):
        self.interval = interval_seconds
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.samples = 0
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        self.thread = None

    def sample(self):
        value = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, value)
        self.samples += 1

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.sample()
        info = self.process.memory_info()
        return {"scope": "whole_process_RSS_including_Python_native_libraries_and_data",
                "sampling_interval_seconds": self.interval, "samples": self.samples,
                "rss_start_bytes": self.start_rss, "rss_peak_sampled_bytes": self.peak_rss,
                "rss_end_bytes": info.rss,
                "windows_peak_working_set_since_process_start_bytes": getattr(info, "peak_wset", None),
                "note": "sampled peak may miss brief spikes; not model-only allocation"}


def cuda_memory(device):
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None, "reason": "CPU run; no CUDA measurement"}
    synchronize(device)
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
