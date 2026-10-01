"""Performance summary every 5 s (logged) and the per-frame CSV recorder (--record).

The CSV also stores all 21 landmarks (image + 3D world), so every recording doubles as
training data for the PyTorch gesture model.
"""
import csv
import math
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import psutil

from .landmarks import flatten


def _pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int((len(s) - 1) * p / 100))]


class Perf:
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)  # first call primes the counter
        self._start = None
        self._reset()

    def _reset(self):
        self.frames = self.hand_frames = self.locked = self.frozen = self.edge = 0
        self.pinch_downs = self.losses = 0
        self.latency, self.vision = [], []

    def add(self, latency_ms, vision_ms, state, on_edge):
        self.frames += 1
        self.latency.append(latency_ms)
        if vision_ms is not None:
            self.vision.append(vision_ms)
        if state.visible:
            self.hand_frames += 1
        self.locked += state.locked
        self.frozen += state.holding_click
        self.edge += on_edge and state.visible
        if state.event == "down":
            self.pinch_downs += 1

    def flush_if_due(self, t, render_fps, camera_dropped):
        if self._start is None:
            self._start = t
            return None
        if t - self._start < self.interval or self.frames == 0:
            return None
        hf = max(self.hand_frames, 1)
        out = dict(
            fps=self.frames / (t - self._start),
            render_fps=render_fps,
            hand_pct=round(100 * self.hand_frames / self.frames),
            latency_p50_ms=_pct(self.latency, 50), latency_p95_ms=_pct(self.latency, 95),
            vision_p50_ms=_pct(self.vision, 50), vision_p95_ms=_pct(self.vision, 95),
            locked_pct=round(100 * self.locked / hf), frozen_pct=round(100 * self.frozen / hf),
            edge_pct=round(100 * self.edge / hf), pinch_downs=self.pinch_downs,
            camera_dropped=camera_dropped,
            cpu_pct=self._proc.cpu_percent(None), rss_mb=self._proc.memory_info().rss / 1_048_576,
        )
        self._start = t
        self._reset()
        return out


class Recorder:
    HEADER = (["t", "hand", "cam_x", "cam_y", "raw_x", "raw_y", "x", "y", "ratio", "pinched", "event",
               "vision_ms", "latency_ms", "sensitivity"]
              + [f"lm{i}_{a}" for i in range(21) for a in "xyz"]
              + [f"w{i}_{a}" for i in range(21) for a in "xyz"])

    def __init__(self, note: str, directory: str = "runs"):
        Path(directory).mkdir(exist_ok=True)
        self.path = Path(directory) / f"run-{datetime.now():%Y%m%d-%H%M%S}-{note}.csv"
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)

    def write(self, s, hand, vision_ms, latency_ms, sensitivity):
        r = lambda v, n=2: "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(v, n)
        self._w.writerow([round(s.t, 4), int(s.visible), r(s.cam_x, 4), r(s.cam_y, 4), r(s.raw_x), r(s.raw_y),
                          r(s.x), r(s.y), r(s.ratio, 3), int(s.pinched), s.event or "", r(vision_ms), r(latency_ms),
                          sensitivity] + [r(v, 4) for v in flatten(hand)])

    def close(self):
        self._f.close()


def system_info() -> dict:
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception:
            return ""
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        git = ""
    return dict(os=platform.mac_ver()[0], chip=sysctl("machdep.cpu.brand_string"), python=platform.python_version(),
                ram_gb=psutil.virtual_memory().total / 1_073_741_824, git=git or "uncommitted")
