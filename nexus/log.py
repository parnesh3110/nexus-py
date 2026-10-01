"""JSON-lines logging: one file per run in ./logs, info and above also printed.

Rule: log state transitions and failures, not per-frame values. Per-frame data goes to the
CSV recorder (--record); a perf summary is logged every 5 seconds.
Set NEXUS_LOG=debug to print debug events too.
"""
import json
import math
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

_LEVELS = {"debug": 0, "info": 1, "warn": 2, "error": 3}


def _clean(v):
    """Make a value JSON-safe and compact."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return round(v, 3) if math.isfinite(v) else None
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    try:  # numpy scalars
        return _clean(v.item())
    except Exception:
        return str(v)


class Logger:
    def __init__(self):
        self.console_level = _LEVELS["debug"] if os.environ.get("NEXUS_LOG") == "debug" else _LEVELS["info"]
        self.path: Path | None = None
        self._file = None
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="nexus-log", daemon=True)
        self._thread.start()

    def open(self, directory: str | None = None) -> Path:
        if directory is None:
            from . import paths
            directory = str(paths.LOGS)
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / f"nexus-{datetime.now():%Y%m%d-%H%M%S}.log"
        self._file = open(self.path, "a", buffering=1)
        return self.path

    def debug(self, cat, event, **f): self._q.put((time.time(), "debug", cat, event, f))
    def info(self, cat, event, **f): self._q.put((time.time(), "info", cat, event, f))
    def warn(self, cat, event, **f): self._q.put((time.time(), "warn", cat, event, f))
    def error(self, cat, event, **f): self._q.put((time.time(), "error", cat, event, f))

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            ts, level, cat, event, fields = item
            fields = {k: _clean(v) for k, v in fields.items()}
            if self._file:
                obj = {"ts": datetime.fromtimestamp(ts).isoformat(timespec="milliseconds"),
                       "lvl": level, "cat": cat, "event": event, **fields}
                self._file.write(json.dumps(obj) + "\n")
            if _LEVELS[level] >= self.console_level:
                extras = " ".join(f"{k}={v}" for k, v in sorted(fields.items()))
                clock = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
                print(f"{clock} {level.upper()} {cat}.{event} {extras}", flush=True)

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=2)
        if self._file:
            self._file.close()


log = Logger()
