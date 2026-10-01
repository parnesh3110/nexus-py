"""Camera capture on its own thread. Only the newest frame is kept, so processing never
works through a backlog of stale frames (that backlog is what makes webcam UIs feel laggy)."""
import threading
import time

import cv2

from .log import log


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._cap = None
        self._cond = threading.Condition()
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self._running = False
        self._thread = None
        self.dropped = 0  # frames replaced before anyone read them

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def start(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't queue frames inside OpenCV either
        if not cap.isOpened():
            raise RuntimeError("Camera didn't open. Allow camera access for your terminal in "
                               "System Settings -> Privacy & Security -> Camera, then run again.")
        self._cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        log.info("camera", "started", index=self.index, size=f"{self.width}x{self.height}",
                 fps=cap.get(cv2.CAP_PROP_FPS), backend=cap.getBackendName())
        self._running = True
        self._thread = threading.Thread(target=self._run, name="nexus-camera", daemon=True)
        self._thread.start()

    def _run(self):
        failures = 0
        while self._running:
            ok, frame = self._cap.read()
            t = time.perf_counter()
            if not ok:
                failures += 1
                if failures in (1, 30) or failures % 300 == 0:
                    log.warn("camera", "read_failed", count=failures)
                time.sleep(0.01)
                continue
            with self._cond:
                if self._frame is not None and self._seq_read != self._seq:
                    self.dropped += 1
                self._frame, self._t = frame, t
                self._seq += 1
                self._cond.notify_all()

    _seq_read = 0

    def latest(self, after_seq: int, timeout: float = 1.0):
        """Newest frame newer than `after_seq`, as (seq, frame_bgr, capture_time), or None."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after_seq or not self._running, timeout):
                return None
            self._seq_read = self._seq
            return self._seq, self._frame, self._t

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=1)
        if self._cap:
            self._cap.release()
        log.info("camera", "stopped", dropped=self.dropped)
