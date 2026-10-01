"""Signal filters. Pure Python, no dependencies, unit-tested."""
import math


class _LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x: float, alpha: float) -> float:
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """One Euro filter (Casiez et al., CHI 2012): a low-pass whose cutoff rises with speed.
    Heavy smoothing when the hand is almost still (kills jitter), light smoothing when it
    moves fast (kills lag)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t = None

    def __call__(self, value: float, t: float) -> float:
        if self._t is None or self._x.y is None:
            self._t = t
            return self._x(value, 1.0)
        dt = t - self._t
        if dt <= 0:
            return self._x.y
        self._t = t
        speed = self._dx((value - self._x.y) / dt, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(speed)
        return self._x(value, _alpha(cutoff, dt))


class PinchDetector:
    """Thumb-index pinch, judged relative to your own hand a moment ago, not only against a fixed
    number.

    In:  the gap is tiny (< instant), or it's fairly small (< enter) AND has shrunk to under `drop`
         of the widest it was in the last half second, for 2 frames. Fingers going from open to
         touching is a click even if your touch reads a bit wide; a relaxed hand drifting slowly
         doesn't click.
    Out: the gap opens past `exit` (or 70% of how open it was before, whichever is larger, capped),
         for 2 frames, so one noisy frame never drops you mid-drag.

    Why: log 20260929-201849 had 37 attempts that bottomed out at 0.14-0.20 against a fixed 0.14
    cutoff (MediaPipe's 3D fingertips never quite meet), and only 12 that counted."""

    def __init__(self, instant: float, enter: float, exit: float, exit_frames: int = 2,
                 drop: float = 0.65, window: float = 0.5):
        self.instant, self.enter, self.exit, self.exit_frames = instant, enter, exit, exit_frames
        self.drop, self.window = drop, window
        self.reset()

    def reset(self):
        self.pinched = False
        self._candidate = 0
        self._exit = 0
        self._hist = []          # (t, ratio, curl) over the last `window` seconds
        self._t = 0.0
        self.press_ref = 0.0     # how open the fingers were just before the pinch
        self.rejected_fists = 0

    def set_thresholds(self, instant: float, enter: float, exit: float):
        """From your hand calibration."""
        self.instant, self.enter, self.exit = instant, enter, exit

    def update(self, ratio: float, t: float | None = None, curl: float | None = None) -> bool:
        """`curl`: middle/ring finger extension (landmarks.finger_curl). When it drops sharply at the
        same time, the whole hand is closing into a fist: thumb and index meet, but it's not a pinch."""
        self._t = t if t is not None else self._t + 1 / 30
        ref = max((r for _, r, _ in self._hist), default=ratio)
        curls = [c for _, _, c in self._hist if c is not None]
        fist = curl is not None and curls and curl < 0.75 * max(curls)
        self._hist.append((self._t, ratio, curl))
        self._hist = [h for h in self._hist if self._t - h[0] <= self.window]
        if not self.pinched:
            closing = ratio < self.enter and ratio <= self.drop * ref
            if fist and (ratio < self.instant or closing):
                self._candidate = 0
                self.rejected_fists += 1
            elif ratio < self.instant:
                self.pinched = True
            elif closing:
                self._candidate += 1
                self.pinched = self._candidate >= 2
            else:
                self._candidate = 0
            if self.pinched:
                self.press_ref, self._candidate, self._exit = ref, 0, 0
        else:
            release_at = max(self.exit, min(self.exit + 0.06, 0.7 * self.press_ref))
            if ratio > release_at:
                self._exit += 1
                if self._exit >= self.exit_frames:
                    self.pinched, self._exit = False, 0
            else:
                self._exit = 0
        return self.pinched
