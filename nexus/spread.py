"""Five-finger spread -> how open the Jarvis panel is. Pure (numpy), unit-tested.

Openness = how far the five fingertips are from each other (their spread around their own
center), in hand sizes, mapped to 0 (all tips bunched together) .. 1 (hand fully spread).

A "session" is when your fingers are driving the panel:
  panel closed: bunch your fingertips (openness <= 0.2) for 0.15 s -> session starts
  panel open:   spread your hand (openness >= 0.85) for 0.15 s    -> session starts
During a session the panel follows your fingers exactly (80% closed = 80% closed).
Fully spread (>= 0.9) locks it open; fully bunched (<= 0.1) closes it. The session ends when
you pinch, your hand leaves, or your fingers sit relaxed in the middle for 0.6 s; then the
panel settles to whichever end is nearer.
Starting to open needs bunched fingertips, which you never do while just pointing, so the
panel doesn't pop open by accident.
"""
import numpy as np

from .filters import OneEuro
from .landmarks import bunched_world, robust_size

TIPS = [4, 8, 12, 16, 20]


def raw_spread(world: np.ndarray) -> float:
    """Mean fingertip distance from the fingertips' center, divided by hand size (robust_size: several
    bones, so turning your hand doesn't change the answer much)."""
    size = robust_size(world)
    tips = world[TIPS]
    return float(np.linalg.norm(tips - tips.mean(axis=0), axis=1).mean() / size)


class SpreadControl:
    START_HOLD = 0.10    # s an extreme pose must be held to start a session
    RELAXED_END = 0.6    # s in the middle band that ends a session
    AFTER_PINCH = 0.8    # s after a click before the panel gesture can start (log 20260928-145541:
                         # releasing a pinch looked like "bunch, then spread" and opened the panel)
    # When a session ends mid-way (you relax, pinch, or your hand leaves), the panel keeps its state
    # unless you clearly went most of the way. Log 20260929-201849: relaxing at 44% used to close it.
    OPEN_ON_RELEASE, CLOSE_ON_RELEASE = 0.65, 0.35
    BUNCHED, SPREAD = 0.2, 0.85
    CLOSE_AT, LOCK_AT = 0.1, 0.9

    def __init__(self, lo: float = 0.10, hi: float = 0.55):
        self.lo, self.hi = lo, hi            # raw spread at "bunched" and "fully spread"
        self.calibrated = False              # True: lo/hi measured on your hand, no guessing
        self.last_raw = None                 # latest raw spread, for calibration and the stats log
        self.filter = OneEuro(min_cutoff=1.5, beta=4.0)
        self.is_open = False                 # panel state when no session is running
        self.session = False
        self.openness = 0.0                  # smoothed finger openness 0..1
        self.target = 0.0                    # how open the panel should be right now
        self._pose_since = None
        self._relaxed_since = None
        self._last_pinch = -1e9
        self._peak = self._trough = 0.0  # extremes since the panel last changed state
        self.events: list[str] = []          # "opened" / "closed" since last read

    def calibrate(self, lo: float, hi: float):
        """Your measured bunched/spread levels (calibration.py): 0 and 1 are now exactly your poses."""
        self.lo, self.hi, self.calibrated = lo, hi, True

    def _normalize(self, spread: float) -> float:
        # Uncalibrated: adapt slowly to this hand, the extremes it actually reaches become 0 and 1
        if not self.calibrated:
            if spread < self.lo:
                self.lo = self.lo * 0.9 + spread * 0.1
            if spread > self.hi:
                self.hi = self.hi * 0.9 + spread * 0.1
        return float(min(max((spread - self.lo) / max(self.hi - self.lo, 1e-3), 0.0), 1.0))

    def _set_open(self, value: bool):
        if value != self.is_open:
            self.is_open = value
            self.events.append("opened" if value else "closed")
            self._peak = self._trough = self.openness

    def _end_session(self):
        if self.session:
            self.session = False
            if self.is_open:
                self._set_open(self._trough > self.CLOSE_ON_RELEASE)  # stays open unless you closed most of the way
            else:
                self._set_open(self._peak >= self.OPEN_ON_RELEASE)    # stays closed unless you opened most of the way
        self._pose_since = self._relaxed_since = None

    def update(self, world: np.ndarray | None, t: float, pinched: bool = False) -> float:
        """Call every frame with the hand's world landmarks (None if no hand). Returns the target
        panel openness 0..1 (the UI eases toward it)."""
        if world is None or pinched:
            if pinched:
                self._last_pinch = t
            self._end_session()
            self.filter.reset()
            self.target = 1.0 if self.is_open else 0.0
            return self.target

        self.last_raw = raw_spread(world)
        o = min(max(self.filter(self._normalize(self.last_raw), t), 0.0), 1.0)
        self.openness = o

        if not self.session:
            if not self.is_open:
                # Opening starts from all five fingertips together, not a curled pinch hand, and not
                # right after a click. Calibrated: "together" is your own bunched level.
                shape_ok = (bunched_world(world, together=0.55, straight=0.35) if self.calibrated
                            else bunched_world(world, together=0.5, straight=0.42))
                at_start_pose = (o <= (0.25 if self.calibrated else self.BUNCHED) and shape_ok
                                 and t - self._last_pinch >= self.AFTER_PINCH)
            else:
                at_start_pose = o >= self.SPREAD
            if at_start_pose:
                self._pose_since = self._pose_since if self._pose_since is not None else t
                if t - self._pose_since >= self.START_HOLD:
                    self.session = True
                    self._relaxed_since = None
                    self._peak = self._trough = o
            else:
                self._pose_since = None
        if self.session:
            self.target = o
            self._peak, self._trough = max(self._peak, o), min(self._trough, o)
            if o >= self.LOCK_AT:
                self._set_open(True)
            elif o <= self.CLOSE_AT:
                self._set_open(False)
            if 0.3 < o < 0.7:
                self._relaxed_since = self._relaxed_since if self._relaxed_since is not None else t
                if t - self._relaxed_since >= self.RELAXED_END:
                    self._end_session()
            else:
                self._relaxed_since = None
        if not self.session:
            self.target = 1.0 if self.is_open else 0.0
        return self.target

    def pop_events(self) -> list[str]:
        e, self.events = self.events, []
        return e

    def force(self, is_open: bool):
        """Open or close from the menu bar or a button, without the gesture. The caller handles the
        conversation, so no event is queued (a stale one could flip it later)."""
        self._end_session()
        self._set_open(is_open)
        self.events.clear()
        self.target = 1.0 if is_open else 0.0
