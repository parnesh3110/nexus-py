"""Hand calibration: your own numbers for "open", "touching", "bunched" and "spread", so pinch and
the Jarvis panel gesture fit your hand and how you hold it, instead of one guess for everyone.
(A touch read 0.035 in one position and 0.20 in another in log 20260929-201849.)

Four short poses, about 1.5 s of samples each after a moment to get into position:
    open    hand up, fingers relaxed and apart       -> pinch "open" level
    pinch   thumb and index touching, held            -> pinch "touching" level
    bunch   all five fingertips together              -> panel "closed" level
    spread  fingers spread wide                       -> panel "open" level
Thresholds are placed between your levels, with sanity limits, and saved in settings.
Pure Python + numpy: unit-tested.
"""
import time

import numpy as np

PHASES = [
    ("open", "hand.raised.fill", "Hold your hand up, relaxed", "Palm facing the screen, fingers a little apart."),
    ("pinch", "hand.point.up.left.fill", "Now pinch and hold", "Touch your thumb and index finger together, like you would to click."),
    ("bunch", "hand.point.up.braille.fill", "Bring all fingertips together", "All five fingertips touching, like a closed flower."),
    ("spread", "hand.raised.fingers.spread.fill", "Spread your fingers wide", "As wide as is comfortable."),
]
SETTLE = 12   # samples ignored at the start of each pose (~0.4 s to get into it)
NEED = 45     # samples kept per pose (~1.5 s at 30 fps)


class Calibrator:
    def __init__(self):
        self.i = 0
        self.samples = {name: [] for name, *_ in PHASES}
        self._seen = 0
        self.started = time.time()

    @property
    def done(self) -> bool:
        return self.i >= len(PHASES)

    @property
    def phase(self):
        return PHASES[min(self.i, len(PHASES) - 1)]

    @property
    def progress(self) -> float:
        """0..1 for the current pose."""
        return min(1.0, len(self.samples[self.phase[0]]) / NEED) if not self.done else 1.0

    def add(self, pinch_ratio: float | None, spread_raw: float | None) -> bool:
        """One frame with a hand in view. Returns True when this pose just finished."""
        if self.done or pinch_ratio is None or spread_raw is None:
            return False
        if not (np.isfinite(pinch_ratio) and np.isfinite(spread_raw)):
            return False
        self._seen += 1
        if self._seen <= SETTLE:
            return False
        name = self.phase[0]
        self.samples[name].append((float(pinch_ratio), float(spread_raw)))
        if len(self.samples[name]) >= NEED:
            self.i += 1
            self._seen = 0
            return True
        return False

    def redo(self):
        """Start the current pose again (hand left the frame, or you asked)."""
        if not self.done:
            self.samples[self.phase[0]] = []
            self._seen = 0

    def result(self) -> tuple[dict | None, str]:
        """(calibration, "") or (None, what went wrong in plain words)."""
        if not self.done:
            return None, "not finished"
        p = lambda name, k, q: float(np.percentile([s[k] for s in self.samples[name]], q))
        open_p, touch_p = p("open", 0, 50), p("pinch", 0, 75)
        bunch_s, spread_s = p("bunch", 1, 70), p("spread", 1, 30)
        if open_p < touch_p * 1.4 or open_p - touch_p < 0.08:
            return None, "Open and pinch looked too similar. Keep your fingers clearly apart for the first pose."
        if spread_s < bunch_s * 1.5:
            return None, "Bunched and spread looked too similar. Spread your fingers wider for the last pose."
        span = open_p - touch_p
        instant = float(np.clip(touch_p + 0.15 * span, 0.06, 0.25))
        enter = float(np.clip(touch_p + 0.50 * span, instant + 0.03, 0.36))
        exit_ = float(np.clip(touch_p + 0.65 * span, enter + 0.02, 0.45))
        return {
            "pinch_touch": round(touch_p, 3), "pinch_open": round(open_p, 3),
            "pinch_instant": round(instant, 3), "pinch_enter": round(enter, 3), "pinch_exit": round(exit_, 3),
            "open_ratio": round(float(np.clip(touch_p + 0.8 * span, exit_, 0.6)), 3),
            "spread_lo": round(bunch_s, 3), "spread_hi": round(spread_s, 3),
            "date": time.strftime("%Y-%m-%d"),
        }, ""


def apply(cal: dict | None, cfg=None, engine=None, spread=None):
    """Use a saved calibration: pinch thresholds (config + live detector) and the panel range."""
    if not cal:
        return
    if cfg is not None:
        cfg.pinch_instant, cfg.pinch_enter, cfg.pinch_exit = cal["pinch_instant"], cal["pinch_enter"], cal["pinch_exit"]
        cfg.open_ratio = cal["open_ratio"]
    if engine is not None:
        engine.pinch.set_thresholds(cal["pinch_instant"], cal["pinch_enter"], cal["pinch_exit"])
    if spread is not None:
        spread.calibrate(cal["spread_lo"], cal["spread_hi"])
