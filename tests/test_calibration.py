"""Hand calibration, the steadier hand measurements, fist rejection, and the calibrated panel."""
import numpy as np

from nexus.calibration import NEED, PHASES, SETTLE, Calibrator, apply
from nexus.config import Config
from nexus.filters import PinchDetector
from nexus.landmarks import Hand, pinch_ratio, robust_size
from nexus.spread import SpreadControl


def feed(cal: Calibrator, levels: dict, noise: float = 0.01, seed: int = 0):
    """Run all four poses with (pinch_ratio, spread_raw) around the given levels."""
    rng = np.random.default_rng(seed)
    for name, *_ in PHASES:
        pr, sp = levels[name]
        for _ in range(SETTLE + NEED):
            cal.add(pr + rng.normal(0, noise), sp + rng.normal(0, noise))


# Levels like a real hand: pinch ratio open 0.38 / touching 0.16; spread bunched 0.12 / wide 0.55
HAND = {"open": (0.38, 0.40), "pinch": (0.16, 0.30), "bunch": (0.30, 0.12), "spread": (0.40, 0.55)}


def test_calibration_places_thresholds_between_your_levels():
    cal = Calibrator()
    feed(cal, HAND)
    assert cal.done
    c, err = cal.result()
    assert err == "" and c is not None
    assert c["pinch_touch"] < c["pinch_instant"] < c["pinch_enter"] < c["pinch_exit"] < c["pinch_open"]
    assert 0.1 < c["spread_lo"] < 0.2 and 0.45 < c["spread_hi"] < 0.6


def test_calibration_rejects_poses_that_look_the_same():
    cal = Calibrator()
    feed(cal, {**HAND, "pinch": (0.36, 0.30)})  # "pinched" but fingers never actually closed
    c, err = cal.result()
    assert c is None and "too similar" in err


def test_settle_frames_are_ignored_and_redo_restarts_a_pose():
    cal = Calibrator()
    for _ in range(SETTLE):
        cal.add(0.9, 0.9)            # getting into position: ignored
    assert cal.samples["open"] == []
    for _ in range(10):
        cal.add(0.38, 0.40)
    cal.redo()
    assert cal.samples["open"] == [] and cal.phase[0] == "open"


def test_apply_sets_live_thresholds_and_panel_range():
    cal = Calibrator()
    feed(cal, HAND)
    c, _ = cal.result()
    cfg, det, spread = Config(), PinchDetector(0.13, 0.22, 0.25), SpreadControl()

    class Engine:
        pinch = det
    apply(c, cfg, Engine(), spread)
    assert det.enter == c["pinch_enter"] and cfg.pinch_exit == c["pinch_exit"]
    assert spread.calibrated and spread.lo == c["spread_lo"]


def _real_hand(gap: float) -> np.ndarray:
    """A plausible full 21-point hand (meters) with thumb and index tips `gap` apart."""
    w = np.zeros((21, 3), np.float32)
    w[0] = [0, 0.09, 0]
    for j, x in zip([5, 9, 13, 17], [-0.025, 0, 0.02, 0.04]):
        w[j] = [x, 0, 0]
        for k in (1, 2, 3):
            w[j + k] = [x, -0.035 * k, 0]            # straight fingers, ~0.035 per bone
    w[1], w[2], w[3] = [-0.03, 0.07, 0.01], [-0.045, 0.045, 0.02], [-0.05, 0.02, 0.025]
    w[4] = w[8] + [gap, 0, 0]
    return w


def test_robust_size_ignores_one_bad_bone():
    w = _real_hand(0.05)
    good = robust_size(w)
    bad = w.copy()
    bad[9] = [0, -0.05, 0.04]  # the middle knuckle estimated badly (common with the hand turned)
    assert abs(robust_size(bad) - good) / good < 0.15
    single_before = np.linalg.norm(w[9] - w[0])
    single_after = np.linalg.norm(bad[9] - bad[0])
    assert abs(single_after - single_before) / single_before > 0.4  # the old one-bone size swung a lot


def test_pads_touching_count_as_a_pinch():
    w = _real_hand(0.05)
    w[3] = w[8] + [0.004, 0.002, 0]  # thumb pad on the index tip, thumb tip beyond it
    w[4] = w[8] + [0.03, -0.02, 0]
    hand = Hand(landmarks=w, world=w, score=1.0)
    assert pinch_ratio(hand) < 0.1


def test_making_a_fist_does_not_click():
    p = PinchDetector(instant=0.13, enter=0.22, exit=0.25)
    ratios = [0.36] * 8 + [0.26, 0.17, 0.12, 0.10, 0.10]
    curls = [1.9] * 8 + [1.6, 1.3, 1.1, 1.0, 1.0]     # middle and ring fingers curling at the same time
    assert not any(p.update(r, curl=c) for r, c in zip(ratios, curls))
    assert p.rejected_fists > 0
    p2 = PinchDetector(instant=0.13, enter=0.22, exit=0.25)
    assert any(p2.update(r, curl=1.9) for r in ratios)  # same thumb-index motion, other fingers still: a click


def test_calibrated_panel_opens_and_closes_with_your_range():
    from tests.test_spread import hand as spread_hand
    from nexus.spread import raw_spread
    c = SpreadControl()
    c.calibrate(raw_spread(spread_hand(0.0)), raw_spread(spread_hand(1.0)))
    t = 0.0
    for o in [0.0] * 8 + [1.0] * 12:
        c.update(spread_hand(o), t); t += 1 / 30
    assert c.is_open
    for o in [1.0] * 5 + [0.0] * 8:
        c.update(spread_hand(o), t); t += 1 / 30
    assert not c.is_open
    assert c.lo == raw_spread(spread_hand(0.0))  # calibrated range doesn't drift
