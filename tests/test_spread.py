"""Five-finger spread control for the Jarvis panel."""
import numpy as np

from nexus.spread import SpreadControl, raw_spread


def hand(openness: float) -> np.ndarray:
    """World landmarks with the five fingertips spread by `openness` (0 bunched .. 1 wide open)."""
    w = np.zeros((21, 3), np.float32)
    w[0] = [0, 0.09, 0]      # wrist, hand size 0.09 m
    w[9] = [0, 0, 0]         # middle knuckle
    center = np.array([0, -0.07, 0])
    angles = np.linspace(-1.2, 1.2, 5)
    radius = 0.008 + openness * 0.05
    for tip, a in zip([4, 8, 12, 16, 20], angles):
        w[tip] = center + radius * np.array([np.sin(a), -np.cos(a), 0])
    return w


def drive(ctrl, values, t0=0.0, dt=1 / 30, pinched=False):
    t, out = t0, []
    for v in values:
        out.append(ctrl.update(hand(v) if v is not None else None, t, pinched))
        t += dt
    return out, t


def test_spread_measure_grows_with_openness():
    assert raw_spread(hand(0.0)) < raw_spread(hand(0.5)) < raw_spread(hand(1.0))


def test_bunch_then_spread_opens_proportionally():
    c = SpreadControl(lo=raw_spread(hand(0)), hi=raw_spread(hand(1)))
    targets, t = drive(c, [0.0] * 10)             # bunched, held -> session
    assert c.session
    targets, t = drive(c, [0.4] * 15, t)          # 40% spread -> ~40% open
    assert 0.3 < targets[-1] < 0.5
    targets, t = drive(c, [1.0] * 10, t)          # fully spread -> locked open
    assert c.is_open and targets[-1] > 0.9
    assert "opened" in c.pop_events()


def test_relaxed_hand_does_not_open_panel():
    c = SpreadControl(lo=raw_spread(hand(0)), hi=raw_spread(hand(1)))
    targets, _ = drive(c, [0.6] * 60 + [1.0] * 30)  # pointing/relaxed, then an open palm
    assert not c.is_open and max(targets) == 0.0    # opening must start from bunched fingertips


def test_partial_close_follows_fingers_then_settles():
    c = SpreadControl(lo=raw_spread(hand(0)), hi=raw_spread(hand(1)))
    _, t = drive(c, [0.0] * 10 + [1.0] * 15)     # open it
    assert c.is_open
    targets, t = drive(c, [1.0] * 8 + [0.2] * 12, t)  # spread, then close to 80%
    assert 0.1 < targets[-1] < 0.35              # panel ~80% closed, following the fingers
    targets, t = drive(c, [0.0] * 10, t)         # all the way
    assert not c.is_open and targets[-1] < 0.1
    assert "closed" in c.pop_events()


def test_pinch_or_lost_hand_settles_to_nearest():
    c = SpreadControl(lo=raw_spread(hand(0)), hi=raw_spread(hand(1)))
    _, t = drive(c, [0.0] * 10 + [0.75] * 5)     # opened most of the way...
    targets, _ = drive(c, [None], t)             # ...then the hand leaves
    assert c.is_open and targets[-1] == 1.0
