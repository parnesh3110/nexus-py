"""Pointer + filter logic, with synthetic hands (no camera, MediaPipe or macOS needed)."""
import math

import numpy as np

from nexus.config import Config
from nexus.filters import OneEuro, PinchDetector
from nexus.landmarks import Hand, palm_point, pinch_ratio
from nexus.pointer import PointerEngine

W, H = 1512.0, 982.0


def make_hand(palm_x=0.5, palm_y=0.5, gap=0.08):
    """Hand facing the camera: knuckles around (palm_x, palm_y) in MIRRORED coords, hand size
    0.2 (wrist to middle knuckle), thumb and index tips `gap` apart."""
    lm = np.zeros((21, 3), dtype=np.float32)
    x = 1 - palm_x  # landmarks are un-mirrored image coords
    for i, dx in zip([5, 9, 13, 17], [-0.03, -0.01, 0.01, 0.03]):
        lm[i] = [x + dx, palm_y, 0]
    lm[0] = [x, palm_y + 0.2, 0]            # wrist
    lm[8] = [x - 0.03, palm_y - 0.1, 0]     # index tip
    lm[4] = [x - 0.03 + gap, palm_y - 0.1, 0]  # thumb tip
    world = lm.copy()  # same shape is fine: pinch_ratio only uses ratios
    return Hand(landmarks=lm, world=world, score=0.99)


def run(engine, frames, t0=0.0, dt=1 / 30):
    states, t = [], t0
    for h in frames:
        states.append(engine.update(h, t))
        t += dt
    return states, t


# --- filters ---

def test_one_euro_converges_and_smooths():
    f = OneEuro(1.0, 0.0)
    out = [f(1.0 if i > 0 else 0.0, i / 30) for i in range(120)]
    assert out[0] == 0.0
    assert 0.0 < out[1] < 1.0          # smoothed, not a jump
    assert abs(out[-1] - 1.0) < 1e-3   # but gets there


def test_pinch_hysteresis():
    p = PinchDetector(instant=0.13, enter=0.22, exit=0.25)
    for r in (0.35, 0.35, 0.35):
        assert p.update(r) is False      # open hand
    assert p.update(0.18) is False      # closing fast, first frame...
    assert p.update(0.18) is True       # ...second frame: a click (0.18 used to be a miss)
    assert p.update(0.23) is True       # opening a little: still pinched (below release)
    assert p.update(0.30) is True       # one open frame: still pinched (noise)
    assert p.update(0.30) is False      # two: released
    for r in (0.35, 0.35):
        p.update(r)
    assert p.update(0.10) is True       # very close: instant


def test_light_touches_from_the_log_count():
    # log 20260929-201849: touches bottomed out at 0.14-0.20 and were missed (cutoff was 0.14)
    for bottom in (0.141, 0.16, 0.18, 0.197):
        p = PinchDetector(instant=0.13, enter=0.22, exit=0.25)
        ratios = [0.34] * 6 + [0.26, bottom, bottom, bottom, 0.26, 0.34, 0.34]
        clicked = [p.update(r) for r in ratios]
        assert any(clicked), bottom
        assert clicked[-1] is False, bottom   # and it lets go afterwards


def test_relaxed_hand_drifting_does_not_click():
    # thumb slowly settling near the index finger over 1.5 s is not a pinch
    p = PinchDetector(instant=0.13, enter=0.22, exit=0.25)
    ratios = [0.30 - 0.11 * k / 45 for k in range(46)] + [0.19] * 20
    assert not any(p.update(r) for r in ratios)


def test_pinch_ratio_scale_invariant():
    near, far = make_hand(gap=0.06), make_hand(gap=0.06)
    far.landmarks *= 0.5
    far.world *= 0.5
    assert math.isclose(pinch_ratio(near), pinch_ratio(far), rel_tol=1e-4)


def test_palm_is_mirrored():
    assert palm_point(make_hand(palm_x=0.3))[0] == np.float32(0.3) or math.isclose(palm_point(make_hand(palm_x=0.3))[0], 0.3, abs_tol=1e-5)


# --- pointer ---

def test_ghost_hand_ignored():
    e = PointerEngine(W, H, Config())
    s = e.update(make_hand(), 0.0)
    assert not s.visible                 # one frame is not enough to acquire
    s = e.update(make_hand(), 1 / 30)
    assert s.visible


def test_hand_movement_moves_pointer_right():
    e = PointerEngine(W, H, Config())
    states, _ = run(e, [make_hand(palm_x=0.5 + i * 0.002) for i in range(40)])
    assert states[-1].x > states[5].x + 50


def test_still_hand_gives_still_pointer():
    e = PointerEngine(W, H, Config())
    rng = np.random.default_rng(0)
    frames = [make_hand(palm_x=0.5 + rng.normal(0, 0.0008), palm_y=0.5 + rng.normal(0, 0.0008)) for _ in range(90)]
    states, _ = run(e, frames)
    xs = [s.x for s in states[45:]]
    assert max(xs) - min(xs) < 3        # tremor absorbed by band + lock


def test_quick_pinch_is_a_click_not_a_drag():
    e = PointerEngine(W, H, Config())
    open_ = [make_hand(gap=0.08) for _ in range(15)]
    closed = [make_hand(gap=0.01) for _ in range(4)]
    states, _ = run(e, open_ + closed + open_[:4])
    events = [s.event for s in states if s.event]
    assert events == ["down", "up"]
    down = next(s for s in states if s.event == "down")
    assert down.holding_click


def test_pinch_and_pull_becomes_drag():
    e = PointerEngine(W, H, Config())
    open_ = [make_hand(gap=0.08) for _ in range(15)]
    held = [make_hand(palm_x=0.5 + i * 0.004, gap=0.01) for i in range(30)]
    states, _ = run(e, open_ + held)
    assert any(s.event == "down" for s in states)
    assert not states[-1].holding_click  # became a drag
    assert states[-1].pinched
    assert states[-1].x > states[16].x + 100


def test_losing_hand_releases_pinch():
    e = PointerEngine(W, H, Config())
    states, t = run(e, [make_hand(gap=0.08)] * 15 + [make_hand(gap=0.01)] * 4)
    assert states[-1].pinched
    lost, _ = run(e, [None] * 20, t0=t)
    assert any(s.event == "up" for s in lost)
    assert not lost[-1].pinched


def test_click_releases_where_it_pressed():
    # log 20260927-210532: pressed at x=1431, released at x=560 as the hand moved on the release frame
    e = PointerEngine(W, H, Config())
    states, t = run(e, [make_hand(gap=0.08)] * 15 + [make_hand(gap=0.01)] * 3)
    down = next(s for s in states if s.event == "down")
    up = e.update(make_hand(palm_x=0.51, gap=0.3), t)  # fingers open, hand moved ~100 px (still a click)
    up = up if up.event == "up" else e.update(make_hand(palm_x=0.51, gap=0.3), t + 1 / 30)
    assert up.event == "up"
    assert abs(up.x - down.x) < 1 and abs(up.y - down.y) < 1


def test_glitch_jump_rejected():
    e = PointerEngine(W, H, Config())
    states, t = run(e, [make_hand(palm_x=0.5)] * 20)
    before = states[-1].x
    s = e.update(make_hand(palm_x=0.9), t)  # 0.4 camera widths in one frame = impossible
    assert abs(s.x - before) < 1
