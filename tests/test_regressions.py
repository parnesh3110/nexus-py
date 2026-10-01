"""Regression tests for bugs found in real logs (log 20260927-213438)."""
import numpy as np
import pytest

from nexus.landmarks import Hand, is_bunched
from nexus.voice import is_hallucination


def _hand(tips_near_thumb: bool, fingers_curled: bool) -> Hand:
    """World landmarks: wrist, knuckles, thumb tip, fingertips placed to model a pose."""
    w = np.zeros((21, 3), np.float32)
    w[0] = [0, 0.09, 0]                                   # wrist; hand size 0.09 m
    for j, x in zip([5, 9, 13, 17], [-0.025, 0, 0.02, 0.04]):
        w[j] = [x, 0, 0]                                  # knuckles
    thumb = np.array([-0.01, -0.02 if fingers_curled else -0.07, 0.03])
    w[4] = thumb
    for tip, mcp in zip([8, 12, 16, 20], [5, 9, 13, 17]):
        if tips_near_thumb:
            w[tip] = thumb + [0.005 * (tip % 5), 0.004, 0]  # all tips right next to the thumb
        else:
            w[tip] = w[mcp] + [0, -0.08, 0]                 # finger extended away
        if fingers_curled and tip != 8:
            w[tip] = w[mcp] + [0, -0.02, 0.025]             # curled: tip tucked near its own knuckle
    if fingers_curled:
        w[8] = thumb + [0.004, 0, 0]                        # index touching thumb = the pinch
    return Hand(landmarks=w, world=w, score=1.0)


def test_real_bunch_is_bunched():
    assert is_bunched(_hand(tips_near_thumb=True, fingers_curled=False))


def test_relaxed_pinch_with_curled_fingers_is_not_bunched():
    # This pose used to be read as the panel gesture, so the click was thrown away
    assert not is_bunched(_hand(tips_near_thumb=False, fingers_curled=True))


def test_open_hand_is_not_bunched():
    assert not is_bunched(_hand(tips_near_thumb=False, fingers_curled=False))


def test_whisper_noise_phrases_dropped():
    for junk in ["We'll see you in the next video.", "Thanks for watching!", "you", "Bye."]:
        assert is_hallucination(junk), junk
    for real in ["Open Apple Music.", "Can you play the reason?", "minimize"]:
        assert not is_hallucination(real), real


def test_actions_with_an_action_argument_dont_crash():
    # music/window/system steps carry an argument named "action"; run() used to crash on them
    pytest.importorskip("Quartz")
    from nexus import actions
    ok, msg = actions.run("no_such_action", action="play", query="x")
    assert not ok and "unknown action" in msg


# --- log 20260928-145541 ---

def test_releasing_a_pinch_does_not_open_the_panel():
    # A curled pinch hand (tips fairly close, fingers NOT straight) then opening the fingers used to
    # count as "bunch, then spread" and pop the panel open during clicks.
    from nexus.spread import SpreadControl, raw_spread
    from tests.test_spread import hand as spread_hand
    curled = _hand(tips_near_thumb=False, fingers_curled=True).world
    c = SpreadControl(lo=raw_spread(spread_hand(0)), hi=raw_spread(spread_hand(1)))
    t = 0.0
    for _ in range(5):                       # pinching
        c.update(curled, t, pinched=True); t += 1 / 30
    for _ in range(10):                      # fingers still close right after the click
        c.update(spread_hand(0.0), t); t += 1 / 30
    for _ in range(15):                      # hand opens
        c.update(spread_hand(1.0), t); t += 1 / 30
    assert not c.is_open and "opened" not in c.pop_events()


def test_panel_still_opens_on_purpose():
    from nexus.spread import SpreadControl, raw_spread
    from tests.test_spread import hand as spread_hand
    c = SpreadControl(lo=raw_spread(spread_hand(0)), hi=raw_spread(spread_hand(1)))
    t = 0.0
    for o in [0.0] * 10 + [1.0] * 15:
        c.update(spread_hand(o), t); t += 1 / 30
    assert c.is_open


def test_pinch_wobble_stays_a_click():
    # The hand wobbling ~180 px while pinching turned 2 of 5 clicks into drags
    from nexus.config import Config
    from nexus.pointer import PointerEngine
    from tests.test_pointer import make_hand, run
    cfg = Config()
    e = PointerEngine(1512, 982, cfg)
    px_per_cam = 1512 / cfg.box_width
    wobble = 180 / px_per_cam                 # 180 px of hand wobble, in camera units
    frames = [make_hand(gap=0.08)] * 15 + [make_hand(palm_x=0.5 + wobble * k / 4, gap=0.01) for k in range(5)]
    states, t = run(e, frames)
    assert any(s.event == "down" for s in states)
    assert states[-1].holding_click           # still a click, pinned to the spot


# --- log 20260929-201849 ---

def test_relaxing_your_hand_keeps_the_panel_open():
    # 20:22:06: the panel was open, the hand relaxed at 44% openness, and it closed by itself
    from nexus.spread import SpreadControl, raw_spread
    from tests.test_spread import hand as spread_hand
    c = SpreadControl(lo=raw_spread(spread_hand(0)), hi=raw_spread(spread_hand(1)))
    t = 0.0
    for o in [0.0] * 10 + [1.0] * 15:          # open it
        c.update(spread_hand(o), t); t += 1 / 30
    assert c.is_open
    for o in [1.0] * 6 + [0.44] * 40:          # spread (a session starts), then just relax
        c.update(spread_hand(o), t); t += 1 / 30
    assert not c.session and c.is_open        # session over, panel still open
    for o in [1.0] * 6 + [0.2] * 4:            # closing most of the way, then the hand leaves
        c.update(spread_hand(o), t); t += 1 / 30
    c.update(None, t)
    assert not c.is_open                       # that one was on purpose: closed


def test_music_service_names_are_dropped():
    from nexus.commands import Step, parse
    assert parse("Play The Reason on YouTube Music.") == [Step("music", {"action": "play", "query": "The Reason"})]
    assert parse("Play.") == [Step("music", {"action": "resume"})]
