"""Hand landmarks -> a stable screen pointer + click/drag events.

Ported from the Swift version, including every fix its logs led to:
- direct pointing through a small camera "box" that slides with your hand (no edge dead zones)
- One Euro smoothing, then a rubber band + still-lock so a still hand gives a still pointer
- clicks snap back to where you aimed before your fingers started closing
- a click only becomes a drag after a clear, held pull
- one-frame "ghost" hands and landmark glitches are ignored; short dropouts are bridged
Pure Python + numpy: no camera, MediaPipe or macOS needed, so it's unit-tested.
"""
import math
from collections import deque
from dataclasses import dataclass

from .config import Config
from .filters import OneEuro, PinchDetector
from .landmarks import Hand, finger_curl, is_bunched, palm_point, pinch_ratio
from .log import log


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ms(s: float) -> int:
    return int(round(s * 1000))


@dataclass
class PointerState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0              # px/s, for the cursor animator's prediction
    vy: float = 0.0
    t: float = 0.0
    visible: bool = False
    visibility: float = 0.0
    pinched: bool = False
    holding_click: bool = False  # pinched, pointer pinned to the click spot
    event: str | None = None     # "down" / "up" on the frame it happens
    ratio: float | None = None
    raw_x: float = 0.0           # unsmoothed mapping, for jitter measurement
    raw_y: float = 0.0
    cam_x: float = -1.0          # palm in camera space (mirrored), for tuning
    cam_y: float = -1.0
    locked: bool = False


class PointerEngine:
    def __init__(self, screen_w: float, screen_h: float, cfg: Config, cam_aspect: float = 4 / 3):
        self.W, self.H, self.cfg, self.aspect = screen_w, screen_h, cfg, cam_aspect
        self.fx = OneEuro(cfg.min_cutoff, cfg.beta)
        self.fy = OneEuro(cfg.min_cutoff, cfg.beta)
        self.pinch = PinchDetector(cfg.pinch_instant, cfg.pinch_enter, cfg.pinch_exit, cfg.pinch_exit_frames)
        self.pos = (screen_w / 2, screen_h / 2)
        self.stable = self.pos
        self.vel = (0.0, 0.0)
        self.box = None
        self.lost_since = None
        self.reported_lost = True
        self.shown_since = None
        self.seen_frames = 0
        self.last_t = None
        self.vel_t = None
        self.still_since = None
        self.locked = False
        self.history: deque = deque()  # (t, pos, ratio) for the click snap-back
        self.click_spot = None
        self._release_at = None
        self.drag_origin = None
        self.offset = (0.0, 0.0)
        self.pinch_down_t = None
        self.last_cam = None
        self.glitches = 0
        self._dip = None  # closest thumb-index gap in the current almost-pinch

    # --- public ---

    def away(self, t: float) -> float:
        """Seconds without a hand (0 while one is tracked). Drives idle mode."""
        return t - self.lost_since if self.reported_lost and self.lost_since is not None else 0.0

    def reset(self, reason: str):
        log.debug("pointer", "reset", reason=reason)
        self.fx.reset()
        self.fy.reset()
        self.box = None
        self.locked = False
        self.still_since = None
        self.click_spot = None
        self.drag_origin = None
        self.offset = (0.0, 0.0)
        self.stable = self.pos
        self.history.clear()
        self.last_t = None
        self.pinch.reset()

    def update(self, hand: Hand | None, t: float) -> PointerState:
        cfg = self.cfg
        s = PointerState(t=t)
        prev = self.pos
        event = None

        if hand is not None and self._is_glitch(hand, t):
            hand = None
        # A new hand has to be there two frames in a row: one-frame ghosts used to throw the pointer
        self.seen_frames = self.seen_frames + 1 if hand is not None else 0
        if hand is not None and self.reported_lost and self.seen_frames < cfg.acquire_frames:
            hand = None

        if hand is not None:
            if self.reported_lost:
                log.info("hand", "acquired", after_ms=_ms(t - (self.lost_since or t)), score=hand.score)
                self.reported_lost = False
                self.shown_since = t
            elif self.lost_since is not None:
                log.debug("hand", "dropout_bridged", ms=_ms(t - self.lost_since))
            if self.lost_since is not None and t - self.lost_since > cfg.reset_after_lost:
                self.reset("reacquired")
            self.lost_since = None

            cam = palm_point(hand)
            self.last_cam = (cam[0], cam[1], t)
            ratio = pinch_ratio(hand, self.aspect)
            if not self.pinch.pinched and is_bunched(hand):
                ratio = 1.0  # all five fingertips together is the panel gesture, not a click
            f = (self.fx(cam[0], t), self.fy(cam[1], t))
            raw, target = self._map(cam, f)
            moved = self._stabilize(target, t)
            self.last_t = t

            armed = (t - (self.shown_since or t)) >= cfg.pinch_arm_delay or self.pinch.pinched
            if armed and math.isfinite(ratio):
                before = self.pinch.pinched
                after = self.pinch.update(ratio, t, finger_curl(hand))
                if not after:
                    self._track_near_miss(ratio)
                if not before and after:
                    aim = self._aim_point(moved)
                    self.click_spot = aim if _dist(aim, moved) <= cfg.rewind_max_px else moved
                    self.drag_origin = moved  # drag distance is measured from the HAND, not the aim point
                    self.pinch_down_t = t
                    self._dip = None  # this attempt worked: not a near miss
                    event = "down"
                    log.info("gesture", "pinch_down", ratio=ratio, pos=self.click_spot,
                             rewind_px=_dist(self.click_spot, moved))
                elif before and not after:
                    event = "up"
                    log.info("gesture", "pinch_up", ratio=ratio, held_ms=_ms(t - (self.pinch_down_t or t)),
                             was_drag=self.click_spot is None)
                    # A click releases exactly where it pressed. Log 20260927-210532: a click pressed at
                    # x=1431 released at x=560 because the hand had just re-entered: macOS saw a drag.
                    self._release_at = self.click_spot
                    self.click_spot = None

            if self.click_spot is not None:
                origin = self.drag_origin or self.click_spot
                pull = _dist(moved, origin)
                age = t - (self.pinch_down_t or t)
                if (pull > cfg.drag_start_px and age >= cfg.drag_min_hold) or pull > cfg.drag_fast_px:
                    # Carry the click-spot/hand offset so the drag starts exactly at the click spot
                    self.offset = (self.click_spot[0] - origin[0], self.click_spot[1] - origin[1])
                    log.debug("gesture", "drag_start", held_ms=_ms(age), pull_px=pull)
                    self.click_spot = None
                    self.pos = (moved[0] + self.offset[0], moved[1] + self.offset[1])
                else:
                    self.pos = self.click_spot
            else:
                if not self.pinch.pinched:
                    # After a drag, ease the offset away so the pointer glides back onto the hand
                    self.offset = (self.offset[0] * 0.8, self.offset[1] * 0.8)
                    if abs(self.offset[0]) + abs(self.offset[1]) < 0.5:
                        self.offset = (0.0, 0.0)
                self.pos = (moved[0] + self.offset[0], moved[1] + self.offset[1])

            self.history.append((t, self.pos, ratio))
            while self.history and t - self.history[0][0] > cfg.rewind_max_s:
                self.history.popleft()

            s.visible, s.visibility, s.ratio = True, 1.0, ratio
            s.raw_x, s.raw_y = raw
            s.cam_x, s.cam_y = cam
        else:
            if self.lost_since is None:
                self.lost_since = t
            lost_for = t - self.lost_since
            s.visibility = 1.0
            if lost_for > cfg.coast:
                if not self.reported_lost:
                    log.info("hand", "lost", visible_ms=_ms(self.lost_since - (self.shown_since or self.lost_since)))
                    self.reported_lost = True
                if self.pinch.pinched:
                    event = "up"  # losing the hand always releases: never leave something grabbed
                    log.info("gesture", "pinch_up", reason="hand_lost")
                    self._release_at = self.click_spot
                s.visibility = max(0.0, 1 - (lost_for - cfg.coast) / cfg.fade)
                self.pinch.reset()
                self.click_spot = None
                self.history.clear()

        # Velocity for prediction; zero while pinned so it never drifts off a click
        if s.visible and self.click_spot is None and not self.locked and self.vel_t is not None and t > self.vel_t:
            dt = t - self.vel_t
            v = ((self.pos[0] - prev[0]) / dt, (self.pos[1] - prev[1]) / dt)
            self.vel = (self.vel[0] * 0.5 + v[0] * 0.5, self.vel[1] * 0.5 + v[1] * 0.5)
        else:
            self.vel = (0.0, 0.0)
        self.vel_t = t

        s.x, s.y = self.pos
        if event == "up" and self._release_at is not None:
            s.x, s.y = self._release_at  # a click (not a drag) lets go at the click spot
        self._release_at = None
        s.vx, s.vy = self.vel
        s.pinched = self.pinch.pinched
        s.holding_click = self.click_spot is not None
        s.event = event
        s.locked = self.locked
        return s

    # --- internals ---

    def _track_near_miss(self, ratio: float):
        """Fingers came close but not close enough to click: log how close, once per attempt."""
        if ratio < 0.26:
            self._dip = ratio if self._dip is None else min(self._dip, ratio)
        elif ratio > 0.3 and self._dip is not None:
            if self._dip >= self.cfg.pinch_instant:
                log.info("gesture", "pinch_near_miss", min_ratio=self._dip, enter_below=self.cfg.pinch_enter,
                         open_before=self.pinch.press_ref or None)
            self._dip = None

    def _is_glitch(self, hand: Hand, t: float) -> bool:
        if self.last_cam is None:
            return False
        lx, ly, lt = self.last_cam
        if not (0 < t - lt < 0.2):
            return False
        speed = _dist(palm_point(hand), (lx, ly)) / (t - lt)
        if speed <= self.cfg.max_jump_per_sec:
            return False
        self.glitches += 1
        if self.glitches <= 5 or self.glitches % 50 == 0:
            log.debug("hand", "glitch_rejected", speed=speed, count=self.glitches)
        return True

    def _map(self, cam, f):
        """A small box in camera space maps onto the whole screen. When the hand goes past an
        edge, the box slides with it: the pointer never gets stuck and you never reach back."""
        bw = self.cfg.box_width
        bh = bw * (self.H / self.W) * self.aspect
        if self.box is None or abs(self.box[2] - bw) > 1e-9:
            # Place it so the hand maps to where the pointer already is: no jump
            self.box = [f[0] - bw * self.pos[0] / self.W, f[1] - bh * self.pos[1] / self.H, bw, bh]
            log.debug("pointer", "box_placed", box_w=bw)
        bx, by = self.box[0], self.box[1]
        if f[0] < bx:
            bx = f[0]
        if f[0] > bx + bw:
            bx = f[0] - bw
        if f[1] < by:
            by = f[1]
        if f[1] > by + bh:
            by = f[1] - bh
        self.box = [bx, by, bw, bh]

        def to_screen(p):
            return (min(max((p[0] - bx) / bw, 0.0), 1.0) * self.W,
                    min(max((p[1] - by) / bh, 0.0), 1.0) * self.H)
        return to_screen(cam), to_screen(f)

    def _stabilize(self, target, t):
        """Still-lock: after a short pause the pointer pins until the hand clearly moves.
        Rubber band: wobble under band_px is ignored; beyond it the pointer follows exactly."""
        c = self.cfg
        dt = max(t - (self.last_t or t), 1e-3)
        if self.locked:
            if _dist(target, self.stable) < c.lock_radius_px:
                return self.stable
            self.locked = False
            self.still_since = None
        p = self.stable
        d = _dist(target, p)
        if d > c.band_px:
            k = (d - c.band_px) / d
            p = (p[0] + (target[0] - p[0]) * k, p[1] + (target[1] - p[1]) * k)
        speed = _dist(p, self.stable) / dt
        self.stable = p
        if speed < c.lock_speed_px:
            if self.still_since is None:
                self.still_since = t
            if t - self.still_since >= c.lock_after:
                self.locked = True
        else:
            self.still_since = None
        return p

    def _aim_point(self, fallback):
        """Where the pointer was right before the fingers started closing."""
        for (_, pos, ratio) in reversed(self.history):
            if ratio is not None and math.isfinite(ratio) and ratio >= self.cfg.open_ratio:
                return pos
        return self.history[0][1] if self.history else fallback
