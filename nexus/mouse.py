"""The real macOS cursor: moves, clicks, double-clicks, drags (Quartz events via PyObjC),
plus a 120 Hz animator that glides the cursor between camera frames.

Posting clicks needs Accessibility permission for the app that runs Python (your terminal):
System Settings -> Privacy & Security -> Accessibility.
"""
import math
import threading
import time

import Quartz

from .config import Config
from .log import log

try:
    from ApplicationServices import AXIsProcessTrusted, AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
except Exception:  # older PyObjC layouts
    AXIsProcessTrusted = AXIsProcessTrustedWithOptions = kAXTrustedCheckOptionPrompt = None


def screen_size() -> tuple[float, float]:
    b = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return float(b.size.width), float(b.size.height)


def accessibility_trusted(prompt: bool = False) -> bool:
    if AXIsProcessTrustedWithOptions is None:
        return True
    if prompt:
        return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}))
    return bool(AXIsProcessTrusted())


class Mouse:
    DOUBLE_CLICK_S = 0.4
    DOUBLE_CLICK_PX = 12

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.down = False
        self._last = None
        self._last_up = (0.0, (0.0, 0.0))
        self._clicks = 1
        self._src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        # Otherwise macOS ignores the real trackpad for 250ms after each posted event
        Quartz.CGEventSourceSetLocalEventsSuppressionInterval(self._src, 0.0)

    def _post(self, kind, x, y):
        e = Quartz.CGEventCreateMouseEvent(self._src, kind, (x, y), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventSetIntegerValueField(e, Quartz.kCGMouseEventClickState, self._clicks)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)

    def move(self, x: float, y: float):
        if not self.enabled:
            return
        if self._last and math.hypot(x - self._last[0], y - self._last[1]) < 0.5:
            return
        self._last = (x, y)
        self._post(Quartz.kCGEventLeftMouseDragged if self.down else Quartz.kCGEventMouseMoved, x, y)

    def press(self, x: float, y: float):
        if not self.enabled:
            return
        now = time.perf_counter()
        t_up, p_up = self._last_up
        near = math.hypot(x - p_up[0], y - p_up[1]) < self.DOUBLE_CLICK_PX
        self._clicks = self._clicks + 1 if (now - t_up < self.DOUBLE_CLICK_S and near) else 1
        self._post(Quartz.kCGEventLeftMouseDown, x, y)
        self.down = True
        self._last = (x, y)
        log.info("mouse", "down", pos=(x, y), click_count=self._clicks)

    def release(self, x: float, y: float):
        if not self.down:
            return
        self._post(Quartz.kCGEventLeftMouseUp, x, y)
        self.down = False
        self._last_up = (time.perf_counter(), (x, y))
        log.info("mouse", "up", pos=(x, y), click_count=self._clicks)

    def release_if_down(self):
        if self.down and self._last:
            self.release(*self._last)


def _smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


class CursorAnimator(threading.Thread):
    """Runs at display rate between camera frames (30 fps): predicts along the velocity for
    clear motion only (on a still hand it would amplify tremor), and eases toward the target
    with a speed-dependent time constant (slow hand = silky, fast hand = snappy)."""

    def __init__(self, mouse: Mouse | None, cfg: Config):
        super().__init__(name="nexus-cursor", daemon=True)
        self.mouse, self.cfg = mouse, cfg
        self.state = None
        self.shown = None
        self.output = None
        self.fps = 0.0
        self._running = True

    def push(self, state):
        """New camera-rate state. On a click, jump straight to the exact spot."""
        self.state = state
        if state.event or state.holding_click:
            self.shown = (state.x, state.y)
            self.output = self.shown

    def stop(self):
        self._running = False

    def run(self):
        c = self.cfg
        period = 1.0 / c.render_hz
        last = time.perf_counter()
        next_tick = last
        ticks, window = 0, last
        while self._running:
            now = time.perf_counter()
            dt = min(now - last, 0.05)
            last = now
            s = self.state
            if s is not None and s.visible:
                speed = math.hypot(s.vx, s.vy)
                target = (s.x, s.y)
                if not s.holding_click:
                    amount = _smoothstep(c.predict_from, c.predict_full, speed)
                    lead = min(max(now - s.t, 0.0), c.predict_lead) * amount
                    dx, dy = s.vx * lead, s.vy * lead
                    n = math.hypot(dx, dy)
                    if n > c.predict_max_px:
                        dx, dy = dx * c.predict_max_px / n, dy * c.predict_max_px / n
                    target = (s.x + dx, s.y + dy)
                p = self.shown or target
                if not s.holding_click:
                    f = _smoothstep(c.slow_speed, c.fast_speed, speed)
                    tau = c.tau_slow + (c.tau_fast - c.tau_slow) * f
                    k = 1 - math.exp(-dt / tau)
                    p = (p[0] + (target[0] - p[0]) * k, p[1] + (target[1] - p[1]) * k)
                self.shown = self.output = p
                if self.mouse:
                    self.mouse.move(*p)
            ticks += 1
            if now - window >= 1:
                self.fps, ticks, window = ticks / (now - window), 0, now
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()  # fell behind: don't try to catch up
