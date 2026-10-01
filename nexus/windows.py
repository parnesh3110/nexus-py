"""The Nexus window and the first-launch setup guide, in native macOS style: an SF Symbol and a
large title on top (like Setup Assistant), System Settings style grouped rows, real switches,
green checkmarks. Everything follows light/dark mode.

Setup guide: Welcome -> Choose -> Permissions -> Try Jarvis -> Try hand control -> Done.
Nothing asks macOS for a permission before the Permissions step explains what it's for.
"""
import threading
import time

import objc
from AppKit import (NSApp, NSBackingStoreBuffered, NSBezierPath, NSButton, NSColor, NSGradient, NSImage,
                    NSImageView, NSProgressIndicator, NSSegmentedControl, NSSlider, NSView, NSWindow)
from Foundation import NSMakeRect, NSObject

from . import native as N
from . import permissions
from .log import log

TITLED, CLOSABLE, MINIATURIZABLE, FULL_SIZE_CONTENT = 1, 2, 4, 1 << 15
PALETTE = [(1.00, 0.33, 0.62), (0.62, 0.36, 1.00), (0.24, 0.55, 1.00), (0.20, 0.86, 0.93)]

HOW_TO = [
    ("hand.point.up.left.fill", "Point and pinch", "Your palm moves the pointer. Pinch to click, pinch and move to drag."),
    ("hand.raised.fingers.spread.fill", "Open Jarvis", "Fingertips together, then spread. Together again to close."),
    ("waveform", "Talk", "\u201cHey Jarvis, open Safari\u201d. While the panel is open, just talk."),
    ("text.bubble.fill", "Do more at once", "\u201cOpen Notes and write a shopping list with milk and eggs\u201d."),
]


def orb_image(size: int = 72) -> NSImage:
    img = NSImage.alloc().initWithSize_((size, size))
    img.lockFocus()
    oval = NSBezierPath.bezierPathWithOvalInRect_(((3, 3), (size - 6, size - 6)))
    NSGradient.alloc().initWithColors_([NSColor.colorWithSRGBRed_green_blue_alpha_(*c, 1) for c in PALETTE]
                                       ).drawInBezierPath_angle_(oval, 45)
    NSGradient.alloc().initWithStartingColor_endingColor_(
        NSColor.colorWithWhite_alpha_(1.0, 0.85), NSColor.colorWithWhite_alpha_(1.0, 0.0)
    ).drawInBezierPath_relativeCenterPosition_(oval, (-0.35, 0.35))
    img.unlockFocus()
    return img


class _WindowTarget(NSObject):
    # Unique name on purpose: Objective-C class names are global (see ui._PanelTarget)
    def initWithOwner_(self, owner):
        self = objc.super(_WindowTarget, self).init()
        if self is None:
            return None
        self.owner = owner
        return self

    def clicked_(self, sender):
        self.owner._clicked(int(sender.tag()), sender)


class _Window:
    """A native window with a transparent title bar and a callback per control."""

    W, H = 640, 620

    def __init__(self, title: str, show_title: bool = True):
        self.win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, self.W, self.H), TITLED | CLOSABLE | MINIATURIZABLE | FULL_SIZE_CONTENT,
            NSBackingStoreBuffered, False)
        self.win.setTitle_(title)
        self.win.setTitlebarAppearsTransparent_(True)
        if not show_title:
            self.win.setTitleVisibility_(1)  # hidden, like Setup Assistant
        self.win.setMovableByWindowBackground_(True)
        self.win.setReleasedWhenClosed_(False)
        self.win.center()
        self.root = self.win.contentView()
        self._callbacks, self._next_tag = {}, 100
        self.target = _WindowTarget.alloc().initWithOwner_(self)

    def _tag(self, fn) -> int:
        self._next_tag += 1
        self._callbacks[self._next_tag] = fn
        return self._next_tag

    def _clicked(self, tag, sender):
        fn = self._callbacks.get(tag)
        if fn:
            try:
                fn(sender)
            except Exception as e:
                log.error("ui", "control_failed", error=repr(e))

    def button(self, parent, title, x, y, w, fn, primary=False):
        return N.push_button(parent, title, x, y, w, self.target, self._tag(fn), primary)

    def switch(self, parent, x, y, on, fn):
        return N.switch(parent, x, y, on, self.target, self._tag(fn))

    def hero(self, parent, icon: str | None, title: str, subtitle: str, tint=None):
        """Big symbol (or the Nexus orb), large title, one line of explanation."""
        if icon is None:
            iv = NSImageView.imageViewWithImage_(orb_image(72))
            iv.setFrame_(NSMakeRect(self.W / 2 - 36, self.H - 118, 72, 72))
            parent.addSubview_(iv)
        else:
            N.symbol(parent, icon, self.W / 2 - 36, self.H - 118, 72, 52,
                     tint or N.color("controlAccentColor", "systemBlueColor"), "regular")
        N.label(parent, title, 40, self.H - 166, self.W - 80, 34, 26, N.BOLD, align=N.CENTER, wrap=False)
        if subtitle:
            N.label(parent, subtitle, 90, self.H - 214, self.W - 180, 42, 13, N.REGULAR,
                    N.color("secondaryLabelColor"), N.CENTER)

    def show(self):
        NSApp.activateIgnoringOtherApps_(True)
        self.win.makeKeyAndOrderFront_(None)

    @property
    def visible(self) -> bool:
        return bool(self.win.isVisible())

    def close(self):
        self.win.orderOut_(None)


def _check(iv, done: bool):
    N.set_symbol(iv, "checkmark.circle.fill" if done else "circle", 18,
                 N.color("systemGreenColor") if done else N.color("tertiaryLabelColor"))


class SetupWindow(_Window):
    """First-launch guide. `ctl` is the app Controller (on/off, counters, settings)."""

    GX = 80  # grouped boxes: left edge
    PERMS = [
        ("camera", "camera.fill", "Camera", "To see your hand. Frames never leave this Mac.", "hands"),
        ("microphone", "mic.fill", "Microphone", "To hear you. Audio never leaves this Mac.", "voice"),
        ("accessibility", "accessibility", "Accessibility", "To move the pointer, click and type for you.", "any"),
        ("automation", "gearshape.2.fill", "App control", "So Jarvis can use Music, Notes, Finder\u2026", "voice"),
        ("ai", "cpu", "Local AI", "Ollama + Qwen, for requests beyond simple commands.", "voice"),
    ]

    def __init__(self, ctl, on_finish):
        super().__init__("Set Up Nexus", show_title=False)
        self.ctl, self.on_finish = ctl, on_finish
        self.use_hands = ctl.hands is not None
        self.use_voice = ctl.jarvis is not None
        self.i = 0
        self.page = None
        self._status = {}
        self._poll = True
        self.perm_rows, self.checks = [], []
        threading.Thread(target=self._poll_status, name="nexus-setup-poll", daemon=True).start()
        self.back_btn = self.button(self.root, "Back", self.W - 262, 22, 110, lambda _: self.go(-1))
        self.next_btn = self.button(self.root, "Continue", self.W - 142, 22, 120, lambda _: self.go(1), primary=True)
        self.step_lbl = N.label(self.root, "", 28, 30, 200, 18, 11, N.REGULAR, N.color("tertiaryLabelColor"), wrap=False)
        self._show()

    # --- navigation ---

    def _steps(self):
        steps = ["welcome", "choose", "permissions"]
        if self.use_voice:
            steps.append("voice")
        if self.use_hands:
            steps.append("hands")
        return steps + ["done"]

    def go(self, delta):
        steps = self._steps()
        name = steps[self.i]
        if delta > 0 and name == "choose" and not (self.use_hands or self.use_voice):
            return
        if delta > 0 and name == "permissions":
            self.ctl.apply_choices(self.use_hands, self.use_voice)  # camera / mic start now, after the explanation
        if delta > 0 and name == "done":
            self._poll = False
            self.on_finish()
            return
        self.i = max(0, min(len(steps) - 1, self.i + delta))
        self._show()

    def _show(self):
        steps = self._steps()
        name = steps[self.i]
        if self.page is not None:
            self.page.removeFromSuperview()
        self.page = NSView.alloc().initWithFrame_(NSMakeRect(0, 70, self.W, self.H - 70))
        self.root.addSubview_(self.page)
        self.H_page = self.H - 70
        self.perm_rows, self.checks = [], []
        j, h = self.ctl.jarvis, self.ctl.hands
        self.base = (getattr(j, "wake_count", 0), getattr(j, "turns", 0), getattr(h, "pinches", 0),
                     getattr(h, "panel_opens", 0))
        self.seen_hand = False
        getattr(self, f"_page_{name}")(self.page)
        self.back_btn.setHidden_(self.i == 0)
        self.next_btn.setTitle_("Get Started" if self.i == 0 else ("Finish" if name == "done" else "Continue"))
        self.step_lbl.setStringValue_(f"Step {self.i + 1} of {len(steps)}")
        log.info("setup", "step", step=name)

    def _group(self, parent, top, rows, height):
        """A grouped box of `rows` rows, its top edge `top` points below the page top."""
        h = rows * height
        return N.group(parent, self.GX, self.H_page - top - h, self.W - 2 * self.GX, h), self.W - 2 * self.GX

    # hero() positions from the window height; pages are 70 pt shorter (the button bar)
    def _hero(self, parent, icon, title, subtitle, tint=None):
        saved, self.H = self.H, self.H_page
        try:
            self.hero(parent, icon, title, subtitle, tint)
        finally:
            self.H = saved

    # --- pages ---

    def _page_welcome(self, v):
        self._hero(v, None, "Welcome to Nexus",
                   "Control your Mac with your hand and your voice.\nEverything runs on this Mac. No account, no cloud.")
        g, w = self._group(v, 240, 3, 62)
        for k, (icon, title, sub) in enumerate([
                ("hand.point.up.left.fill", "Point and pinch", "Your hand moves the pointer. Pinch to click and drag."),
                ("waveform", "Talk to Jarvis", "Say \u201cHey Jarvis\u201d and ask for anything, even several steps."),
                ("bubble.left.and.bubble.right.fill", "The Jarvis panel", "Spread your fingers to open it and just talk.")]):
            N.row(g, k, 3, 62, icon, title, sub)

    def _page_choose(self, v):
        self._hero(v, "switch.2", "What do you want to use?", "Pick one or both. You can change this anytime.")
        g, w = self._group(v, 240, 2, 66)
        y, _ = N.row(g, 0, 2, 66, "hand.raised.fill", "Hand control", "Point, pinch and drag. Uses the camera.")
        sw = self.switch(g, w - 60, y + 21, self.use_hands, lambda b: setattr(self, "use_hands", bool(b.state())))
        if self.ctl.hands is None:
            sw.setEnabled_(False)
            sw.setState_(0)
            self.use_hands = False
        y, _ = N.row(g, 1, 2, 66, "mic.fill", "Jarvis", "Your voice assistant. Uses the microphone.")
        self.switch(g, w - 60, y + 21, self.use_voice, lambda b: setattr(self, "use_voice", bool(b.state())))

    def _page_permissions(self, v):
        self._hero(v, "lock.shield.fill", "A few permissions",
                   "Click Allow, then accept the macOS prompt. If you said no before,\nOpen Settings takes you to the right place.")
        wanted = [p for p in self.PERMS if p[4] == "any" or (p[4] == "hands" and self.use_hands)
                  or (p[4] == "voice" and self.use_voice)]
        g, w = self._group(v, 240, len(wanted), 56)
        for k, (kind, icon, title, sub, _) in enumerate(wanted):
            y, _ = N.row(g, k, len(wanted), 56, icon, title, sub)
            mark = N.symbol(g, "checkmark.circle.fill", w - 42, y + 16, 24, 18, N.color("systemGreenColor"))
            btn = self.button(g, "Allow", w - 132, y + 12, 112, lambda _, k=kind: permissions.request(k))
            self.perm_rows.append((kind, mark, btn))

    def _page_voice(self, v):
        self._hero(v, "waveform", "Try Jarvis", "Say \u201cHey Jarvis\u201d, wait for the pop, then say \u201copen Notes\u201d.")
        g, w = self._group(v, 240, 3, 54)
        y, _ = N.row(g, 0, 3, 54, "ear.fill", "How well Jarvis hears you")
        self.meter = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(w - 200, y + 18, 180, 18))
        self.meter.setIndeterminate_(False)
        self.meter.setMinValue_(0.0)
        self.meter.setMaxValue_(1.0)
        g.addSubview_(self.meter)
        for k, text in enumerate(["Say \u201cHey Jarvis\u201d", "Then say \u201copen Notes\u201d"], start=1):
            y, _ = N.row(g, k, 3, 54, "circle.dashed", text)
            self.checks.append(N.symbol(g, "circle", w - 42, y + 15, 24, 18, N.color("tertiaryLabelColor")))
        self.heard_lbl = N.label(v, "", self.GX, self.H_page - 420, self.W - 2 * self.GX, 20, 12, N.REGULAR,
                                 N.color("secondaryLabelColor"))
        self.voice_tip = N.label(v, "", self.GX, self.H_page - 460, self.W - 2 * self.GX, 36, 11, N.REGULAR,
                                 N.color("tertiaryLabelColor"))

    def _page_hands(self, v):
        self._hero(v, "hand.raised.fingers.spread.fill", "Try hand control",
                   "Sit about an arm\u2019s length away, hand up, palm facing the screen.")
        g, w = self._group(v, 236, 4, 56)
        y, _ = N.row(g, 0, 4, 56, "slider.horizontal.3", "Calibrate your hand",
                     "20 seconds. Makes pinch and the panel fit your hand.")
        self.calib_btn = self.button(g, "Calibrate\u2026", w - 152, y + 12, 112, lambda _: self.ctl.show_calibration())
        self.checks.append(N.symbol(g, "circle", w - 38, y + 16, 24, 18, N.color("tertiaryLabelColor")))
        for k, (icon, title, sub) in enumerate([
                ("hand.raised.fill", "Raise your hand", "The pointer follows your palm."),
                ("hand.point.up.left.fill", "Pinch", "A light tap of thumb and index finger: a click."),
                ("bubble.left.and.bubble.right.fill", "Open the Jarvis panel", "All fingertips together, then spread.")],
                start=1):
            y, _ = N.row(g, k, 4, 56, icon, title, sub)
            self.checks.append(N.symbol(g, "circle", w - 38, y + 16, 24, 18, N.color("tertiaryLabelColor")))

    def _page_done(self, v):
        self._hero(v, "checkmark.seal.fill", "You\u2019re all set", "", N.color("systemGreenColor"))
        g, w = self._group(v, 200, len(HOW_TO), 52)
        for k, (icon, title, sub) in enumerate(HOW_TO):
            N.row(g, k, len(HOW_TO), 52, icon, title, sub)
        s = self.ctl.settings()
        g2, w2 = self._group(v, 200 + len(HOW_TO) * 52 + 16, 2, 44)
        y, _ = N.row(g2, 0, 2, 44, "power", "Open Nexus when you log in")
        self.switch(g2, w2 - 60, y + 10, s["login"], lambda b: self.ctl.set_login(bool(b.state())))
        y, _ = N.row(g2, 1, 2, 44, "macwindow", "Show this window when Nexus opens")
        self.switch(g2, w2 - 60, y + 10, s["show_home"], lambda b: self.ctl.save(show_home=bool(b.state())))

    # --- live updates ---

    def _poll_status(self):
        while self._poll:
            for kind, *_ in self.PERMS:
                try:
                    self._status[kind] = permissions.status(kind)
                except Exception:
                    self._status[kind] = "unknown"
            time.sleep(0.8)

    def tick(self):
        if not self.visible:
            return
        name = self._steps()[self.i]
        j, h = self.ctl.jarvis, self.ctl.hands
        if name == "permissions":
            for kind, mark, btn in self.perm_rows:
                st = self._status.get(kind, "unknown")
                mark.setHidden_(st != "granted")
                btn.setHidden_(st == "granted")
                if kind == "ai":
                    btn.setTitle_({"missing": "Set Up", "ask": "Download 2 GB", "working": "Downloading\u2026"}
                                  .get(st, "Set Up"))
                    btn.setEnabled_(st != "working")
                else:
                    btn.setTitle_("Open Settings" if st == "denied" else "Allow")
        elif name == "voice" and j is not None and self.checks:
            self.meter.setDoubleValue_(min(1.0, getattr(j, "last_score", 0.0) / max(j.wake_threshold, 1e-3) * 0.8))
            _check(self.checks[0], getattr(j, "wake_count", 0) > self.base[0])
            _check(self.checks[1], getattr(j, "turns", 0) > self.base[1])
            self.heard_lbl.setStringValue_(f"Heard: \u201c{j.heard}\u201d" if j.heard else "")
            tip = ""
            if getattr(j, "wake_count", 0) <= self.base[0] and getattr(j, "near_misses", 0) > 0:
                tip = "Almost. Say it a little louder, with a short pause after \u201cHey Jarvis\u201d."
            if j.state == "starting":
                tip = "Jarvis is starting (the speech models take a few seconds)\u2026"
            elif j.paused:
                tip = "Jarvis is off. Go back and switch it on."
            self.voice_tip.setStringValue_(tip)
        elif name == "hands" and h is not None and self.checks:
            calibrated = bool(getattr(h.args, "calibration", None))
            _check(self.checks[0], calibrated)
            self.calib_btn.setTitle_("Redo\u2026" if calibrated else "Calibrate\u2026")
            self.seen_hand = self.seen_hand or getattr(h, "hand_visible", False)
            _check(self.checks[1], self.seen_hand)
            _check(self.checks[2], getattr(h, "pinches", 0) > self.base[2])
            _check(self.checks[3], getattr(h, "panel_opens", 0) > self.base[3])


class CalibrationWindow(_Window):
    """Four poses, ~2 s each: learns your pinch and spread so both work in the positions you use.
    Clicks and the panel gesture pause while it runs (the pinch pose must not click things)."""

    W, H = 560, 500

    def __init__(self, ctl):
        super().__init__("Calibrate Hand", show_title=False)
        self.ctl, self.cal, self.result, self.state = ctl, None, None, "intro"
        v, W, H = self.root, self.W, self.H
        self.icon = N.symbol(v, "hand.raised.fill", W / 2 - 36, H - 130, 72, 52,
                             N.color("controlAccentColor", "systemBlueColor"))
        self.title = N.label(v, "", 40, H - 178, W - 80, 34, 24, N.BOLD, align=N.CENTER, wrap=False)
        self.sub = N.label(v, "", 60, H - 228, W - 120, 46, 13, N.REGULAR, N.color("secondaryLabelColor"), N.CENTER)
        self.step = N.label(v, "", 40, H - 262, W - 80, 18, 11, N.REGULAR, N.color("tertiaryLabelColor"),
                            N.CENTER, wrap=False)
        self.bar = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(110, H - 296, W - 220, 16))
        self.bar.setIndeterminate_(False)
        self.bar.setMinValue_(0.0)
        self.bar.setMaxValue_(1.0)
        v.addSubview_(self.bar)
        self.hint = N.label(v, "", 60, H - 350, W - 120, 40, 12, N.REGULAR, N.color("secondaryLabelColor"), N.CENTER)
        self.cancel_btn = self.button(v, "Cancel", W - 282, 22, 120, lambda _: self._cancel())
        self.primary = self.button(v, "Start", W - 150, 22, 130, lambda _: self._primary(), primary=True)
        self._render()

    def _render(self):
        from .calibration import PHASES
        if self.state == "intro":
            N.set_symbol(self.icon, "hand.raised.fill", 52)
            self.title.setStringValue_("Calibrate your hand")
            self.sub.setStringValue_("Four quick poses, about 2 seconds each. Nexus learns what open, pinched, "
                                     "bunched and spread look like for your hand.")
            self.step.setStringValue_("Sit where you normally use Nexus, hand up, about an arm\u2019s length away.")
            self.bar.setHidden_(True)
            self.primary.setTitle_("Start")
        elif self.state == "running":
            name, icon, title, sub = self.cal.phase
            N.set_symbol(self.icon, icon, 52, N.color("controlAccentColor", "systemBlueColor"))
            self.title.setStringValue_(title)
            self.sub.setStringValue_(sub)
            self.step.setStringValue_(f"Pose {self.cal.i + 1} of {len(PHASES)}")
            self.bar.setHidden_(False)
            self.primary.setTitle_("Redo Pose")
        elif self.state == "done":
            c = self.result
            N.set_symbol(self.icon, "checkmark.circle.fill", 52, N.color("systemGreenColor"))
            self.title.setStringValue_("Calibrated")
            self.sub.setStringValue_(f"Pinch: touching {c['pinch_touch']:.2f}, open {c['pinch_open']:.2f}.  "
                                     f"Panel: bunched {c['spread_lo']:.2f}, spread {c['spread_hi']:.2f}.")
            self.step.setStringValue_("Try a few pinches and the panel gesture. Recalibrate anytime from the Nexus window.")
            self.bar.setHidden_(True)
            self.hint.setStringValue_("")
            self.primary.setTitle_("Done")
            self.cancel_btn.setTitle_("Redo")
        elif self.state == "failed":
            N.set_symbol(self.icon, "exclamationmark.triangle.fill", 52, N.color("systemOrangeColor"))
            self.title.setStringValue_("Let\u2019s try that again")
            self.sub.setStringValue_(self.error)
            self.step.setStringValue_("")
            self.bar.setHidden_(True)
            self.hint.setStringValue_("")
            self.primary.setTitle_("Try Again")

    def _primary(self):
        if self.state in ("intro", "failed"):
            self.cal = self.ctl.start_calibration()
            if self.cal is None:
                return
            self.state = "running"
        elif self.state == "running":
            self.cal.redo()
        elif self.state == "done":
            self.close()
            return
        self._render()

    def _cancel(self):
        if self.state == "done":  # "Redo"
            self.state = "intro"
            self.cancel_btn.setTitle_("Cancel")
            self._render()
            return
        self.ctl.finish_calibration(None)
        self.state = "intro"
        self.close()

    def tick(self):
        if self.state != "running":
            return
        if not self.visible:  # closed with the red button mid-way
            self.ctl.finish_calibration(None)
            self.state = "intro"
            return
        before = self.cal.i
        self.bar.setDoubleValue_(self.cal.progress)
        seen = getattr(self.ctl.hands, "hand_visible", False)
        self.hint.setStringValue_("" if seen else "Keep your hand in view of the camera\u2026")
        if self.cal.done:
            self.result, err = self.cal.result()
            if self.result:
                self.ctl.finish_calibration(self.result)
                self.state = "done"
            else:
                self.ctl.finish_calibration(None)
                self.error, self.state = err, "failed"
            self._render()
        elif self.cal.i != before or str(self.title.stringValue()) != self.cal.phase[2]:
            self._render()


class HomeWindow(_Window):
    """What you see when you open Nexus: status and switches (like a System Settings page), how
    to use it, and buttons for the panel, setup and logs."""

    W, H = 600, 800
    GX = 40
    ROW, HOWTO_ROW = 44, 44

    def __init__(self, ctl):
        super().__init__("Nexus")
        self.ctl = ctl
        v = self.root
        iv = NSImageView.imageViewWithImage_(orb_image(64))
        iv.setFrame_(NSMakeRect(self.W / 2 - 32, self.H - 104, 64, 64))
        v.addSubview_(iv)
        N.label(v, "Nexus", 40, self.H - 146, self.W - 80, 32, 24, N.BOLD, align=N.CENTER, wrap=False)
        self.status_lbl = N.label(v, "", 40, self.H - 170, self.W - 80, 18, 12, N.REGULAR,
                                  N.color("secondaryLabelColor"), N.CENTER, wrap=False)
        w = self.W - 2 * self.GX

        N.label(v, "Controls", self.GX + 4, self.H - 208, 200, 18, 13, N.SEMIBOLD, wrap=False)
        top, R, n = self.H - 214, self.ROW, 6
        g = N.group(v, self.GX, top - n * R, w, n * R)
        y, _ = N.row(g, 0, n, R, "hand.raised.fill", "Hand control", "Camera")
        self.hands_sw = self.switch(g, w - 60, y + 10, ctl.hands_on(), lambda b: ctl.set_hands(bool(b.state())))
        y, _ = N.row(g, 1, n, R, "mic.fill", "Jarvis", "Microphone")
        self.voice_sw = self.switch(g, w - 60, y + 10, ctl.voice_on(), lambda b: ctl.set_voice(bool(b.state())))
        y, _ = N.row(g, 2, n, R, "cursorarrow.motionlines", "Pointer speed", "How far the pointer moves with your hand")
        self.speed = NSSegmentedControl.segmentedControlWithLabels_trackingMode_target_action_(
            ["Slow", "Medium", "Fast"], 0, self.target, "clicked:")
        self.speed.setFrame_(NSMakeRect(w - 226, y + 9, 206, 26))
        self.speed.setTag_(self._tag(lambda s: ctl.set_sensitivity(("low", "medium", "high")[int(s.selectedSegment())])))
        self.speed.setSelectedSegment_({"low": 0, "medium": 1, "high": 2}.get(ctl.settings()["sensitivity"], 1))
        g.addSubview_(self.speed)
        y, _ = N.row(g, 3, n, R, "hand.point.up.left.fill", "Hand calibration", "Fits pinch and the panel gesture to you")
        self.calib_lbl = N.label(g, "", w - 258, y + 13, 118, 18, 12, N.REGULAR, N.color("secondaryLabelColor"),
                                 N.RIGHT, wrap=False)
        self.calib_btn = self.button(g, "Calibrate\u2026", w - 132, y + 6, 114, lambda _: ctl.show_calibration())
        y, _ = N.row(g, 4, n, R, "ear.fill", "\u201cHey Jarvis\u201d sensitivity", "Higher hears you more easily")
        threshold = ctl.jarvis.wake_threshold if ctl.jarvis is not None else 0.35
        self.slider = NSSlider.sliderWithValue_minValue_maxValue_target_action_(0.8 - threshold, 0.2, 0.6,
                                                                                self.target, "clicked:")
        self.slider.setFrame_(NSMakeRect(w - 190, y + 10, 170, 24))
        self.slider.setTag_(self._tag(lambda s: ctl.set_wake_threshold(round(0.8 - float(s.doubleValue()), 2))))
        g.addSubview_(self.slider)
        y, _ = N.row(g, 5, n, R, "waveform.circle.fill", "Microphone in use")
        self.mic_lbl = N.label(g, "", w - 250, y + 12, 230, 20, 12, N.REGULAR, N.color("secondaryLabelColor"),
                               N.RIGHT, wrap=False)

        HR = self.HOWTO_ROW
        N.label(v, "How to use", self.GX + 4, top - n * R - 34, 200, 18, 13, N.SEMIBOLD, wrap=False)
        top2 = top - n * R - 40
        g2 = N.group(v, self.GX, top2 - len(HOW_TO) * HR, w, len(HOW_TO) * HR)
        for k, (icon, title, sub) in enumerate(HOW_TO):
            N.row(g2, k, len(HOW_TO), HR, icon, title, sub)

        self.button(v, "Open Jarvis", self.W - self.GX - 150, 22, 150, lambda _: ctl.toggle_panel(), primary=True)
        self.button(v, "Setup & Permissions\u2026", self.W - self.GX - 350, 22, 192, lambda _: ctl.show_setup())
        self.button(v, "Logs", self.GX, 22, 80, lambda _: ctl.open_logs())
        box = NSButton.checkboxWithTitle_target_action_("Show this window when Nexus opens", self.target, "clicked:")
        box.setFrame_(NSMakeRect(self.GX + 2, 62, 300, 20))
        box.setState_(1 if ctl.settings()["show_home"] else 0)
        box.setTag_(self._tag(lambda b: ctl.save(show_home=bool(b.state()))))
        v.addSubview_(box)
        self._last = None

    def tick(self):
        if not self.visible:
            return
        c, j = self.ctl, self.ctl.jarvis
        if j is None:
            status = "Jarvis unavailable"
        elif not c.voice_on():
            status = "Jarvis is off"
        elif j.hold_open:
            status = "In a conversation (panel open)"
        else:
            status = {"idle": "Listening for \u201cHey Jarvis\u201d", "starting": "Starting\u2026"}.get(j.state, j.state.capitalize())
        if c.hands is None:
            status += " \u00b7 hand control unavailable (allow Camera in Setup)"
        mic = getattr(getattr(j, "mic", None), "device_name", "") if j is not None else ""
        cal = getattr(getattr(c.hands, "args", None), "calibration", None)
        calib = f"Done {cal['date']}" if cal else ("Not yet" if c.hands is not None else "\u2014")
        key = (status, c.hands_on(), c.voice_on(), mic, calib)
        if key == self._last:
            return
        self._last = key
        self.status_lbl.setStringValue_(status)
        self.hands_sw.setState_(1 if c.hands_on() else 0)
        self.hands_sw.setEnabled_(c.hands is not None)
        self.voice_sw.setState_(1 if c.voice_on() else 0)
        self.voice_sw.setEnabled_(j is not None)
        self.mic_lbl.setStringValue_(mic or "\u2014")
        self.calib_lbl.setStringValue_(calib)
        self.calib_btn.setEnabled_(c.hands is not None)
        self.calib_btn.setTitle_("Recalibrate\u2026" if cal else "Calibrate\u2026")
