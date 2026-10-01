"""The Jarvis panel (a native-looking chat popover), the menu bar icon, the Dock/app-menu wiring,
and the main-thread event loop. Native AppKit via PyObjC.

The panel: translucent like Spotlight/Siri, Messages-style bubbles (you on the right in blue,
Jarvis on the left), a typing indicator while Jarvis thinks, a live mic level while it listens,
and a text field to type instead of talking. How open it is follows your five fingers
(SpreadControl): 80% closed fingers = 80% closed panel. Pinch on its header and drag to move it.
"""
import math
import signal
import time

import objc
from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, NSBox, NSButton,
                    NSColor, NSEvent, NSFont, NSPanel, NSRunLoop, NSRunLoopCommonModes, NSScreen, NSScrollView,
                    NSStatusWindowLevel, NSTextField, NSTimer, NSView, NSVisualEffectView,
                    NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
                    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel)
from Foundation import NSMakePoint, NSMakeRect, NSObject
from Quartz import (CAGradientLayer, CALayer, CATransaction, CATransform3DMakeRotation, CATransform3DMakeTranslation,
                    CATransform3DScale, CATransform3DTranslate)

from . import native as N
from .log import log

POPOVER_MATERIAL, BEHIND_WINDOW, ACTIVE = 6, 0, 1  # NSVisualEffectView enums
PALETTE = [(1.00, 0.33, 0.62), (0.62, 0.36, 1.00), (0.24, 0.55, 1.00), (0.20, 0.86, 0.93), (1.00, 0.62, 0.24)]


def _cg(r, g, b, a=1.0):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, a).CGColor()


class _KeyPanel(NSPanel):
    """Borderless panels can't take keyboard focus by default; this one can, so you can type in it.
    It only becomes key when you click into it, so opening it never steals your typing."""

    def canBecomeKeyWindow(self):
        return True


class _Flipped(NSView):
    """Top-to-bottom coordinates, for laying out chat bubbles."""

    def isFlipped(self):
        return True


class _PanelTarget(NSObject):
    # Objective-C class names are global across the whole app: every NSObject subclass in Nexus
    # needs a unique name (a second "_Target" in windows.py crashed the app on launch)
    def initWithOwner_(self, owner):
        self = objc.super(_PanelTarget, self).init()
        if self is None:
            return None
        self.owner = owner
        return self

    def clicked_(self, sender):
        try:
            self.owner._clicked(int(sender.tag()), sender)
        except Exception as e:
            log.error("ui", "control_failed", error=repr(e))


STATUS = {"idle": "Say \u201cHey Jarvis\u201d, or type below", "listening": "Listening\u2026",
          "thinking": "Thinking\u2026", "speaking": "Speaking\u2026", "starting": "Starting up\u2026",
          "off": "Jarvis is off \u00b7 you can still type"}


class JarvisPanel:
    W, H = 380, 560
    HEADER, FOOTER = 58, 64
    SEND, CLOSE = 1, 2

    def __init__(self, jarvis, spread, auto_show: bool = False, on_close=None):
        self.jarvis, self.spread, self.auto_show = jarvis, spread, auto_show
        self.on_close = on_close or (lambda: None)
        self.pinned = False          # opened from the menu bar / window, regardless of the gesture
        vf = NSScreen.mainScreen().visibleFrame()
        x = vf.origin.x + vf.size.width - self.W - 18
        y = vf.origin.y + vf.size.height - self.H - 10
        p = _KeyPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, self.W, self.H), NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        p.setOpaque_(False)
        p.setBackgroundColor_(NSColor.clearColor())
        p.setHasShadow_(True)
        p.setLevel_(NSStatusWindowLevel)
        p.setMovableByWindowBackground_(True)
        p.setFloatingPanel_(True)
        p.setBecomesKeyOnlyIfNeeded_(True)
        p.setHidesOnDeactivate_(False)
        p.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary)
        p.setAlphaValue_(0.0)
        self.target = _PanelTarget.alloc().initWithOwner_(self)

        root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, self.W, self.H))
        root.setWantsLayer_(True)
        glass = NSVisualEffectView.alloc().initWithFrame_(root.bounds())
        glass.setMaterial_(POPOVER_MATERIAL)       # follows light/dark mode, like Spotlight
        glass.setBlendingMode_(BEHIND_WINDOW)
        glass.setState_(ACTIVE)
        glass.setWantsLayer_(True)
        glass.layer().setCornerRadius_(16)
        glass.layer().setMasksToBounds_(True)
        glass.layer().setBorderWidth_(0.5)
        glass.layer().setBorderColor_(_cg(0.5, 0.5, 0.5, 0.35))
        root.addSubview_(glass)
        p.setContentView_(root)
        self.panel, self.root, self.glass = p, root, glass
        self._build_header()
        self._build_messages()
        self._build_footer()
        self.shown, self.visible, self.phase = 0.0, False, 0.0
        self._last_key = None

    # --- building ---

    def _build_header(self):
        g, W, H = self.glass, self.W, self.H
        self.orb = CAGradientLayer.layer()
        self.orb.setType_("conic")
        self.orb.setFrame_(((16, H - 44), (30, 30)))
        self.orb.setCornerRadius_(15)
        self.orb.setStartPoint_((0.5, 0.5))
        self.orb.setEndPoint_((0.5, 0.0))
        self.orb.setColors_([_cg(*c) for c in PALETTE + [PALETTE[0]]])
        g.layer().addSublayer_(self.orb)
        N.label(g, "Jarvis", 56, H - 30, 200, 18, 14, N.SEMIBOLD, wrap=False)
        self.status = N.label(g, "", 56, H - 48, W - 110, 16, 11, N.REGULAR, N.color("secondaryLabelColor"), wrap=False)
        close = NSButton.alloc().initWithFrame_(NSMakeRect(W - 38, H - 40, 22, 22))
        close.setBordered_(False)
        close.setImage_(N.symbol_image("xmark.circle.fill", 17))
        close.setContentTintColor_(N.color("tertiaryLabelColor"))
        close.setTarget_(self.target)
        close.setAction_("clicked:")
        close.setTag_(self.CLOSE)
        close.setToolTip_("Close (or bring your fingertips together)")
        g.addSubview_(close)
        N.separator(g, 0, H - self.HEADER, W)

    def _build_messages(self):
        h = self.H - self.HEADER - self.FOOTER
        self.scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, self.FOOTER, self.W, h))
        self.scroll.setDrawsBackground_(False)
        self.scroll.setHasVerticalScroller_(True)
        self.scroll.setAutohidesScrollers_(True)
        self.doc = _Flipped.alloc().initWithFrame_(NSMakeRect(0, 0, self.W, h))
        self.scroll.setDocumentView_(self.doc)
        self.glass.addSubview_(self.scroll)

    def _build_footer(self):
        g, W = self.glass, self.W
        N.separator(g, 0, self.FOOTER - 1, W)
        self.bars = []
        for i in range(4):  # live mic level while listening
            bar = CALayer.layer()
            bar.setBackgroundColor_(N.color("controlAccentColor", "systemBlueColor").CGColor())
            bar.setCornerRadius_(1.5)
            bar.setFrame_(((16 + i * 6, 30), (3, 4)))
            g.layer().addSublayer_(bar)
            self.bars.append(bar)
        self.field = NSTextField.alloc().initWithFrame_(NSMakeRect(48, 17, W - 48 - 50, 30))
        self.field.setBezeled_(True)
        self.field.setBezelStyle_(1)   # rounded, like Messages
        self.field.setPlaceholderString_("Message Jarvis")
        self.field.setFont_(NSFont.systemFontOfSize_(13))
        self.field.setTarget_(self.target)
        self.field.setAction_("clicked:")
        self.field.setTag_(self.SEND)
        g.addSubview_(self.field)
        send = NSButton.alloc().initWithFrame_(NSMakeRect(W - 42, 18, 28, 28))
        send.setBordered_(False)
        send.setImage_(N.symbol_image("arrow.up.circle.fill", 22))
        send.setContentTintColor_(N.color("controlAccentColor", "systemBlueColor"))
        send.setTarget_(self.target)
        send.setAction_("clicked:")
        send.setTag_(self.SEND)
        g.addSubview_(send)

    # --- actions ---

    def _clicked(self, tag, sender):
        if tag == self.SEND:
            text = str(self.field.stringValue()).strip()
            if text and self.jarvis is not None:
                self.jarvis.submit_text(text)
                self.field.setStringValue_("")
        elif tag == self.CLOSE:
            self.on_close()

    # --- every frame ---

    def _target(self) -> float:
        t = self.spread.target if self.spread is not None else 0.0
        if self.pinned:
            t = 1.0
        if self.auto_show and self.jarvis is not None and self.jarvis.state not in ("idle", "starting", "off"):
            t = max(t, 1.0)  # no hand control: show while Jarvis is busy
        return t

    def tick(self, dt: float):
        target = self._target()
        following = self.spread is not None and self.spread.session
        tau = 0.045 if following else 0.16  # snappy while your fingers drive it, soft when it settles
        self.shown += (target - self.shown) * (1 - math.exp(-dt / tau))
        a = self.shown
        if a < 0.004:
            if self.visible:
                self.panel.orderOut_(None)
                self.visible = False
            return
        if not self.visible:
            self.panel.orderFrontRegardless()  # visible, but not key: your typing stays in your app
            self.visible = True
        ease = a * a * (3 - 2 * a)
        state = self.jarvis.state if self.jarvis is not None else "off"
        # While your fingers drive it, show it clearly from the first moment (small, but visible), so
        # you can see the gesture was recognized; the size still follows your fingers exactly.
        alpha = max(ease, 0.3) if following else ease
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        self.panel.setAlphaValue_(alpha)
        s = 0.86 + 0.14 * ease  # grows out of its top edge
        t = CATransform3DMakeTranslation(self.W / 2, self.H, 0)
        t = CATransform3DScale(t, s, s, 1)
        t = CATransform3DTranslate(t, -self.W / 2, -self.H, 0)
        self.root.layer().setSublayerTransform_(t)
        speed = {"listening": 1.2, "thinking": 3.0, "speaking": 0.8}.get(state, 0.3)
        self.phase += dt * speed * 2 * math.pi
        pulse = 1.0 + (0.06 * math.sin(self.phase * 0.8) if state in ("listening", "speaking") else 0.0)
        self.orb.setTransform_(CATransform3DScale(CATransform3DMakeRotation(self.phase, 0, 0, 1), pulse, pulse, 1))
        level = 0.0
        if state == "listening" and self.jarvis is not None and getattr(self.jarvis, "mic", None) is not None:
            level = min(1.0, self.jarvis.mic.level * 35)
        for i, bar in enumerate(self.bars):
            wobble = 0.55 + 0.45 * math.sin(self.phase * 2.3 + i * 1.3)
            h = 4 + (16 * level * wobble if state == "listening" else 0)
            bar.setFrame_(((16 + i * 6, 32 - h / 2), (3, h)))
            bar.setOpacity_(1.0 if state == "listening" else 0.25)
        CATransaction.commit()
        self._refresh(state)

    def _refresh(self, state: str):
        j = self.jarvis
        status = STATUS.get(state, "")
        if j is not None and j.dictating:
            status = "Dictating \u00b7 say \u201cstop dictation\u201d to finish"
        elif j is not None and state == "idle" and j.hold_open:
            status = "Listening\u2026"
        turns = list(j.conversation)[-20:] if j is not None else []
        key = (status, tuple(turns), state == "thinking")
        if key == self._last_key:
            return
        self._last_key = key
        self.status.setStringValue_(status)
        self._layout(turns, typing=(state == "thinking"))

    def _layout(self, turns, typing: bool):
        for v in list(self.doc.subviews()):
            v.removeFromSuperview()
        width = self.W - 16
        visible_h = self.scroll.contentSize().height
        if not turns and not typing:
            self.doc.setFrame_(NSMakeRect(0, 0, width, visible_h))
            N.label(self.doc, "Ask Jarvis anything.\nSay it, or type below.", 20, visible_h / 2 - 20, width - 40, 40,
                    13, N.REGULAR, N.color("secondaryLabelColor"), N.CENTER)
            return
        y = 12.0
        items = [(r, t) for r, t in turns] + ([("typing", "\u2022 \u2022 \u2022")] if typing else [])
        for role, text in items:
            mine = role == "you"
            max_w = width * 0.78
            lbl = NSTextField.wrappingLabelWithString_(text)
            lbl.setFont_(NSFont.systemFontOfSize_(13))
            lbl.setTextColor_(NSColor.whiteColor() if mine else
                              (N.color("secondaryLabelColor") if role == "typing" else N.color("labelColor")))
            lbl.setSelectable_(role != "typing")
            size = lbl.cell().cellSizeForBounds_(NSMakeRect(0, 0, max_w - 24, 10000))
            bw, bh = min(max_w, size.width + 26), size.height + 16
            bubble = NSBox.alloc().initWithFrame_(NSMakeRect(width - bw - 8 if mine else 10, y, bw, bh))
            bubble.setBoxType_(4)
            bubble.setTitlePosition_(0)
            bubble.setBorderWidth_(0)
            bubble.setCornerRadius_(15)
            bubble.setContentViewMargins_((0, 0))
            bubble.setFillColor_(N.color("systemBlueColor") if mine else
                                 N.color("tertiarySystemFillColor", "controlBackgroundColor"))
            lbl.setFrame_(NSMakeRect(13, 8, bw - 26, size.height))
            bubble.contentView().addSubview_(lbl)
            self.doc.addSubview_(bubble)
            y += bh + 8
        doc_h = max(y + 4, visible_h)
        self.doc.setFrame_(NSMakeRect(0, 0, width, doc_h))
        clip = self.scroll.contentView()
        clip.scrollToPoint_(NSMakePoint(0, max(0.0, doc_h - visible_h)))  # newest message in view
        self.scroll.reflectScrolledClipView_(clip)


# --- menu bar icon ---

class _MenuTarget(NSObject):
    """Receives menu clicks and forwards them to plain Python callbacks."""

    def initWithHandlers_(self, handlers):
        self = objc.super(_MenuTarget, self).init()
        if self is None:
            return None
        self.handlers = handlers
        return self

    def clicked_(self, item):
        fn = self.handlers.get(item.representedObject())
        if fn:
            try:
                fn(item)
            except Exception as e:
                log.error("ui", "menu_failed", item=str(item.title()), error=repr(e))


class StatusMenu:
    """Menu bar icon. `items`: (key, title, checked or None, callback(item) -> new checked state or
    None), or None for a separator."""

    def __init__(self, items):
        from AppKit import NSImage, NSMenu, NSMenuItem, NSStatusBar, NSVariableStatusItemLength
        self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        icon = NSImage.imageWithSystemSymbolName_accessibilityDescription_("hand.point.up.left.fill", "Nexus")
        if icon is not None:
            icon.setTemplate_(True)
            self.item.button().setImage_(icon)
        else:
            self.item.button().setTitle_("\u25c9")
        handlers = {}

        def wrap(cb):
            def run(menu_item):
                result = cb(menu_item)
                if result is not None:
                    menu_item.setState_(1 if result else 0)
            return run

        self.target = _MenuTarget.alloc().initWithHandlers_(handlers)
        self.by_key = {}
        menu = NSMenu.alloc().init()
        for entry in items:
            if entry is None:
                menu.addItem_(NSMenuItem.separatorItem())
                continue
            key, title, checked, cb = entry
            mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "clicked:", "q" if key == "quit" else "")
            mi.setTarget_(self.target)
            mi.setRepresentedObject_(key)
            if checked is not None:
                mi.setState_(1 if checked else 0)
            handlers[key] = wrap(cb)
            self.by_key[key] = mi
            menu.addItem_(mi)
        self.item.setMenu_(menu)

    def set_checked(self, key: str, on: bool):
        if key in self.by_key:
            self.by_key[key].setState_(1 if on else 0)


# --- Dock and the app menu (Nexus.app) ---

class _AppDelegate(NSObject):
    """Dock icon click / opening Nexus again -> show its window. A quit that comes from macOS
    (Dock menu, logging out): release the camera and mic, then let it quit right away."""

    def initWithReopen_terminate_(self, on_reopen, on_terminate):
        self = objc.super(_AppDelegate, self).init()
        if self is None:
            return None
        self.on_reopen, self.on_terminate = on_reopen, on_terminate
        return self

    def applicationShouldHandleReopen_hasVisibleWindows_(self, app, has_windows):
        self.on_reopen()
        return True

    def applicationShouldTerminate_(self, app):
        try:
            self.on_terminate()
        except Exception as e:
            log.error("ui", "terminate_cleanup_failed", error=repr(e))
        return 1  # NSTerminateNow. Returning "cancel" here is what made Quit loop forever.


def install_app_menu_and_dock(ctl):
    """For Nexus.app: a Nexus menu in the menu bar, and Dock behavior. Returns objects to keep alive."""
    from AppKit import NSMenu, NSMenuItem
    delegate = _AppDelegate.alloc().initWithReopen_terminate_(ctl.show_home, ctl.shutdown)
    NSApplication.sharedApplication().setDelegate_(delegate)
    handlers = {"home": lambda _: ctl.show_home(), "setup": lambda _: ctl.show_setup(),
                "panel": lambda _: ctl.toggle_panel(), "quit": lambda _: ctl.quit(),
                "about": lambda _: ctl.about(), "calibrate": lambda _: ctl.show_calibration()}
    target = _MenuTarget.alloc().initWithHandlers_(handlers)
    bar, app_item = NSMenu.alloc().init(), NSMenuItem.alloc().init()
    bar.addItem_(app_item)
    menu = NSMenu.alloc().initWithTitle_("Nexus")
    for key, title, shortcut in (("about", "About Nexus", ""), (None, None, None),
                                 ("home", "Open Nexus", "0"), ("setup", "Setup & Permissions\u2026", ","),
                                 ("calibrate", "Calibrate Hand\u2026", "k"), ("panel", "Show or Hide Jarvis", "j"),
                                 (None, None, None), ("hide", "Hide Nexus", "h"), ("quit", "Quit Nexus", "q")):
        if key is None:
            menu.addItem_(NSMenuItem.separatorItem())
            continue
        if key == "hide":
            menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "hide:", shortcut))
            continue
        mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "clicked:", shortcut)
        mi.setTarget_(target)  # our own quit, not NSApp.terminate: a clean shutdown, then the loop stops
        mi.setRepresentedObject_(key)
        menu.addItem_(mi)
    app_item.setSubmenu_(menu)
    edit_item = NSMenuItem.alloc().init()  # so copy/paste/select-all work in the Jarvis text field
    bar.addItem_(edit_item)
    edit = NSMenu.alloc().initWithTitle_("Edit")
    for title, action, key in (("Cut", "cut:", "x"), ("Copy", "copy:", "c"), ("Paste", "paste:", "v"),
                               ("Select All", "selectAll:", "a")):
        edit.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key))
    edit_item.setSubmenu_(edit)
    NSApplication.sharedApplication().setMainMenu_(bar)
    return delegate, target, bar


# --- event loop ---

class _Ticker(NSObject):
    def initWithCallback_(self, callback):
        self = objc.super(_Ticker, self).init()
        if self is None:
            return None
        self.callback = callback
        self.last = time.perf_counter()
        return self

    def tick_(self, _timer):
        now = time.perf_counter()
        dt, self.last = min(now - self.last, 0.05), now
        try:
            self.callback(dt)
        except Exception as e:
            log.error("ui", "tick_failed", error=repr(e))


def init_app(regular: bool = False):
    """regular=True (Nexus.app): Dock icon, app switcher, its own menu bar. Otherwise (running from
    Terminal) an accessory app: no Dock icon, which would show up as "python3.11"."""
    from AppKit import NSApplicationActivationPolicyRegular
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular if regular else NSApplicationActivationPolicyAccessory)
    return app


def run(tick, should_stop, fps: int = 60):
    """Main-thread event loop. `tick(dt)` runs every frame. Returns when `should_stop()` or Ctrl+C.

    Stops with NSApp.stop (the loop returns to Python, and the normal shutdown code runs), not
    NSApp.terminate (which exits the process on the spot). Mixing the two made Quit loop forever."""
    app = NSApplication.sharedApplication()
    interrupted = {"flag": False, "stopping": False}
    try:
        signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__("flag", True))
    except ValueError:
        pass  # not the main thread

    def frame(dt):
        if interrupted["stopping"]:
            return
        if interrupted["flag"] or should_stop():
            interrupted["stopping"] = True
            app.stop_(None)
            # stop only takes effect after one more event: post an empty one
            ev = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                15, NSMakePoint(0, 0), 0, 0.0, 0, None, 0, 0, 0)  # 15 = NSEventTypeApplicationDefined
            app.postEvent_atStart_(ev, True)
            return
        tick(dt)

    ticker = _Ticker.alloc().initWithCallback_(frame)
    timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(1.0 / fps, ticker, "tick:", None, True)
    NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
    app.run()
    timer.invalidate()
    log.info("ui", "event_loop_stopped", by="ctrl-c" if interrupted["flag"] else "quit")
