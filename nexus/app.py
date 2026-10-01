"""NEXUS: hand cursor + Jarvis (voice) + the Siri-style Jarvis panel.

    python -m nexus              # everything; bunch your fingertips then spread them to open the panel
    python -m nexus --preview    # OpenCV camera window instead of the panel (debugging)
    python -m nexus --no-hands   # Jarvis only, the camera never turns on (panel shows while Jarvis is busy)
    python -m nexus --no-voice   # hand cursor only, the mic never turns on
    python -m nexus --record     # save every frame + landmarks to runs/
    python -m nexus --gestures   # trained gesture model: swipe = switch desktop, open palm = stop
"""
import argparse
import sys
import threading
import time

import cv2

from .camera import Camera
from .config import SENSITIVITY, Config
from .log import log
from .mouse import CursorAnimator, Mouse, accessibility_trusted, screen_size
from .perf import Perf, Recorder, system_info
from .pointer import PointerEngine
from .spread import SpreadControl

MOTION_THRESHOLD = 1.5   # mean pixel change (0-255) on a 64x48 gray thumbnail
MOTION_RECHECK = 15      # with no motion, still run hand tracking every Nth frame
REOPEN = threading.Event()  # set when Nexus.app is opened again while running (SIGUSR1 from the launcher)


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="nexus", description="Hand + voice control for your Mac, all local")
    p.add_argument("--preview", action="store_true", help="OpenCV camera window with landmarks (replaces the panel)")
    p.add_argument("--no-ui", action="store_true", help="no Jarvis panel")
    p.add_argument("--record", action="store_true", help="save per-frame data + landmarks to runs/")
    p.add_argument("--no-mouse", action="store_true", help="don't move or click the real cursor")
    p.add_argument("--sensitivity", choices=list(SENSITIVITY), default="medium")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--no-hands", action="store_true", help="voice only: never turn the camera on")
    p.add_argument("--no-voice", action="store_true", help="hands only: never turn the mic on")
    p.add_argument("--whisper", default="base.en", help="speech model: tiny.en (fastest), base.en, small.en")
    p.add_argument("--voice", default=None, help='Jarvis voice, e.g. "Nathan" (default: best installed)')
    p.add_argument("--llm", default=None, help="local Ollama model for complex requests (default: qwen2.5:3b)")
    p.add_argument("--wake-threshold", type=float, default=0.35,
                   help='how sure the "Hey Jarvis" detector must be, 0-1 (default 0.35). Lower if it misses you, higher if it triggers by itself')
    p.add_argument("--mic", default=None,
                   help='microphone, any part of its name, e.g. "MacBook" (default: built-in if the default is AirPods)')
    p.add_argument("--gestures", nargs="?", const="models/gesture.pt", default=None,
                   help="use the trained gesture model: swipe = switch desktop, open palm = stop")
    p.add_argument("--app", action="store_true",
                   help="run as the Nexus app: saved settings + menu bar icon (used by Nexus.app)")
    return p.parse_args(argv)


def apply_saved_settings(args):
    """--app: start from ~/.config/nexus/settings.json. Hands/voice start paused instead of absent,
    so the menu can switch them on without a restart."""
    from pathlib import Path

    from . import settings
    s = settings.load()
    args.start_hands, args.start_voice = s["hands"], s["voice"]
    args.first_launch = not s["onboarded"]
    if args.first_launch:  # nothing turns on (and nothing asks for permission) until the setup guide explains it
        args.start_hands = args.start_voice = False
    args.no_hands = args.no_voice = False
    args.mic = args.mic or s["mic"]
    args.wake_threshold = s["wake_threshold"]
    args.sensitivity = s["sensitivity"]
    args.calibration = s["calibration"]
    from . import paths
    if s["gestures"] and paths.gesture_model().exists():
        try:
            import torch  # noqa: F401  the trained model needs PyTorch; a release build may not include it
            args.gestures = args.gestures or str(paths.gesture_model())
        except ImportError:
            log.info("gesture_model", "skipped", reason="PyTorch not in this build")
    args.preview, args.no_ui = False, False
    return s


def draw_preview(frame, hand, state, fps, jarvis=None, gesture=""):
    h, w = frame.shape[:2]
    view = cv2.flip(frame, 1)
    if hand is not None:
        pts = [(int((1 - x) * w), int(y * h)) for x, y, _ in hand.landmarks]
        bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                 (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
        color = (80, 220, 255) if state.pinched else (255, 255, 255)
        for a, b in bones:
            cv2.line(view, pts[a], pts[b], color, 2)
    label = f"{fps:.0f} fps  ratio {state.ratio:.2f}" if state.ratio else f"{fps:.0f} fps  no hand"
    if state.pinched:
        label += "  CLICK" if state.holding_click else "  DRAG"
    if gesture:
        label += f"  last gesture: {gesture}"
    cv2.putText(view, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if jarvis is not None:
        cv2.putText(view, f"Jarvis: {jarvis.state}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 80), 2)
        if jarvis.heard:
            cv2.putText(view, f'"{jarvis.heard[:60]}"', (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        if jarvis.reply:
            cv2.putText(view, jarvis.reply[:60], (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 160), 1)
    cv2.imshow("NEXUS", view)
    return cv2.waitKey(1) & 0xFF


class HandLoop:
    """Camera -> MediaPipe -> pointer, clicks, gestures, finger-spread panel control."""

    def __init__(self, args, cfg, mouse, jarvis, spread, W, H, start_paused: bool = False):
        from .hands import HandTracker  # imported here so --help works without MediaPipe
        self.args, self.cfg, self.mouse, self.jarvis, self.spread = args, cfg, mouse, jarvis, spread
        self.W, self.H = W, H
        self.tracker = HandTracker(cfg.model_path)
        self.cam = Camera(cfg.camera_index, cfg.camera_width, cfg.camera_height, cfg.camera_fps)
        if not start_paused:  # first launch: the camera waits until the setup guide has explained it
            self.cam.start()
        self.pinches = self.panel_opens = 0  # for the setup guide's checklist
        self.hand_visible = False
        self.engine = PointerEngine(W, H, cfg, self.cam.aspect)
        # Your saved hand calibration: pinch thresholds and the panel's bunched/spread range
        from . import calibration
        calibration.apply(getattr(args, "calibration", None), cfg, self.engine, spread)
        self.calibrator = None  # set while calibrating: samples go here, clicks and the panel pause
        self._stats = {"ratio": [], "spread": []}
        self.recognizer = None
        if args.gestures:
            try:
                from .gesture_model import GestureRecognizer
                self.recognizer = GestureRecognizer(args.gestures)
            except Exception as e:
                log.warn("gesture_model", "not_loaded", error=str(e), fix="python -m nexus.collect, then python -m nexus.train")
        self.animator = CursorAnimator(mouse, cfg)
        self.animator.start()
        self.perf = Perf()
        self.recorder = Recorder(cfg.sensitivity) if args.record else None
        if self.recorder:
            log.info("recorder", "started", file=str(self.recorder.path))
        self.stop = threading.Event()
        self.paused = threading.Event()
        if start_paused:
            self.paused.set()
        self.skipped_still = 0

    def pause(self):
        """Menu bar: hand control off. The camera is closed, so its green light goes off."""
        if self.paused.is_set():
            return
        self.paused.set()
        if self.mouse:
            self.mouse.release_if_down()
        self.cam.stop()
        log.info("app", "hands_paused")

    def resume(self):
        if not self.paused.is_set():
            return
        self.cam.start()
        self.paused.clear()
        log.info("app", "hands_resumed")

    def _gesture(self, g):
        from . import actions
        if g == "swipe_left":
            actions.run("keys", combo="ctrl+left")   # previous desktop
        elif g == "swipe_right":
            actions.run("keys", combo="ctrl+right")  # next desktop
        elif g == "open_palm":                         # stop: release any drag, silence Jarvis
            if self.mouse:
                self.mouse.release_if_down()
            if self.jarvis is not None and getattr(self.jarvis, "speaker", None):
                self.jarvis.speaker.stop()

    def run(self, preview: bool):
        cfg, t0 = self.cfg, time.perf_counter()
        seq, frame_i, fps, last_fps_t, fps_frames = 0, 0, 0.0, t0, 0
        idle, prev_thumb, gesture_label = False, None, ""
        try:
            while not self.stop.is_set():
                if self.paused.is_set():
                    self.hand_visible = False
                    time.sleep(0.2)
                    continue
                got = self.cam.latest(seq, timeout=1.0)
                if got is None:
                    if not self.paused.is_set():
                        log.warn("camera", "no_frames")
                    continue
                seq, frame, t_cap = got
                frame_i += 1

                # Idle: no hand for a while. A 64x48 thumbnail diff is almost free; hand tracking only
                # runs when something moved (a hand can't appear without motion), plus a periodic check.
                now_idle = self.engine.away(t_cap) > cfg.idle_after
                if now_idle != idle:
                    idle = now_idle
                    log.info("perf", "idle_on" if idle else "idle_off", motion_skipped=self.skipped_still)
                    self.skipped_still = 0
                if idle:
                    thumb = cv2.cvtColor(cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                    moved = prev_thumb is None or float(cv2.absdiff(thumb, prev_thumb).mean()) > MOTION_THRESHOLD
                    prev_thumb = thumb
                    if not moved and frame_i % MOTION_RECHECK:
                        self.skipped_still += 1
                        continue
                else:
                    prev_thumb = None

                v0 = time.perf_counter()
                hand = self.tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int((t_cap - t0) * 1000))
                vision_ms = (time.perf_counter() - v0) * 1000

                state = self.engine.update(hand, t_cap)
                calibrating = self.calibrator is not None
                if calibrating:
                    if hand is not None and hand.world is not None:
                        from .spread import raw_spread
                        self.calibrator.add(state.ratio, raw_spread(hand.world))
                    state.event = None  # the pinch pose must not click anything while calibrating
                self.animator.push(state)
                self.hand_visible = state.visible
                if state.event == "down":
                    self.pinches += 1
                if self.spread is not None and not calibrating:
                    self.spread.update(hand.world if hand is not None else None, t_cap, pinched=state.pinched)
                    for e in self.spread.pop_events():
                        log.info("ui", f"panel_{e}", openness=self.spread.openness)
                        if e == "opened":
                            self.panel_opens += 1
                        if self.jarvis is not None:
                            # Panel open = a conversation: Jarvis keeps listening, no wake word. Closed = over.
                            self.jarvis.set_hold_open(e == "opened")
                if self.recognizer is not None:
                    g = self.recognizer.update(hand, t_cap)
                    if g:
                        gesture_label = g
                        self._gesture(g)
                if self.mouse and state.event == "down":
                    self.mouse.press(*(self.animator.output or (state.x, state.y)))
                elif self.mouse and state.event == "up":
                    self.mouse.release(*(self.animator.output or (state.x, state.y)))
                if self.mouse and not state.visible and state.visibility <= 0:
                    self.mouse.release_if_down()

                latency_ms = (time.perf_counter() - t_cap) * 1000
                on_edge = state.x <= 0.5 or state.y <= 0.5 or state.x >= self.W - 0.5 or state.y >= self.H - 0.5
                self.perf.add(latency_ms, vision_ms, state, on_edge)
                if state.visible and state.ratio is not None:
                    self._stats["ratio"].append(state.ratio)
                    if self.spread is not None and self.spread.last_raw is not None:
                        self._stats["spread"].append(self.spread.last_raw)
                summary = self.perf.flush_if_due(t_cap, self.animator.fps, self.cam.dropped)
                if summary:
                    log.info("perf", "summary", sensitivity=cfg.sensitivity, motion_skipped=self.skipped_still, **summary)
                    self._log_hand_stats()
                if self.recorder:
                    self.recorder.write(state, hand, vision_ms, latency_ms, cfg.sensitivity)

                fps_frames += 1
                if t_cap - last_fps_t >= 1:
                    fps, fps_frames, last_fps_t = fps_frames / (t_cap - last_fps_t), 0, t_cap
                if preview and draw_preview(frame, hand, state, fps, self.jarvis, gesture_label) == ord("q"):
                    break
        except Exception as e:
            log.error("app", "hand_loop_failed", error=repr(e))
        finally:
            self.stop.set()

    def _log_hand_stats(self):
        """Your hand's actual pinch and spread ranges over the last 5 s: the data for tuning."""
        import numpy as np
        r, s = self._stats["ratio"], self._stats["spread"]
        if len(r) >= 10:
            pr = np.percentile(r, [5, 50, 95])
            fields = dict(pinch_p5=pr[0], pinch_p50=pr[1], pinch_p95=pr[2], frames=len(r),
                          fists_ignored=self.engine.pinch.rejected_fists,
                          pinch_enter=self.engine.pinch.enter, calibrated=bool(getattr(self.args, "calibration", None)))
            if len(s) >= 10:
                sp = np.percentile(s, [5, 50, 95])
                fields.update(spread_p5=sp[0], spread_p50=sp[1], spread_p95=sp[2],
                              spread_lo=self.spread.lo, spread_hi=self.spread.hi)
            log.info("hand", "stats", **fields)
        self._stats = {"ratio": [], "spread": []}

    def close(self):
        self.stop.set()
        if self.mouse:
            self.mouse.release_if_down()
        self.animator.stop()
        self.cam.stop()
        self.tracker.close()
        if self.recorder:
            self.recorder.close()
            log.info("recorder", "stopped", file=str(self.recorder.path))


def _notify(text: str):
    import subprocess
    subprocess.run(["osascript", "-e", f'display notification "{text}" with title "Nexus"'], capture_output=True)


class Controller:
    """One place for on/off switches and windows, shared by the menu bar icon, the Nexus window and
    the setup guide, so they always agree."""

    def __init__(self, hands, jarvis, spread, panel, quit_requested, reopen):
        self.hands, self.jarvis, self.spread, self.panel = hands, jarvis, spread, panel
        self.quit_requested, self.reopen = quit_requested, reopen
        self.menu = self.home = self.setup = None
        # Auto-update: watch the code; restart (exit 75, launch.sh starts us again) when it changes
        self.exit_code = 0
        self._code_seen = self._code_mtime()
        self._code_changed_at = None
        self._last_code_check = 0.0

    UPDATE_CHECK_S = 3.0   # how often to look at the files
    UPDATE_SETTLE_S = 5.0  # wait until nothing has changed for this long (a multi-file update is done)

    @staticmethod
    def _code_mtime() -> float:
        from pathlib import Path
        files = list(Path("nexus").glob("*.py")) + [Path("requirements.txt")]
        return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)

    def _idle(self) -> bool:
        """Safe to restart: not mid-conversation, mid-dictation, mid-drag, or in the setup guide."""
        j = self.jarvis
        if j is not None and (j.state not in ("idle", "off") or j.dictating):
            return False
        if self.hands is not None and self.hands.mouse is not None and self.hands.mouse.down:
            return False
        return self.setup is None or not self.setup.visible

    def _check_for_update(self):
        from . import paths
        if paths.BUNDLED:
            return  # a released app doesn't watch its own code; updates come as a new version
        now = time.perf_counter()
        if now - self._last_code_check < self.UPDATE_CHECK_S:
            return
        self._last_code_check = now
        m = self._code_mtime()
        if m > self._code_seen:
            self._code_seen = m
            self._code_changed_at = now
            log.info("app", "update_pending")
        if self._code_changed_at and now - self._code_changed_at >= self.UPDATE_SETTLE_S and self._idle():
            log.info("app", "update_restart")
            _notify("Nexus was updated. Restarting…")
            self.exit_code = 75
            self.quit_requested.set()

    # --- state ---

    def settings(self) -> dict:
        from . import settings
        return settings.load()

    def save(self, **changes):
        from . import settings
        settings.update(**changes)

    def hands_on(self) -> bool:
        return self.hands is not None and not self.hands.paused.is_set()

    def voice_on(self) -> bool:
        return self.jarvis is not None and not self.jarvis.paused

    def set_hands(self, on: bool) -> bool:
        if self.hands is None:
            return False
        try:
            self.hands.resume() if on else self.hands.pause()
        except Exception as e:
            log.error("app", "hands_toggle_failed", error=str(e))
            _notify("Hand control could not start. Allow Camera for Nexus in Setup and permissions.")
        self.save(hands=self.hands_on())
        self.panel.auto_show = not self.hands_on()  # without hands, the panel shows itself while Jarvis is busy
        if self.menu:
            self.menu.set_checked("hands", self.hands_on())
        return self.hands_on()

    def set_voice(self, on: bool) -> bool:
        if self.jarvis is None:
            return False
        try:
            self.jarvis.resume() if on else self.jarvis.pause()
        except Exception as e:
            log.error("app", "voice_toggle_failed", error=str(e))
            _notify("Jarvis could not open the microphone. Allow Microphone for Nexus in Setup and permissions.")
        self.save(voice=self.voice_on())
        if self.menu:
            self.menu.set_checked("voice", self.voice_on())
        return self.voice_on()

    def apply_choices(self, hands: bool, voice: bool):
        """From the setup guide, right after the permissions step."""
        log.info("setup", "choices", hands=hands, voice=voice)
        self.set_hands(hands)
        self.set_voice(voice)

    def toggle_panel(self):
        is_open = self.panel.pinned or (self.spread is not None and self.spread.is_open)
        if is_open:
            return self.close_panel()
        if self.spread is None or not self.hands_on():
            self.panel.pinned = True  # no hand control: the menu/window/button opens it
        else:
            self.spread.force(True)  # the gesture can still close it
        if self.jarvis is not None:
            self.jarvis.set_hold_open(True)  # panel open = a conversation, no wake word

    def close_panel(self):
        self.panel.pinned = False
        if self.spread is not None:
            self.spread.force(False)
        if self.jarvis is not None:
            self.jarvis.set_hold_open(False)

    # --- hand calibration, pointer speed, About ---

    def calibration(self) -> dict | None:
        return self.settings().get("calibration")

    def start_calibration(self):
        """Clicks and the panel gesture pause while it runs; returns the Calibrator (None without hands)."""
        from .calibration import Calibrator
        if self.hands is None:
            return None
        if not self.hands_on():
            self.set_hands(True)
        self.hands.calibrator = Calibrator()
        log.info("calibration", "started")
        return self.hands.calibrator

    def finish_calibration(self, cal: dict | None):
        from . import calibration
        if self.hands is not None:
            self.hands.calibrator = None
        if cal and self.hands is not None:
            self.save(calibration=cal)
            self.hands.args.calibration = cal
            calibration.apply(cal, self.hands.cfg, self.hands.engine, self.spread)
            log.info("calibration", "saved", **{k: v for k, v in cal.items() if k != "date"})

    def set_sensitivity(self, name: str):
        """Pointer speed: how much hand movement crosses the screen. Applies right away."""
        if self.hands is not None:
            self.hands.cfg.sensitivity = name
        self.save(sensitivity=name)

    def about(self):
        from AppKit import NSApp, NSAttributedString
        from . import __version__
        credits = NSAttributedString.alloc().initWithString_(
            "Hand and voice control for your Mac. Everything runs on this Mac.\n"
            "MediaPipe, Whisper, openWakeWord, MiniLM and Qwen via Ollama.")
        NSApp.activateIgnoringOtherApps_(True)
        NSApp.orderFrontStandardAboutPanelWithOptions_({"ApplicationName": "Nexus",
                                                         "ApplicationVersion": __version__, "Credits": credits})

    def show_calibration(self):
        from .windows import CalibrationWindow
        if self.hands is None:
            _notify("Hand control is not available. Allow Camera for Nexus in Setup and permissions.")
            return
        if getattr(self, "calib_win", None) is None or not self.calib_win.visible:
            self.calib_win = CalibrationWindow(self)
        self.calib_win.show()

    def set_wake_threshold(self, t: float):
        j = self.jarvis
        if j is None:
            return
        j.wake_threshold = t
        if getattr(j, "wake", None) is not None:
            j.wake.threshold = t
        self.save(wake_threshold=t)

    def shutdown(self):
        """macOS is quitting us (Dock menu, logging out): release the camera and mic right now."""
        log.info("app", "terminating")
        if self.hands is not None:
            self.hands.pause()
        if self.jarvis is not None:
            self.jarvis.pause()

    def set_login(self, on: bool):
        import os
        import subprocess
        app = os.environ.get("NEXUS_APP")
        if not app:
            log.warn("app", "login_item_skipped", reason="only works when running as Nexus.app")
            return
        script = (f'tell application "System Events" to make login item at end with properties {{path:"{app}", hidden:true}}'
                  if on else 'tell application "System Events" to delete login item "Nexus"')
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        log.info("app", "login_item", on=on, ok=r.returncode == 0)
        self.save(login=on)

    def open_folder(self, name: str):
        import subprocess
        from pathlib import Path
        Path(name).mkdir(exist_ok=True)
        subprocess.run(["open", str(Path(name).resolve())])

    def open_logs(self):
        from . import paths
        self.open_folder(str(paths.LOGS))

    def quit(self):
        self.quit_requested.set()

    # --- windows ---

    def show_home(self):
        from .windows import HomeWindow
        if self.home is None:
            self.home = HomeWindow(self)
        self.home.show()

    def show_setup(self):
        from .windows import SetupWindow
        if self.setup is None or not self.setup.visible:
            self.setup = SetupWindow(self, self._setup_done)
        self.setup.show()

    def _setup_done(self):
        self.save(onboarded=True)
        log.info("setup", "finished", hands=self.hands_on(), voice=self.voice_on())
        self.setup.close()
        self.show_home()

    def build_menu(self):
        from .ui import StatusMenu
        items = [
            ("home", "Open Nexus\u2026", None, lambda _: self.show_home()),
            ("setup", "Setup & permissions\u2026", None, lambda _: self.show_setup()),
            None,
            ("hands", "Hand control (camera)", self.hands_on(), lambda _: self.set_hands(not self.hands_on()))
            if self.hands else "skip",
            ("voice", "Jarvis (\u201cHey Jarvis\u201d)", self.voice_on(), lambda _: self.set_voice(not self.voice_on()))
            if self.jarvis else "skip",
            ("panel", "Show / hide Jarvis panel", None, lambda _: self.toggle_panel()),
            ("calibrate", "Calibrate hand\u2026", None, lambda _: self.show_calibration()) if self.hands else "skip",
            None,
            ("about", "About Nexus", None, lambda _: self.about()),
            ("logs", "Open logs", None, lambda _: self.open_logs()),
            ("results", "Open benchmark results", None, lambda _: self.open_folder("results")),
            None,
            ("quit", "Quit Nexus", None, lambda _: self.quit()),
        ]
        self.menu = StatusMenu([i for i in items if i != "skip"])
        return self.menu

    def tick(self, dt: float):
        self.panel.tick(dt)
        self._check_for_update()
        if self.reopen.is_set():  # Nexus.app was opened again while running
            self.reopen.clear()
            self.show_home()
        if self.setup is not None:
            self.setup.tick()
        if self.home is not None:
            self.home.tick()
        if getattr(self, "calib_win", None) is not None:
            self.calib_win.tick()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.app:
        # First thing: without a handler, the launcher's "show your window" signal would quit the app
        # if it arrives while the models are still loading
        import signal
        signal.signal(signal.SIGUSR1, lambda *_: REOPEN.set())
    args.start_hands = args.start_voice = True
    args.first_launch = False
    if args.app:
        apply_saved_settings(args)
    else:  # Terminal runs use your saved hand calibration too
        from . import settings
        args.calibration = settings.load().get("calibration")
    if args.no_hands and args.no_voice:
        print("--no-hands and --no-voice together leave nothing to run", file=sys.stderr)
        return 1
    cfg = Config(camera_index=args.camera, sensitivity=args.sensitivity)
    path = log.open()
    W, H = screen_size()
    use_ui = not args.no_ui and not args.preview
    log.info("app", "start", log_file=str(path), screen=f"{int(W)}x{int(H)}", sensitivity=cfg.sensitivity,
             hands=not args.no_hands, voice=not args.no_voice, ui=use_ui, **system_info())
    if args.preview and not args.no_ui:
        log.info("ui", "panel_off", reason="--preview uses the OpenCV window instead")

    mouse = None
    if not args.no_mouse and not args.no_hands:
        if not accessibility_trusted(prompt=not args.first_launch):
            log.warn("mouse", "no_accessibility",
                     fix="System Settings -> Privacy & Security -> Accessibility -> enable Nexus (or your terminal), then reopen")
        mouse = Mouse()

    jarvis = None
    if not args.no_voice:
        from .jarvis import Jarvis
        jarvis = Jarvis(whisper_model=args.whisper, voice=args.voice, llm_model=args.llm,
                        wake_threshold=args.wake_threshold, mic=args.mic)
        if not args.start_voice:
            jarvis.paused = True  # models load, the mic stays closed until switched on
        jarvis.start()

    spread = SpreadControl() if (use_ui and not args.no_hands) else None
    hands = None
    if not args.no_hands:
        try:
            hands = HandLoop(args, cfg, mouse, jarvis, spread, W, H, start_paused=not args.start_hands)
        except Exception as e:
            # No camera (usually permission not granted yet): keep Jarvis running, voice-only.
            # Log 20260927-214756: the whole app used to quit here, taking "Hey Jarvis" with it.
            log.error("app", "hands_unavailable", error=str(e), note="continuing with voice only")
            if jarvis is None:
                log.close()
                print(f"\n{e}", file=sys.stderr)
                return 1
            args.no_hands = True
            spread = None
            if args.app:
                import subprocess
                subprocess.run(["osascript", "-e", 'display notification "Hand control is off: allow Camera for Nexus '
                                "in System Settings, then reopen Nexus. Jarvis still works.\" with title \"Nexus\""])
                subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"])

    ctl = None
    try:
        if use_ui:
            from .ui import JarvisPanel, init_app, install_app_menu_and_dock, run
            # A real Nexus.app process (the compiled stub): Dock icon, app switcher, its own menu bar.
            # Anything else (Terminal, the script fallback) would show up as "python3.11", so stays hidden.
            as_bundle = args.app and "/Contents/MacOS/" in sys.executable
            init_app(regular=as_bundle)
            panel = JarvisPanel(jarvis, spread, auto_show=args.no_hands or not args.start_hands)

            def close_panel():  # the panel's close button (the Controller replaces this in the app)
                panel.pinned = False
                if spread is not None:
                    spread.force(False)
                if jarvis is not None:
                    jarvis.set_hold_open(False)
            panel.on_close = close_panel
            if hands:
                threading.Thread(target=hands.run, args=(False,), name="nexus-hands", daemon=True).start()
            quit_requested = threading.Event()
            ctl = None
            if args.app:
                ctl = Controller(hands, jarvis, spread, panel, quit_requested, REOPEN)
                panel.on_close = ctl.close_panel
                ctl.build_menu()
                dock = install_app_menu_and_dock(ctl) if as_bundle else None  # keep alive
                s = ctl.settings()
                if not s["onboarded"]:
                    ctl.show_setup()
                elif s["show_home"]:
                    ctl.show_home()
            # Quit from the menu, or when the hand loop stops (Ctrl+C, camera error), or in voice-only
            # mode when Jarvis stops
            stopped = lambda: quit_requested.is_set() or (hands.stop.is_set() if hands else not jarvis.is_alive())
            run(ctl.tick if ctl else panel.tick, stopped)
        elif hands:
            hands.run(preview=args.preview)  # main thread (OpenCV windows need it)
        else:
            while jarvis.is_alive():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if hands:
            hands.close()
        if jarvis:
            jarvis.stop()
        log.info("app", "quit", restart_for_update=bool(ctl and ctl.exit_code == 75))
        log.close()
        cv2.destroyAllWindows()
    return ctl.exit_code if ctl else 0  # 75 = launch.sh starts the updated code right away
