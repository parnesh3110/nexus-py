"""Things Jarvis can do on the Mac. Each returns (ok, message).

Apps via `open` + AppleScript, keys and typing via Quartz keyboard events, windows via
System Events (Accessibility), music via Music's AppleScript dictionary. The first time an
app is scripted, macOS asks "Terminal wants to control X": click OK.
"""
import subprocess
import time
import urllib.parse

import Quartz

from .apps import index
from .log import log

# macOS virtual key codes (ANSI layout)
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12,
    "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23,
    "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34,
    "p": 35, "return": 36, "enter": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43,
    "/": 44, "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49, "`": 50, "delete": 51, "backspace": 51,
    "escape": 53, "esc": 53, "left": 123, "right": 124, "down": 125, "up": 126, "home": 115, "end": 119,
    "pageup": 116, "pagedown": 121, "forwarddelete": 117,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101,
    "f10": 109, "f11": 103, "f12": 111,
}
MODIFIERS = {
    "cmd": Quartz.kCGEventFlagMaskCommand, "command": Quartz.kCGEventFlagMaskCommand,
    "shift": Quartz.kCGEventFlagMaskShift, "opt": Quartz.kCGEventFlagMaskAlternate,
    "option": Quartz.kCGEventFlagMaskAlternate, "alt": Quartz.kCGEventFlagMaskAlternate,
    "ctrl": Quartz.kCGEventFlagMaskControl, "control": Quartz.kCGEventFlagMaskControl,
}
_src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)


def osascript(script: str, timeout: float = 15) -> tuple[bool, str]:
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    out = (r.stdout or r.stderr).strip()
    return r.returncode == 0, out


def _esc(s: str) -> str:
    return s.replace("\\", "").replace('"', "")


def frontmost_app() -> str:
    ok, out = osascript('tell application "System Events" to get name of first application process whose frontmost is true', 3)
    return out if ok else ""


# --- apps ---

def open_app(name: str) -> tuple[bool, str]:
    app = index.resolve(name)
    if app is None:
        return False, f"I couldn't find an app called {name}."
    subprocess.run(["open", app.path])
    # Wait until it's actually in front, so the next step (typing, a shortcut) goes to it
    for _ in range(30):
        if frontmost_app().lower() in (app.name.lower(), app.key):
            break
        time.sleep(0.1)
    time.sleep(0.3)  # a brand-new window needs a moment
    return True, f"Opening {app.name}."


def quit_app(name: str) -> tuple[bool, str]:
    if name in ("this", "it", "this app", "front"):
        name = frontmost_app()
    app = index.resolve(name)
    target = app.name if app else name
    ok, out = osascript(f'tell application "{_esc(target)}" to quit')
    return ok, f"Quit {target}." if ok else f"Couldn't quit {target}."


def focus_app(name: str) -> tuple[bool, str]:
    app = index.resolve(name)
    if app is None:
        return False, f"I couldn't find {name}."
    ok, _ = osascript(f'tell application "{_esc(app.name)}" to activate')
    return ok, f"Switched to {app.name}."


# --- keyboard ---

def keys(combo: str) -> tuple[bool, str]:
    parts = combo.lower().replace(" ", "").split("+")
    key, mods = parts[-1], parts[:-1]
    if key not in KEYCODES or any(m not in MODIFIERS for m in mods):
        return False, f"unknown shortcut {combo}"
    flags = 0
    for m in mods:
        flags |= MODIFIERS[m]
    for down in (True, False):
        e = Quartz.CGEventCreateKeyboardEvent(_src, KEYCODES[key], down)
        Quartz.CGEventSetFlags(e, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)
    return True, f"pressed {combo}"


def type_text(text: str) -> tuple[bool, str]:
    for i in range(0, len(text), 20):  # a keyboard event carries ~20 characters at most
        chunk = text[i:i + 20]
        for down in (True, False):
            e = Quartz.CGEventCreateKeyboardEvent(_src, 0, down)
            Quartz.CGEventKeyboardSetUnicodeString(e, len(chunk), chunk)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)
    return True, f"typed {len(text)} characters"


# --- windows (front window of the front app) ---

_FRONT = 'tell application "System Events" to tell (first application process whose frontmost is true)'


def window(action: str) -> tuple[bool, str]:
    if action == "minimize":
        ok, out = osascript(f'{_FRONT} to set value of attribute "AXMinimized" of window 1 to true')
    elif action == "fullscreen":
        ok, out = osascript(f'{_FRONT} to set value of attribute "AXFullScreen" of window 1 to '
                            f'not (value of attribute "AXFullScreen" of window 1)')
    elif action == "close":
        ok, out = osascript(f'{_FRONT} to click (first button of window 1 whose subrole is "AXCloseButton")')
    elif action in ("maximize", "left_half", "right_half"):
        from AppKit import NSScreen
        s = NSScreen.mainScreen()
        full, vf = s.frame(), s.visibleFrame()
        x, w, h = vf.origin.x, vf.size.width, vf.size.height
        y = full.size.height - (vf.origin.y + vf.size.height)  # AppKit bottom-left -> top-left
        if action == "left_half":
            w = w / 2
        elif action == "right_half":
            x, w = x + w / 2, w / 2
        ok, out = osascript(f'''tell application "System Events"
            tell (first application process whose frontmost is true)
                tell window 1
                    set position to {{{int(x)}, {int(y)}}}
                    set size to {{{int(w)}, {int(h)}}}
                end tell
            end tell
        end tell''')
    else:
        return False, f"unknown window action {action}"
    return ok, "" if ok else f"couldn't {action.replace('_', ' ')} the window"


# --- music ---

def music(action: str, query: str = "") -> tuple[bool, str]:
    q = _esc(query)
    scripts = {
        "pause": 'tell application "Music" to pause',
        "resume": 'tell application "Music" to play',
        "next": 'tell application "Music" to next track',
        "previous": 'tell application "Music" to previous track',
    }
    if action != "play" or not q:
        ok, _ = osascript(scripts.get(action, scripts["resume"]))
        return ok, "" if ok else "Music didn't respond."
    ok, out = osascript(f'''tell application "Music"
        set found to (search playlist "Library" for "{q}")
        if found is {{}} then return "NOT_IN_LIBRARY"
        play item 1 of found
        return (name of item 1 of found) & " by " & (artist of item 1 of found)
    end tell''')
    if out == "NOT_IN_LIBRARY":
        # Songs that aren't in your library can't be played by script: open the search instead
        subprocess.run(["open", "music://music.apple.com/search?term=" + urllib.parse.quote(query)])
        return True, f"{query} isn't in your library, so I opened it in Apple Music."
    return ok, f"Playing {out}." if ok else "Music didn't respond."


# --- web, system ---

def open_url(url: str) -> tuple[bool, str]:
    t = url.strip()
    if "." in t and " " not in t:
        url = t if t.startswith("http") else "https://" + t
        subprocess.run(["open", url])
        return True, f"Opening {t}."
    subprocess.run(["open", "https://www.google.com/search?q=" + urllib.parse.quote(t)])
    return True, f"Searching for {t}."


def system(action: str) -> tuple[bool, str]:
    scripts = {
        "volume_up": "set volume output volume ((output volume of (get volume settings)) + 12)",
        "volume_down": "set volume output volume ((output volume of (get volume settings)) - 12)",
        "mute": "set volume with output muted",
        "unmute": "set volume without output muted",
    }
    if action == "lock_screen":
        return keys("ctrl+cmd+q")
    if action not in scripts:
        return False, f"unknown system action {action}"
    ok, _ = osascript(scripts[action])
    return ok, ""


# --- menus, buttons, scripts, shell, files ---

def menu(path, app: str = "") -> tuple[bool, str]:
    """Click a menu bar item, e.g. ["File", "New Window"] or ["View", "Sort By", "Name"]."""
    if isinstance(path, str):
        path = [p.strip() for p in path.split(">")]
    if len(path) < 2:
        return False, "a menu path needs a menu and an item"
    if app:
        resolved = index.resolve(app)
        app = resolved.name if resolved else app
        osascript(f'tell application "{_esc(app)}" to activate')
        time.sleep(0.3)
    proc = app or frontmost_app()
    ref = f'menu bar item "{_esc(path[0])}" of menu bar 1'
    for name in path[1:]:
        ref = f'menu item "{_esc(name)}" of menu 1 of {ref}'
    ok, _ = osascript(f'tell application "System Events" to tell process "{_esc(proc)}" to click {ref}')
    return ok, "" if ok else f"I couldn't find {' > '.join(path)} in {proc}."


def click_button(name: str) -> tuple[bool, str]:
    """Click the first control in the front window whose label or description matches."""
    n = _esc(name)
    ok, out = osascript(f'''tell application "System Events"
        tell (first application process whose frontmost is true)
            repeat with e in (entire contents of window 1)
                try
                    if (name of e is "{n}") or (description of e is "{n}") then
                        click e
                        return "CLICKED"
                    end if
                end try
            end repeat
        end tell
    end tell
    return "NOT_FOUND"''', timeout=12)
    if out == "CLICKED":
        return True, ""
    return False, f"I couldn't find {name} in this window."


def applescript(script: str, summary: str = "") -> tuple[bool, str]:
    ok, out = osascript(script, timeout=30)
    return ok, out or ("done" if ok else "the script failed")


def shell(command: str, summary: str = "") -> tuple[bool, str]:
    try:
        r = subprocess.run(["/bin/zsh", "-lc", command], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return False, "timed out after 20 s"
    out = (r.stdout + r.stderr).strip()
    return r.returncode == 0, (out[:3000] or "(no output)")


def find_files(query: str) -> tuple[bool, str]:
    """Spotlight search by name in your home folder, newest first."""
    import os
    try:
        out = subprocess.run(["mdfind", "-onlyin", os.path.expanduser("~"), "-name", query],
                             capture_output=True, text=True, timeout=8).stdout
    except subprocess.TimeoutExpired:
        return False, "search timed out"
    paths = [p for p in out.splitlines() if "/Library/" not in p and "/." not in p]
    paths.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    if not paths:
        return False, f"No files matching {query}."
    return True, "\n".join(paths[:10])


def open_target(target: str, app: str = "") -> tuple[bool, str]:
    """Open a file, folder or URL (optionally in a given app), or an app by name."""
    import os
    t = os.path.expanduser(target.strip())
    is_url = "://" in t
    if not is_url and not os.path.exists(t):
        if index.resolve(t):
            return open_app(t)
        return False, f"I couldn't find {target}."
    cmd = ["open"]
    if app:
        resolved = index.resolve(app)
        cmd += ["-a", resolved.path if resolved else app]
    subprocess.run(cmd + [t])
    return True, f"Opening {os.path.basename(t.rstrip('/')) or t}."


def run(name: str, /, **kw) -> tuple[bool, str]:
    """Dispatch by name, with logging. Used by commands.py and the local LLM planner.
    `name` is positional-only: many actions take an argument called `action` (music, window,
    system), which used to collide with it and crash (log 20260927-213438)."""
    action = name
    fn = {"open_app": open_app, "quit_app": quit_app, "focus_app": focus_app, "keys": keys, "type": type_text,
          "window": window, "music": music, "open_url": open_url, "system": system, "menu": menu,
          "click": click_button, "applescript": applescript, "shell": shell, "find_files": find_files,
          "open": open_target}.get(action)
    if fn is None:
        return False, f"unknown action {action}"
    start = time.perf_counter()
    try:
        ok, msg = fn(**kw)
    except Exception as e:
        ok, msg = False, str(e)
    (log.info if ok else log.warn)("action", action, args=str(kw)[:200], ok=ok, result=msg[:200],
                                   ms=int((time.perf_counter() - start) * 1000))
    return ok, msg
