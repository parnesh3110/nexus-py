"""Saved app settings (~/Library/Application Support/Nexus/settings.json; moved there once from the
old ~/.config/nexus). The menu bar toggles, the Nexus window and the setup guide write here."""
import json
from pathlib import Path

from . import paths

PATH = paths.SETTINGS
DEFAULTS = {
    "hands": True,           # hand control (camera)
    "voice": True,           # Jarvis (mic)
    "gestures": True,        # use models/gesture.pt if it exists
    "mic": None,             # part of a mic name; None = built-in when the default is Bluetooth
    "wake_threshold": 0.35,  # "Hey Jarvis" scored 0.44-0.48 on the built-in mic (log 20260927-214838)
    "sensitivity": "medium",
    "onboarded": False,     # the setup guide has been completed
    "show_home": True,      # show the Nexus window when the app opens
    "login": False,         # open Nexus at login
    "calibration": None,    # your hand's measured pinch / spread levels (calibration.py)
}


def load(path: Path | None = None) -> dict:
    if path is None:
        paths.ensure_dirs()  # includes the one-time move from ~/.config/nexus
    path = path or PATH
    try:
        saved = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        saved = {}
    if saved.get("wake_threshold") == 0.5:  # the old default, saved by an earlier version: use the new one
        saved.pop("wake_threshold")
    if saved.get("sensitivity") == "high":  # same: "high" was the old default, not a choice you made
        saved.pop("sensitivity")
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def save(values: dict, path: Path | None = None):
    path = path or PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: values[k] for k in DEFAULTS if k in values}, indent=2))


def update(path: Path | None = None, **changes) -> dict:
    path = path or PATH
    s = load(path)
    s.update({k: v for k, v in changes.items() if k in DEFAULTS})
    save(s, path)
    return s
