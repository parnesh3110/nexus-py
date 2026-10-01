"""Where Nexus keeps things. Two modes:

- Development (running this project from Terminal or the dev Nexus.app): models/, logs/, runs/,
  results/ inside the project folder, like before.
- Release (a standalone Nexus.app built by scripts/build_release.sh; NEXUS_BUNDLED=1): models ship
  inside the app, and your files go where Mac apps keep them:
      settings, recordings, results   ~/Library/Application Support/Nexus
      logs                            ~/Library/Logs/Nexus
Settings live in Application Support in both modes (moved from ~/.config/nexus once).
"""
import os
import shutil
from pathlib import Path

BUNDLED = os.environ.get("NEXUS_BUNDLED") == "1"
PROJECT = Path(__file__).resolve().parent.parent
APP_SUPPORT = Path.home() / "Library" / "Application Support" / "Nexus"

if BUNDLED:
    RESOURCES = Path(os.environ.get("NEXUS_RESOURCES", PROJECT))
    MODELS = RESOURCES / "models"
    LOGS = Path.home() / "Library" / "Logs" / "Nexus"
    DATA = APP_SUPPORT
else:
    RESOURCES = PROJECT
    MODELS = PROJECT / "models"
    LOGS = PROJECT / "logs"
    DATA = PROJECT

SETTINGS = APP_SUPPORT / "settings.json"
OLD_SETTINGS = Path.home() / ".config" / "nexus" / "settings.json"


def hand_model() -> Path:
    return MODELS / "hand_landmarker.task"


def gesture_model() -> Path:
    """Your own trained gesture model (it's yours, so it lives with your data, not in the app)."""
    return DATA / "models" / "gesture.pt"


def whisper_model(name: str) -> str:
    """A Whisper model folder shipped inside the app, or the name (downloaded on first use)."""
    local = MODELS / f"whisper-{name}"
    return str(local) if local.is_dir() else name


def embedding_cache() -> str | None:
    """Where the intent model's files are: inside the app when shipped, else the default cache."""
    local = MODELS / "fastembed"
    return str(local) if local.is_dir() else None


def ensure_dirs():
    for d in (APP_SUPPORT, LOGS, DATA):
        d.mkdir(parents=True, exist_ok=True)
    if not SETTINGS.exists() and OLD_SETTINGS.exists():  # one-time move from the old location
        shutil.copy2(OLD_SETTINGS, SETTINGS)
