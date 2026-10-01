"""Permission status and requests, for the setup guide. macOS grants these to Nexus.app
(or to your terminal when you run from Terminal)."""
import subprocess
import threading
import time
import urllib.request

from .log import log

PANES = {
    "camera": "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "automation": "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation",
}


def open_settings(kind: str):
    subprocess.run(["open", PANES[kind]])


def _av(media: str):
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio, AVMediaTypeVideo
        return AVCaptureDevice, (AVMediaTypeVideo if media == "camera" else AVMediaTypeAudio)
    except Exception:
        return None, None


def status(kind: str) -> str:
    """"granted" | "denied" | "ask" (not asked yet) | "unknown\""""
    if kind in ("camera", "microphone"):
        dev, media = _av(kind)
        if dev is None:
            return "unknown"
        s = int(dev.authorizationStatusForMediaType_(media))
        return {0: "ask", 1: "denied", 2: "denied", 3: "granted"}.get(s, "unknown")
    if kind == "accessibility":
        from .mouse import accessibility_trusted
        return "granted" if accessibility_trusted() else "ask"
    if kind == "automation":
        return _automation
    if kind == "ai":
        if _pulling:
            return "working"
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.4) as r:
                return "granted" if AI_MODEL.split(":")[0].encode() in r.read() else "ask"
        except Exception:
            return "missing"  # Ollama not installed, or not running
    return "unknown"


AI_MODEL = "qwen2.5:3b"
_pulling = False


_automation = "ask"


def request(kind: str):
    """Show macOS's own prompt where there is one; otherwise open the right Settings pane."""
    global _automation
    log.info("setup", "permission_request", kind=kind, before=status(kind))
    if kind in ("camera", "microphone"):
        dev, media = _av(kind)
        if dev is not None and status(kind) == "ask":
            dev.requestAccessForMediaType_completionHandler_(media, lambda granted: log.info(
                "setup", "permission_answer", kind=kind, granted=bool(granted)))
        else:
            open_settings(kind)  # already denied: only Settings can change it now
    elif kind == "accessibility":
        from .mouse import accessibility_trusted
        if not accessibility_trusted(prompt=True):
            open_settings("accessibility")
    elif kind == "automation":
        # A harmless script makes macOS ask "Nexus wants to control System Events" now, not mid-command
        def ask():
            global _automation
            r = subprocess.run(["osascript", "-e", 'tell application "System Events" to get name of first process'],
                               capture_output=True, text=True, timeout=60)
            _automation = "granted" if r.returncode == 0 else "denied"
            log.info("setup", "permission_answer", kind="automation", granted=r.returncode == 0)
        threading.Thread(target=ask, daemon=True).start()
    elif kind == "ai":
        threading.Thread(target=_prepare_ai, name="nexus-ai-setup", daemon=True).start()


def _prepare_ai():
    """Local AI, step by step: start Ollama if it's installed (its app or Homebrew), else open its
    download page; once it runs, download the model through Ollama's own API."""
    global _pulling
    import json
    import os
    if status("ai") == "missing":
        if os.path.exists("/Applications/Ollama.app"):
            subprocess.run(["open", "-a", "Ollama"])
        elif subprocess.run(["/bin/zsh", "-lc", "command -v ollama"], capture_output=True).returncode == 0:
            subprocess.run(["/bin/zsh", "-lc", "brew services start ollama || (ollama serve >/dev/null 2>&1 &)"],
                           capture_output=True, timeout=30)
        else:
            subprocess.run(["open", "https://ollama.com/download"])
            return
        for _ in range(40):  # up to 20 s for it to come up
            if status("ai") != "missing":
                break
            time.sleep(0.5)
    if status("ai") == "ask":
        _pulling = True
        log.info("setup", "ai_model_download", model=AI_MODEL)
        try:
            req = urllib.request.Request("http://localhost:11434/api/pull",
                                         data=json.dumps({"model": AI_MODEL, "stream": False}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3600).read()
            log.info("setup", "ai_model_ready", model=AI_MODEL)
        except Exception as e:
            log.warn("setup", "ai_model_download_failed", error=repr(e))
        finally:
            _pulling = False
