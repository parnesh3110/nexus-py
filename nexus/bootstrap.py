"""Nexus.app's entry point. The app's own executable (scripts/nexus_stub.c) runs this inside its
process, so macOS credits everything to "Nexus" (permissions, Dock, app switcher).

Before the app starts:
  - loads the project's .venv packages
  - sends output to logs/app-output.log (a Finder-launched app has no terminal)
  - installs new dependencies when requirements.txt changed
  - starts Ollama (the local LLM) if it isn't running
Then runs the app. Exit code 75 means "my code was updated": re-launch this same executable, so
the new code runs and macOS still sees Nexus. A crash right after starting shows a dialog.
Only the standard library is imported here, so an update can replace any package safely.
"""
import hashlib
import os
import site
import subprocess
import sys
import time
import traceback

BUNDLED = os.environ.get("NEXUS_BUNDLED") == "1"  # a standalone release build (scripts/build_release.sh)
ROOT = os.environ.get("NEXUS_RESOURCES") if BUNDLED else os.getcwd()  # dev: the stub chdir'd into the project
OUT = (os.path.expanduser("~/Library/Logs/Nexus/app-output.log") if BUNDLED
       else os.path.join(ROOT, "logs", "app-output.log"))


def _log(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} bootstrap: {msg}", flush=True)


def _dialog(text: str):
    subprocess.run(["osascript", "-e", f'display dialog "{text}" buttons {{"OK"}} default button 1 with title "Nexus"'],
                   capture_output=True)


def _notify(text: str):
    subprocess.run(["osascript", "-e", f'display notification "{text}" with title "Nexus"'], capture_output=True)


def _redirect_output():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    f = open(OUT, "a", buffering=1)
    os.dup2(f.fileno(), 1)  # fd level, so native libraries (MediaPipe, PortAudio) land here too
    os.dup2(f.fileno(), 2)
    sys.stdout = sys.stderr = f


def _sync_dependencies():
    req = os.path.join(ROOT, "requirements.txt")
    marker = os.path.join(ROOT, ".venv", ".nexus-requirements")
    want = hashlib.sha1(open(req, "rb").read()).hexdigest()
    have = open(marker).read().strip() if os.path.exists(marker) else ""
    if want == have:
        return
    _notify("Installing an update. Nexus opens in a moment.")
    _log("requirements.txt changed, installing")
    env = {**os.environ, "VIRTUAL_ENV": os.path.join(ROOT, ".venv")}
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    r = subprocess.run(["uv", "pip", "install", "-r", req], env=env, cwd=ROOT)
    if r.returncode == 0:
        with open(marker, "w") as m:
            m.write(want)
        _log("dependencies up to date")
    else:
        _log("dependency install failed")
        _notify("An update could not be installed. Details are in logs/app-output.log.")


def _start_ollama():
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.5)
    except Exception:
        if os.path.exists("/Applications/Ollama.app"):
            subprocess.run(["open", "-g", "-a", "Ollama"], capture_output=True)  # its app runs the server
        elif not BUNDLED:
            subprocess.run(["brew", "services", "start", "ollama"], capture_output=True)


def main() -> int:
    _redirect_output()
    # Only needed to start the interpreter; don't leak them into commands Jarvis runs
    os.environ.pop("PYTHONHOME", None)
    os.environ.pop("PYTHONPATH", None)
    if BUNDLED:  # everything ships inside the app
        venv_site = os.path.join(ROOT, "site-packages")
    else:
        venv_site = os.path.join(ROOT, ".venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                                 "site-packages")
    if not os.path.isdir(venv_site):
        _dialog("Nexus is not set up yet. In Terminal, go to the nexus-py folder and run: bash scripts/setup.sh"
                if not BUNDLED else "This copy of Nexus is incomplete. Please download it again.")
        return 1
    exe = sys.executable
    os.environ["NEXUS_APP"] = exe.split("/Contents/MacOS/")[0] if "/Contents/MacOS/" in exe else ""
    _log(f"starting ({exe}){' [release build]' if BUNDLED else ''}")
    if not BUNDLED:  # development: install whatever requirements.txt added since the last start
        try:
            _sync_dependencies()
        except Exception as e:
            _log(f"dependency check failed: {e!r}")
    _start_ollama()

    site.addsitedir(venv_site)  # after the sync, so freshly installed packages are what gets loaded
    started = time.time()
    try:
        from nexus.app import main as app_main
        code = app_main(["--app"])
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
    except Exception:
        traceback.print_exc()
        code = 1
    if code == 75:
        _log("restarting for an update")
        sys.stdout.flush()
        os.execv(exe, [exe])  # same executable = same identity for macOS; runs the new code
    _log(f"exited with code {code}")
    if code not in (0, None) and time.time() - started < 20:
        _dialog("Nexus stopped right after starting. The details are in logs/app-output.log in the nexus-py folder.")
    return code or 0


if __name__ == "__main__":
    sys.exit(main())
