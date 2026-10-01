"""Every app on this Mac (folder scan + Spotlight), with fuzzy matching.
"apple music" -> Music, "word" -> Microsoft Word, "vs code" -> Visual Studio Code."""
import difflib
import plistlib
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .log import log

ALIASES = {
    "apple music": "Music", "itunes": "Music", "music app": "Music",
    "word": "Microsoft Word", "ms word": "Microsoft Word", "excel": "Microsoft Excel",
    "powerpoint": "Microsoft PowerPoint", "power point": "Microsoft PowerPoint",
    "outlook": "Microsoft Outlook", "teams": "Microsoft Teams",
    "chrome": "Google Chrome", "vs code": "Visual Studio Code", "vscode": "Visual Studio Code",
    "visual studio": "Visual Studio Code", "settings": "System Settings", "system preferences": "System Settings",
    "preferences": "System Settings", "app store": "App Store", "imessage": "Messages", "zoom": "zoom.us",
    "text edit": "TextEdit", "quicktime": "QuickTime Player", "activity monitor": "Activity Monitor",
    "facetime": "FaceTime", "files": "Finder",
}


def normalize(s: str) -> str:
    s = s.lower().replace(".app", "")
    return " ".join("".join(c if c.isalnum() else " " for c in s).split())


def loosen(s: str) -> str:
    n = normalize(s)
    for vendor in ("apple ", "microsoft ", "google ", "adobe "):
        if n.startswith(vendor):
            n = n[len(vendor):]
    return n.replace(" ", "")


@dataclass
class App:
    name: str
    path: str
    key: str
    loose: str


class AppIndex:
    def __init__(self):
        self.apps: list[App] = []
        self.ready = threading.Event()

    def load(self):
        threading.Thread(target=self._scan, name="nexus-apps", daemon=True).start()

    def _scan(self):
        start = time.perf_counter()
        seen, found = set(), []

        def add(path: Path):
            p = str(path.resolve())
            if p in seen or not path.exists():
                return
            seen.add(p)
            name = path.stem
            try:
                with open(path / "Contents" / "Info.plist", "rb") as f:
                    info = plistlib.load(f)
                name = info.get("CFBundleDisplayName") or info.get("CFBundleName") or name
            except Exception:
                pass
            for n in {name, path.stem}:
                found.append(App(n, p, normalize(n), loosen(n)))

        home = Path.home()
        for d in [Path("/Applications"), Path("/Applications/Utilities"), Path("/System/Applications"),
                  Path("/System/Applications/Utilities"), home / "Applications"]:
            if not d.exists():
                continue
            for item in d.iterdir():
                if item.suffix == ".app":
                    add(item)
                elif item.is_dir():  # one level of subfolders, e.g. Microsoft Office
                    for inner in item.glob("*.app"):
                        add(inner)
        add(Path("/System/Library/CoreServices/Finder.app"))
        try:
            out = subprocess.run(["mdfind", "kMDItemContentType == 'com.apple.application-bundle'"],
                                 capture_output=True, text=True, timeout=8).stdout
            for line in out.splitlines():
                if line.endswith(".app") and ".app/" not in line and not line.startswith(("/System/Library", "/Library/Apple")):
                    add(Path(line))
        except Exception as e:
            log.warn("apps", "spotlight_failed", error=str(e))
        self.apps = found
        self.ready.set()
        log.info("apps", "indexed", count=len(found), ms=int((time.perf_counter() - start) * 1000))

    def resolve(self, spoken: str) -> App | None:
        q = normalize(spoken)
        for filler in ("the ", "my "):
            q = q[len(filler):] if q.startswith(filler) else q
        for filler in (" app", " application"):
            q = q[: -len(filler)] if q.endswith(filler) else q
        if not q:
            return None
        if q in ALIASES:
            hit = next((a for a in self.apps if a.key == normalize(ALIASES[q])), None)
            if hit:
                return hit
        lq = loosen(q)
        best, best_score = None, 1e9
        for a in self.apps:
            if a.key == q:
                score = 0
            elif a.loose == lq:
                score = 1
            elif len(lq) >= 3 and (a.key.startswith(q + " ") or a.loose.startswith(lq)):
                score = 2 + len(a.key) - len(q)
            elif f" {q} " in f" {a.key} ":
                score = 20 + len(a.key)
            elif len(lq) >= 4 and difflib.SequenceMatcher(None, a.loose, lq).ratio() >= 0.8:
                score = 40
            else:
                continue
            if score < best_score:
                best, best_score = a, score
        return best

    def names(self, limit: int = 60) -> list[str]:
        return sorted({a.name for a in self.apps})[:limit]


index = AppIndex()
