"""Jarvis's voice: macOS `say` with the best installed voice (Nathan (Enhanced) if you have it),
plus short system sounds for feedback. Runs out of process, so it can be cut off instantly."""
import re
import subprocess

from .log import log

SOUNDS = "/System/Library/Sounds"


def chime(name: str, volume: float = 0.5):
    """Pop = listening, Tink = got it, Bottle = gave up, Basso = error."""
    subprocess.Popen(["afplay", "-v", str(volume), f"{SOUNDS}/{name}.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def installed_voices() -> list[tuple[str, str]]:
    """(name, locale) for every voice `say` knows, e.g. ("Nathan (Enhanced)", "en_US")."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    voices = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s{2,}([a-z]{2}_[A-Z0-9]+)\s", line)
        if m:
            voices.append((m.group(1).strip(), m.group(2)))
    return voices


def pick_voice(wanted: str | None = None) -> str | None:
    voices = installed_voices()
    english = [v for v in voices if v[1].startswith("en")]
    if wanted:
        hit = next((n for n, _ in english if wanted.lower() in n.lower()), None)
        if hit:
            return hit
    for quality in ("(Premium)", "(Enhanced)"):
        for preferred in ("Nathan", "Zoe", "Jamie", "Ava", "Evan"):
            hit = next((n for n, _ in english if n.startswith(preferred) and quality in n), None)
            if hit:
                return hit
        hit = next((n for n, _ in english if quality in n), None)
        if hit:
            return hit
    return None  # system default voice


class Speaker:
    def __init__(self, voice: str | None = None):
        self.voice = pick_voice(voice)
        self._proc = None
        log.info("voice", "tts_voice", name=self.voice or "system default",
                 enhanced=bool(self.voice and ("Enhanced" in self.voice or "Premium" in self.voice)))

    def say(self, text: str):
        self.stop()
        cmd = ["say"] + (["-v", self.voice] if self.voice else []) + [text]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @property
    def speaking(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def wait(self, timeout: float = 30):
        if self._proc:
            try:
                self._proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.stop()

    def stop(self):
        if self.speaking:
            self._proc.terminate()
            log.info("voice", "tts_interrupted")
