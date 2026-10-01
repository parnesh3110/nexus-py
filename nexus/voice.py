"""Local voice: microphone -> "Hey Jarvis" wake word (openWakeWord) -> record until you stop
talking -> speech to text (faster-whisper). Nothing leaves the Mac.

Microphone access is granted to your terminal app (System Settings -> Privacy & Security -> Microphone).
"""
import math
import queue
import time
from collections import deque

import numpy as np

from .log import log

RATE = 16000
CHUNK = 1280  # 80 ms at 16 kHz: the frame size openWakeWord expects


def rms(chunk: np.ndarray) -> float:
    x = chunk.astype(np.float32) / 32768.0
    return float(math.sqrt(float(np.mean(x * x)))) if len(x) else 0.0


class Resampler:
    """The mic's own rate (usually 48 kHz) -> 16 kHz, streaming, in exact 1280-sample chunks.

    Why: opening the mic at 16 kHz can make macOS reconfigure the audio hardware, which can change
    how music sounds while Nexus listens. Recording at the device's native rate and converting
    here leaves the hardware alone. A low-pass filter first (anti-aliasing, state kept between
    blocks so there are no clicks at block edges), then linear interpolation with a carried phase."""

    def __init__(self, native: int, out_rate: int = RATE, chunk: int = CHUNK):
        self.native, self.ratio, self.chunk = native, native / out_rate, chunk
        self.pos = 0.0
        self.prev = np.zeros(1, np.float32)
        self.pending = np.zeros(0, np.float32)
        self.taps = None
        if native != out_rate:
            try:
                from scipy.signal import firwin, lfilter
                self.taps = firwin(63, 0.45 * out_rate, fs=native)
                self.zi = np.zeros(len(self.taps) - 1)
                self._lfilter = lfilter
            except Exception:  # scipy missing: skip the filter (slightly more aliasing, still works)
                self.taps = None

    def _convert(self, x: np.ndarray) -> np.ndarray:
        if self.native == RATE:
            return x
        if self.taps is not None:
            x, self.zi = self._lfilter(self.taps, 1.0, x, zi=self.zi)
        buf = np.concatenate([self.prev, x.astype(np.float32)])  # index 0 = last sample of the previous block
        if len(buf) - 1 < self.pos:
            self.pos -= len(buf) - 1
            self.prev = buf[-1:]
            return np.zeros(0, np.float32)
        n = int((len(buf) - 1 - self.pos) // self.ratio) + 1
        t = self.pos + np.arange(n) * self.ratio
        out = np.interp(t, np.arange(len(buf)), buf)
        self.pos = t[-1] + self.ratio - (len(buf) - 1)
        self.prev = buf[-1:]
        return out.astype(np.float32)

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        """float32 block at the native rate -> zero or more int16 chunks of exactly `chunk` samples."""
        self.pending = np.concatenate([self.pending, self._convert(x)])
        out = []
        while len(self.pending) >= self.chunk:
            c, self.pending = self.pending[: self.chunk], self.pending[self.chunk:]
            out.append(np.clip(c * 32768.0, -32768, 32767).astype(np.int16))
        return out


BLUETOOTH_HINTS = ("airpods", "bluetooth", "headset", "beats", "buds")


def pick_input(wanted: str | None = None):
    """Which microphone to use: `wanted` (any part of its name) if given; otherwise the default,
    unless the default is Bluetooth (AirPods): then the MacBook's own mic. Bluetooth mics drop to
    phone-call quality when recording, and "Hey Jarvis" never triggered through them
    (log 20260927-213438: Whisper heard "We'll see you in the next video.")."""
    import sounddevice as sd
    devs = sd.query_devices()
    inputs = [(i, d["name"]) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    log.info("voice", "inputs", available=[n for _, n in inputs])
    if wanted:
        hit = next((i for i, n in inputs if wanted.lower() in n.lower()), None)
        if hit is not None:
            return hit, "--mic"
        log.warn("voice", "mic_not_found", wanted=wanted)
    default = sd.default.device[0]
    default_name = devs[default]["name"].lower() if default is not None and default >= 0 else ""
    if any(h in default_name for h in BLUETOOTH_HINTS):
        builtin = next((i for i, n in inputs if "macbook" in n.lower() or "built-in" in n.lower()), None)
        if builtin is not None:
            return builtin, "default was Bluetooth, using the built-in mic"
    return None, "system default"


class Mic:
    """16 kHz mono int16 chunks from the default input, pushed into a queue by PortAudio."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue(maxsize=250)
        self.noise = 0.004  # running estimate of background loudness
        self.level = 0.0    # latest loudness, for the UI
        self.device_name = ""
        self._stream = None
        self.overflows = 0

    def start(self, wanted: str | None = None):
        import sounddevice as sd
        if self._stream is not None:  # already open
            return
        device, reason = pick_input(wanted)
        dev = sd.query_devices(device if device is not None else sd.default.device[0])
        native = int(round(dev.get("default_samplerate") or RATE))
        self._resampler = Resampler(native)
        extra = {}
        try:  # never let PortAudio change the device's own settings (rate, buffer size)
            extra["extra_settings"] = sd.CoreAudioSettings(change_device_parameters=False)
        except Exception:
            pass
        self._stream = sd.InputStream(samplerate=native, channels=1, dtype="float32",
                                      blocksize=int(round(native * CHUNK / RATE)),
                                      callback=self._callback, device=device, **extra)
        self._stream.start()
        self.device_name = dev.get("name", "?")
        log.info("voice", "mic_started", device=self.device_name, why=reason, native_rate=native, rate=RATE,
                 chunk_ms=int(CHUNK / RATE * 1000))

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.overflows += 1
        for chunk in self._resampler(indata[:, 0]):
            try:
                self.q.put_nowait(chunk)
            except queue.Full:
                pass  # processing fell behind: drop audio rather than lag

    def read(self, timeout: float = 1.0) -> np.ndarray:
        c = self.q.get(timeout=timeout)
        self.level = rms(c)
        # Track the noise floor slowly; only from quiet-ish chunks
        if self.level < self.noise * 2:
            self.noise = self.noise * 0.98 + self.level * 0.02
        else:
            self.noise = self.noise * 0.999 + self.level * 0.001
        return c

    def drain(self):
        """Throw away queued audio (e.g. what was heard while Jarvis was talking)."""
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class WakeWord:
    """openWakeWord's pre-trained "hey jarvis" model: a small neural net over audio embeddings,
    scored every 80 ms."""

    def __init__(self, threshold: float = 0.35):
        from openwakeword.model import Model
        self.model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        self.threshold = threshold
        log.info("voice", "wake_word_ready", models=list(self.model.models.keys()), threshold=threshold)

    def score(self, chunk: np.ndarray) -> float:
        preds = self.model.predict(chunk)
        return float(max(preds.values())) if preds else 0.0

    def reset(self):
        self.model.reset()


def record_utterance(mic: Mic, start_timeout: float = 5.0, end_silence: float = 1.0, max_s: float = 15.0,
                     should_stop=None):
    """Record from when you start talking until you pause. None if you never started, or if
    `should_stop()` says so (the panel closed, or you typed something instead)."""
    threshold = max(0.012, mic.noise * 3)
    pre = deque(maxlen=4)  # ~0.3 s from before speech started, so the first word isn't clipped
    chunks, started = [], False
    t0 = last_voice = time.perf_counter()
    while True:
        if should_stop is not None and should_stop():
            return None
        try:
            c = mic.read(0.5)
        except queue.Empty:
            if time.perf_counter() - t0 > start_timeout and not started:
                return None
            continue
        now = time.perf_counter()
        loud = mic.level > threshold
        if not started:
            pre.append(c)
            if loud:
                started, last_voice = True, now
                chunks.extend(pre)
            elif now - t0 > start_timeout:
                return None
        else:
            chunks.append(c)
            if mic.level > threshold * 0.7:
                last_voice = now
            if now - last_voice > end_silence or now - t0 > max_s:
                break
    audio = np.concatenate(chunks)
    log.debug("voice", "utterance", seconds=len(audio) / RATE, threshold=threshold)
    return audio


# Words Whisper should expect. It uses this as context, the way a person listening to commands would.
# Log 20260927-214838: "pause the music" came out as "Buy as the music".
COMMAND_HINTS = ("Jarvis. Pause the music. Resume the music. Next song. Previous song. Play The Reason. "
                 "Volume up. Volume down. Mute. Minimize. Maximize. Full screen. New tab. Close tab. "
                 "Go back. Scroll down. Draft what I'm saying. Stop dictation.")


class Transcriber:
    """faster-whisper (Whisper on CTranslate2, int8 on CPU). `hints` bias it toward words it
    should expect, like app names, the same way a human listener uses context."""

    def __init__(self, model: str = "base.en"):
        from faster_whisper import WhisperModel

        from . import paths
        start = time.perf_counter()
        self.model = WhisperModel(paths.whisper_model(model), device="cpu", compute_type="int8")
        self.name = model
        self.hints = COMMAND_HINTS + " Open Safari, Notes, Music, Microsoft Word."
        log.info("voice", "whisper_ready", model=model, load_ms=int((time.perf_counter() - start) * 1000))

    def set_hints(self, app_names: list[str]):
        self.hints = COMMAND_HINTS + " Open " + ", ".join(app_names[:40]) + "."

    def __call__(self, audio: np.ndarray) -> str:
        x = audio.astype(np.float32) / 32768.0
        # vad_filter drops the non-speech parts; the per-segment checks drop what Whisper invents
        # when it hears only noise (it tends to write YouTube outros like "see you in the next video")
        segments, _ = self.model.transcribe(x, language="en", beam_size=1, without_timestamps=True,
                                            condition_on_previous_text=False, initial_prompt=self.hints,
                                            vad_filter=True, vad_parameters={"min_silence_duration_ms": 400})
        kept, dropped = [], []
        for s in segments:
            # Whisper's own rule: skip only if it looks like silence AND it's unsure. Using "or" threw
            # away short real commands (log 20260927-214838: "Minimize." was dropped).
            (dropped if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0 else kept).append(s.text.strip())
        text = " ".join(kept).strip()
        if text and is_hallucination(text):
            dropped.append(text)
            text = ""
        if dropped:
            log.info("voice", "stt_dropped", text=" | ".join(dropped)[:200])
        return text


HALLUCINATIONS = ("thanks for watching", "thank you for watching", "see you in the next video", "see you next time",
                  "please subscribe", "like and subscribe", "subtitles by", "transcribed by")


def is_hallucination(text: str) -> bool:
    t = text.lower().strip(" .!?,")
    return t in ("you", "bye", "thank you", "") or any(h in t for h in HALLUCINATIONS)
