"""Rephrased command -> known command, with a sentence-transformers model (all-MiniLM-L6-v2,
a small Transformer, run locally on the CPU).

Every command has a few example phrasings. At startup they're embedded once; at runtime what
you said is embedded too, and the closest example (cosine similarity) wins if it's close enough.
"shrink this window" -> minimize, "it's too loud" -> volume down, in ~10 ms and without the LLM.
"""
import threading
import time

import numpy as np

from .commands import EXACT
from .log import log

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

# canonical command (an EXACT phrase in commands.py) -> other ways people say it
EXAMPLES = {
    "minimize": ["shrink this window", "make this window smaller", "put this window away", "hide this window for now"],
    "maximize": ["make this window bigger", "make it take up the whole screen", "expand this window"],
    "full screen": ["go into full screen", "make this full screen", "enter full screen mode"],
    "close this": ["close this window", "get rid of this window", "shut this window"],
    "left half": ["put this on the left side", "move this window to the left half"],
    "right half": ["put this on the right side", "move this window to the right half"],
    "new tab": ["open a new tab", "give me a fresh tab", "i need another tab"],
    "close tab": ["close this tab", "get rid of this tab"],
    "new window": ["open a new window", "give me another window"],
    "go back": ["go to the previous page", "take me back", "back one page"],
    "go forward": ["go to the next page", "forward one page"],
    "refresh": ["reload this page", "refresh the page", "load the page again"],
    "scroll down": ["scroll down a bit", "move the page down", "show me more below"],
    "scroll up": ["scroll up a bit", "move the page up"],
    "volume up": ["turn the volume up", "make it louder", "i can't hear it", "increase the volume"],
    "volume down": ["turn the volume down", "make it quieter", "it's too loud", "lower the volume"],
    "mute": ["mute the sound", "silence everything", "turn off the sound"],
    "unmute": ["turn the sound back on"],
    "pause music": ["pause the music", "stop the song", "stop playing music", "hold the music"],
    "resume music": ["resume the music", "keep playing the music", "continue the song", "play the music again"],
    "next song": ["skip this song", "play the next song", "skip to the next one", "next one"],
    "previous song": ["go back a song", "play the last song again", "previous one"],
    "undo": ["undo that", "take that back", "revert what i just did"],
    "redo": ["redo that", "do that again"],
    "copy": ["copy this", "copy the selected text"],
    "paste": ["paste it here", "paste what i copied"],
    "save": ["save this file", "save my work"],
    "select all": ["select everything", "highlight all of it"],
    "lock screen": ["lock my computer", "lock the mac"],
    "new note": ["make a new note", "start a new document", "create a new file"],
    "quit this app": ["quit this application", "close this whole app"],
}
THRESHOLD = 0.72       # cosine similarity needed to accept a match
STRICT_THRESHOLD = 0.82
STRICT = {"close this", "quit this app", "lock screen", "close tab"}  # things that close or lock


class IntentRouter:
    def __init__(self, model: str = MODEL):
        self.model_name = model
        self.ready = threading.Event()
        self._model = None
        self._phrases: list[tuple[str, str]] = []  # (phrase, canonical)
        self._vecs = None
        threading.Thread(target=self._load, name="nexus-intents", daemon=True).start()

    def _load(self):
        start = time.perf_counter()
        for canonical, phrases in EXAMPLES.items():
            assert canonical in EXACT, f"{canonical} isn't a known command"
            for p in [canonical] + phrases:
                self._phrases.append((p, canonical))
        try:
            # fastembed: the same MiniLM model on ONNX Runtime (no PyTorch, ~500 MB smaller app)
            from fastembed import TextEmbedding
            from . import paths
            model = TextEmbedding(self.model_name, cache_dir=paths.embedding_cache())
            self._encode = lambda texts: np.array(list(model.embed(texts)), dtype=np.float32)
            backend = "fastembed-onnx"
        except Exception as first:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.model_name, device="cpu")
                self._encode = lambda texts: np.asarray(model.encode(texts), dtype=np.float32)
                backend = "sentence-transformers"
            except Exception as e:
                log.warn("nlu", "unavailable", error=f"{first!r} / {e!r}", note="rephrased commands go to the LLM")
                return
        try:
            self._vecs = _normalize(self._encode([p for p, _ in self._phrases]))
            self.ready.set()
            log.info("nlu", "ready", model=self.model_name, backend=backend, examples=len(self._phrases),
                     intents=len(EXAMPLES), dims=int(self._vecs.shape[1]), ms=int((time.perf_counter() - start) * 1000))
        except Exception as e:
            log.warn("nlu", "unavailable", error=repr(e), note="rephrased commands go to the LLM")

    def route(self, text: str) -> tuple[str, float, str] | None:
        """(canonical command, similarity, matched example), or None if nothing is close enough."""
        if not self.ready.is_set():
            return None
        start = time.perf_counter()
        v = _normalize(self._encode([text.lower()]))[0]
        sims = self._vecs @ v
        i = int(sims.argmax())
        phrase, canonical = self._phrases[i]
        sim = float(sims[i])
        limit = STRICT_THRESHOLD if canonical in STRICT else THRESHOLD
        accepted = sim >= limit
        # Logged every time: the data for tuning the thresholds
        log.info("nlu", "match" if accepted else "no_match", text=text, nearest=canonical, via=phrase,
                 similarity=sim, limit=limit, ms=int((time.perf_counter() - start) * 1000))
        return (canonical, sim, phrase) if accepted else None
