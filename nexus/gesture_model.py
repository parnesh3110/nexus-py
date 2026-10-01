"""Temporal gesture classifier (PyTorch) and the live recognizer that runs it every frame.

Model: 1D temporal CNN over a 30-frame (~1 s) window of landmark features. Three conv layers
with growing dilation see ~1 s of motion; mean + max pooling over time; linear head. ~55k
parameters: under a millisecond per prediction on the CPU.
"""
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .gesture_features import CLASSES, FEATURES, WINDOW, clip_features
from .log import log


class GestureNet(nn.Module):
    def __init__(self, features: int = FEATURES, classes: int = len(CLASSES), hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(features, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, features) -> logits (batch, classes)"""
        z = self.conv(x.transpose(1, 2))
        z = torch.cat([z.mean(dim=-1), z.amax(dim=-1)], dim=1)
        return self.head(self.drop(z))


def save(path, model: GestureNet, mean: np.ndarray, std: np.ndarray, info: dict):
    torch.save({"state": model.state_dict(), "classes": CLASSES, "window": WINDOW, "features": FEATURES,
                "mean": mean.tolist(), "std": std.tolist(), "info": info}, path)


def load(path) -> tuple[GestureNet, np.ndarray, np.ndarray, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GestureNet(ckpt["features"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, np.array(ckpt["mean"], np.float32), np.array(ckpt["std"], np.float32), ckpt


class GestureRecognizer:
    """Feeds the last 30 frames to the model every frame; fires an event when a gesture is
    confidently seen for a few frames in a row, then waits out a cooldown."""

    def __init__(self, path: str, threshold: float = 0.85, confirm_frames: int = 3, cooldown: float = 0.9):
        self.model, self.mean, self.std, ckpt = load(path)
        self.classes = ckpt["classes"]
        self.threshold, self.confirm_frames, self.cooldown = threshold, confirm_frames, cooldown
        self.img: deque = deque(maxlen=WINDOW)
        self.world: deque = deque(maxlen=WINDOW)
        self._streak = ("idle", 0)
        self._last_event = 0.0
        self.last = ("idle", 0.0)
        self.infer_ms = 0.0
        torch.set_num_threads(1)  # tiny model: one thread is fastest and leaves the CPU to MediaPipe
        log.info("gesture_model", "loaded", path=str(path), classes=self.classes,
                 test_accuracy=ckpt.get("info", {}).get("test_accuracy"))

    def reset(self):
        self.img.clear()
        self.world.clear()
        self._streak = ("idle", 0)

    def update(self, hand, t: float) -> str | None:
        """Call every frame with the current Hand (or None). Returns a gesture name when one fires."""
        if hand is None or hand.world is None:
            self.reset()
            return None
        self.img.append(hand.landmarks)
        self.world.append(hand.world)
        if len(self.img) < WINDOW:
            return None
        start = time.perf_counter()
        x = (clip_features(np.stack(self.img), np.stack(self.world)) - self.mean) / self.std
        with torch.no_grad():
            p = torch.softmax(self.model(torch.from_numpy(x[None])), dim=1)[0].numpy()
        self.infer_ms = (time.perf_counter() - start) * 1000
        i = int(p.argmax())
        label, prob = self.classes[i], float(p[i])
        self.last = (label, prob)
        if label == "idle" or prob < self.threshold:
            self._streak = ("idle", 0)
            return None
        n = self._streak[1] + 1 if self._streak[0] == label else 1
        self._streak = (label, n)
        if n >= self.confirm_frames and t - self._last_event > self.cooldown:
            self._last_event = t
            self._streak = ("idle", 0)
            self.img.clear()  # start fresh, so one gesture can't fire twice
            self.world.clear()
            log.info("gesture_model", "event", gesture=label, prob=prob, infer_ms=self.infer_ms)
            return label
        return None
