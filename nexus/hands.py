"""MediaPipe Hand Landmarker (Tasks API, VIDEO mode): 21 landmarks + 3D world landmarks."""
from pathlib import Path

import numpy as np

from .landmarks import Hand
from .log import log


class HandTracker:
    def __init__(self, model_path: str, max_hands: int = 1):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} missing: run scripts/setup.sh to download the hand model")
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,  # tracks between frames: faster than re-detecting
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1
        self._failures = 0
        log.info("vision", "hand_landmarker_ready", model=str(model_path), mediapipe=mp.__version__)

    def detect(self, rgb: np.ndarray, ts_ms: int) -> Hand | None:
        ts_ms = max(ts_ms, self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts_ms
        try:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(image, ts_ms)
        except Exception as e:
            self._failures += 1
            if self._failures <= 3 or self._failures % 100 == 0:
                log.warn("vision", "detect_failed", error=str(e), count=self._failures)
            return None
        if not result.hand_landmarks:
            return None
        lms = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        world = None
        if result.hand_world_landmarks:
            world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], dtype=np.float32)
        cat = result.handedness[0][0] if result.handedness else None
        return Hand(landmarks=lms, world=world, score=float(cat.score) if cat else 0.0,
                    handedness=cat.category_name if cat else "")

    def close(self):
        self._landmarker.close()
