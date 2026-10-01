"""Guided gesture recording: the training (and test) data for the gesture model.

    python -m nexus.collect              # 25 of each gesture + 60 s of free movement, ~5 minutes
    python -m nexus.collect --reps 10    # quicker session

A window shows what to do next ("NEXT: SWIPE LEFT"), then "GO" while it records 1.2 s.
Keys: q = stop and save, r = redo the previous clip.
Each session is saved to data/gestures/<time>.npz. Record 2-3 sessions (different lighting,
standing/sitting): training holds one session out for testing, which is the honest way to
measure accuracy.
"""
import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .config import Config
from .gesture_features import CLASSES, fill_gaps, resample
from .log import log

INSTRUCTIONS = {
    "idle": "Move your hand naturally. No pinch, no swipe.",
    "pinch": "Tap thumb and index together once.",
    "swipe_left": "Swipe your hand to the LEFT.",
    "swipe_right": "Swipe your hand to the RIGHT.",
    "open_palm": "Hold up an open palm, fingers spread.",
}
READY_S, CLIP_S = 1.0, 1.2
YELLOW, GREEN, WHITE, RED = (60, 200, 255), (100, 230, 120), (255, 255, 255), (80, 80, 255)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


class Session:
    """Camera + hand tracker + a window with prompts. Shared by collect and bench."""

    def __init__(self, title: str):
        from .hands import HandTracker
        self.cfg = Config()
        self.cam = Camera(self.cfg.camera_index, self.cfg.camera_width, self.cfg.camera_height, self.cfg.camera_fps)
        self.cam.start()
        self.tracker = HandTracker(self.cfg.model_path)
        self.title = title
        self.seq = 0
        self.t0 = time.perf_counter()

    def frame(self):
        got = None
        while got is None:
            got = self.cam.latest(self.seq, 1.0)
        self.seq, bgr, t = got
        v0 = time.perf_counter()
        hand = self.tracker.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int((t - self.t0) * 1000))
        return bgr, hand, t, (time.perf_counter() - v0) * 1000

    def show(self, bgr, hand, lines, color=WHITE, progress: float | None = None) -> int:
        h, w = bgr.shape[:2]
        view = cv2.flip(bgr, 1)
        if hand is not None:
            pts = [(int((1 - x) * w), int(y * h)) for x, y, _ in hand.landmarks]
            for a, b in BONES:
                cv2.line(view, pts[a], pts[b], (255, 255, 255), 2)
        cv2.rectangle(view, (0, 0), (w, 30 + 30 * len(lines)), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(view, line, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75 if i == 0 else 0.6,
                        color if i == 0 else WHITE, 2 if i == 0 else 1)
        if progress is not None:
            cv2.rectangle(view, (0, h - 8), (int(w * progress), h), color, -1)
        if hand is None:
            cv2.putText(view, "no hand", (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
        cv2.imshow(self.title, view)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        self.cam.stop()
        self.tracker.close()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nexus.collect")
    ap.add_argument("--reps", type=int, default=25, help="clips per gesture")
    ap.add_argument("--free", type=int, default=60, help="seconds of free movement (false-activation test)")
    ap.add_argument("--out", default="data/gestures")
    args = ap.parse_args(argv)
    log.open()
    s = Session("NEXUS: gesture recording")
    plan = [c for c in CLASSES for _ in range(args.reps)]
    random.Random().shuffle(plan)  # mixed order, so the model can't learn "what came before"
    imgs, worlds, labels = [], [], []
    free_img, free_world, free_valid, free_t = [], [], [], []
    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    log.info("collect", "start", session=session, clips=len(plan), free_s=args.free)
    try:
        i = 0
        while i < len(plan):
            label = plan[i]
            end = time.perf_counter() + READY_S
            while time.perf_counter() < end:
                bgr, hand, _, _ = s.frame()
                key = s.show(bgr, hand, [f"NEXT: {label.upper()}   ({i + 1}/{len(plan)})", INSTRUCTIONS[label],
                                         "q = stop and save   r = redo previous"], YELLOW)
                if key == ord("q"):
                    raise KeyboardInterrupt
                if key == ord("r") and labels:
                    imgs.pop(); worlds.pop(); labels.pop()
                    i -= 1
                    label = plan[i]
                    end = time.perf_counter() + READY_S
            img, world, valid = [], [], []
            start = time.perf_counter()
            while time.perf_counter() - start < CLIP_S:
                bgr, hand, _, _ = s.frame()
                ok = hand is not None and hand.world is not None
                img.append(hand.landmarks if ok else np.zeros((21, 3), np.float32))
                world.append(hand.world if ok else np.zeros((21, 3), np.float32))
                valid.append(ok)
                s.show(bgr, hand, [f"GO: {label.upper()}", INSTRUCTIONS[label]], GREEN,
                       (time.perf_counter() - start) / CLIP_S)
            filled = fill_gaps(np.array(img), np.array(world), np.array(valid))
            if filled is None:
                log.info("collect", "clip_skipped", label=label, reason="hand not visible enough")
                continue  # same prompt again
            imgs.append(resample(filled[0]))
            worlds.append(resample(filled[1]))
            labels.append(CLASSES.index(label))
            i += 1

        if args.free > 0:
            end = time.perf_counter() + 3
            while time.perf_counter() < end:
                bgr, hand, _, _ = s.frame()
                s.show(bgr, hand, [f"NEXT: FREE MOVEMENT ({args.free} s)",
                                   "Use your hand normally: point around, rest it, move it.",
                                   "Don't do the gestures on purpose."], YELLOW)
            start = time.perf_counter()
            while time.perf_counter() - start < args.free:
                bgr, hand, t, _ = s.frame()
                ok = hand is not None and hand.world is not None
                free_img.append(hand.landmarks if ok else np.zeros((21, 3), np.float32))
                free_world.append(hand.world if ok else np.zeros((21, 3), np.float32))
                free_valid.append(ok)
                free_t.append(t)
                if s.show(bgr, hand, ["FREE MOVEMENT", "Just move naturally."], GREEN,
                          (time.perf_counter() - start) / args.free) == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        s.close()
    if not labels:
        print("Nothing recorded.")
        log.close()
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{session}.npz"
    np.savez_compressed(path, img=np.array(imgs, np.float32), world=np.array(worlds, np.float32),
                        labels=np.array(labels), classes=np.array(CLASSES),
                        free_img=np.array(free_img, np.float32).reshape(-1, 21, 3),
                        free_world=np.array(free_world, np.float32).reshape(-1, 21, 3),
                        free_valid=np.array(free_valid, bool), free_t=np.array(free_t))
    counts = {c: int(np.sum(np.array(labels) == k)) for k, c in enumerate(CLASSES)}
    log.info("collect", "saved", file=str(path), clips=len(labels), per_class=counts, free_frames=len(free_t))
    print(f"\nSaved {len(labels)} clips + {len(free_t)} free-movement frames to {path}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
