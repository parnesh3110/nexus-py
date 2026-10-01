"""Landmark sequences -> model input features. Pure numpy, shared by collection, training,
benchmarking and live inference, so all four see exactly the same numbers.

Per frame (68 values):
  63  3D world landmarks relative to the wrist, divided by hand size   -> hand SHAPE (pinch, open palm)
   2  palm position relative to the window's first frame, in hand sizes -> hand PATH (swipes)
   2  palm velocity                                                      -> how fast
   1  pinch ratio (thumb-index gap / hand size)
"""
import numpy as np

CLASSES = ["idle", "pinch", "swipe_left", "swipe_right", "open_palm"]
WINDOW = 30          # frames per sample (~1 s at 30 fps)
FEATURES = 68
KNUCKLES = [5, 9, 13, 17]


def _hand_size_world(world: np.ndarray) -> np.ndarray:
    """(T, 21, 3) -> (T,) wrist to middle knuckle, meters."""
    return np.maximum(np.linalg.norm(world[:, 9] - world[:, 0], axis=-1), 1e-4)


def clip_features(img: np.ndarray, world: np.ndarray, aspect: float = 4 / 3) -> np.ndarray:
    """img, world: (T, 21, 3) landmarks for one window -> (T, 68) features."""
    T = img.shape[0]
    size_w = _hand_size_world(world)
    shape = (world - world[:, :1]) / size_w[:, None, None]            # wrist-relative, scale-free

    xy = img[:, :, :2].copy()
    xy[:, :, 0] = (1 - xy[:, :, 0]) * aspect                          # mirrored, x on the same scale as y
    palm = xy[:, KNUCKLES].mean(axis=1)                               # (T, 2)
    size_i = np.maximum(np.linalg.norm(xy[:, 9] - xy[:, 0], axis=-1), 1e-4)
    path = (palm - palm[0]) / size_i.mean()                           # displacement in hand sizes
    vel = np.vstack([np.zeros((1, 2)), np.diff(path, axis=0)]) * 10

    pinch = np.linalg.norm(world[:, 4] - world[:, 8], axis=-1) / size_w

    feats = np.concatenate([shape.reshape(T, 63), path, vel, pinch[:, None]], axis=1)
    return feats.astype(np.float32)


def resample(seq: np.ndarray, length: int = WINDOW) -> np.ndarray:
    """Linearly resample a (T, ...) sequence to `length` frames (clips come in at slightly varying fps)."""
    T = seq.shape[0]
    if T == length:
        return seq
    src = np.linspace(0, T - 1, length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo).reshape(-1, *([1] * (seq.ndim - 1)))
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(seq.dtype)


def fill_gaps(img: np.ndarray, world: np.ndarray, valid: np.ndarray):
    """Frames where the hand wasn't detected get the previous good frame. Returns None if more
    than 30% of the clip is missing (not worth keeping)."""
    if valid.mean() < 0.7 or not valid.any():
        return None
    img, world = img.copy(), world.copy()
    first = int(np.argmax(valid))
    img[:first], world[:first] = img[first], world[first]
    for t in range(first + 1, len(valid)):
        if not valid[t]:
            img[t], world[t] = img[t - 1], world[t - 1]
    return img, world


# --- rule-based baseline, evaluated on exactly the same clips as the model ---

def rule_baseline(img: np.ndarray, world: np.ndarray, aspect: float = 4 / 3) -> str:
    """What hand-written thresholds would say about a clip. This is the bar the model has to beat."""
    f = clip_features(img, world, aspect)
    pinch, path = f[:, -1], f[:, 63:65]
    closed = pinch < 0.14
    if np.any(closed[1:] & closed[:-1]):                   # same rule as the live pinch detector
        return "pinch"
    dx = path[-1, 0] - path[0, 0]
    dy = path[-1, 1] - path[0, 1]
    if abs(dx) > 1.2 and abs(dx) > 2 * abs(dy):            # moved more than ~1 hand width sideways
        return "swipe_right" if dx > 0 else "swipe_left"
    # open palm: fingertips far from the wrist (fingers extended) and the hand held fairly still
    tips = np.linalg.norm(world[:, [8, 12, 16, 20]] - world[:, :1], axis=-1).mean(axis=1) / _hand_size_world(world)
    still = np.abs(np.diff(path, axis=0)).sum() < 1.0
    if np.mean(tips > 1.85) > 0.8 and still:
        return "open_palm"
    return "idle"
