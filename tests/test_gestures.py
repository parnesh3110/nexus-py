"""Gesture features, the rule baseline, and (if PyTorch is installed) that the model learns."""
import numpy as np
import pytest

from nexus.gesture_features import CLASSES, FEATURES, WINDOW, clip_features, fill_gaps, resample, rule_baseline


def synthetic_clip(kind: str, rng, T: int = WINDOW):
    """A fake hand doing a gesture: wrist, knuckles, fingertips moving plausibly."""
    img = np.zeros((T, 21, 3), np.float32)
    world = np.zeros((T, 21, 3), np.float32)
    t = np.linspace(0, 1, T)
    cx = 0.5 + rng.normal(0, 0.02)
    dx = {"swipe_left": 0.25, "swipe_right": -0.25}.get(kind, 0.0)  # image x is un-mirrored
    for f in range(T):
        x = cx + dx * t[f] + rng.normal(0, 0.003)
        y = 0.5 + rng.normal(0, 0.003)
        world[f, 0] = [0, 0.09, 0]                       # wrist
        for j, (kx, tip_x) in zip([5, 9, 13, 17], [(-0.02, -0.025), (0, 0), (0.02, 0.02), (0.035, 0.04)]):
            world[f, j] = [kx, 0, 0]                     # knuckles, hand size ~0.09 m
        extend = 1.3 if kind == "open_palm" else 0.8     # fingertips distance from the knuckles
        for tip, kx in zip([8, 12, 16, 20], [-0.025, 0, 0.02, 0.04]):
            world[f, tip] = [kx * (1.6 if kind == "open_palm" else 1), -0.08 * extend, 0]
        gap = 0.07
        if kind == "pinch" and 0.35 < t[f] < 0.65:
            gap = 0.004
        world[f, 4] = world[f, 8] + [gap, 0, 0]          # thumb tip
        world[f] += rng.normal(0, 0.001, (21, 3))
        img[f, :, :2] = world[f, :, :2] * 1.5 + [x, y]
    return img, world


def test_feature_shape_and_translation_invariance():
    rng = np.random.default_rng(0)
    img, world = synthetic_clip("idle", rng)
    f = clip_features(img, world)
    assert f.shape == (WINDOW, FEATURES)
    shifted = img.copy()
    shifted[:, :, 0] += 0.2  # same hand, elsewhere in the frame
    assert np.allclose(f, clip_features(shifted, world), atol=1e-4)


def test_resample_and_gaps():
    seq = np.arange(36, dtype=np.float32)[:, None]
    r = resample(seq, 30)
    assert r.shape == (30, 1) and r[0, 0] == 0 and r[-1, 0] == 35
    img, world = np.ones((10, 21, 3)), np.ones((10, 21, 3))
    valid = np.array([True] * 8 + [False] * 2)
    assert fill_gaps(img, world, valid) is not None
    assert fill_gaps(img, world, np.array([True] * 5 + [False] * 5)) is None  # half missing: dropped


def test_rule_baseline_on_clear_cases():
    rng = np.random.default_rng(1)
    assert rule_baseline(*synthetic_clip("pinch", rng)) == "pinch"
    assert rule_baseline(*synthetic_clip("swipe_right", rng)) == "swipe_right"
    assert rule_baseline(*synthetic_clip("swipe_left", rng)) == "swipe_left"
    assert rule_baseline(*synthetic_clip("idle", rng)) == "idle"


def test_model_learns_synthetic_gestures():
    torch = pytest.importorskip("torch")
    from nexus.gesture_model import GestureNet
    rng = np.random.default_rng(2)
    X, y = [], []
    for k, c in enumerate(CLASSES):
        for _ in range(40):
            X.append(clip_features(*synthetic_clip(c, rng)))
            y.append(k)
    X, y = np.stack(X), np.array(y)
    mean, std = X.reshape(-1, FEATURES).mean(0), X.reshape(-1, FEATURES).std(0) + 1e-6
    Xn = torch.from_numpy(((X - mean) / std).astype(np.float32))
    yt = torch.from_numpy(y)
    torch.manual_seed(0)
    model = GestureNet()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(60):
        perm = torch.randperm(len(y))
        for b in range(0, len(y), 32):
            idx = perm[b:b + 32]
            loss = torch.nn.functional.cross_entropy(model(Xn[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xn).argmax(1) == yt).float().mean().item()
    assert acc > 0.95
