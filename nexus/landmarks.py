"""Hand landmark data and the few measurements the pointer needs. Pure numpy (no MediaPipe
import), so it can be unit-tested and reused by the ML training code.

MediaPipe indices: 0 wrist, 4 thumb tip, 8 index tip, 5/9/13/17 index/middle/ring/pinky knuckles.
Image coords are normalized 0..1 with origin top-left, NOT mirrored.
"""
from dataclasses import dataclass

import numpy as np

WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
KNUCKLES = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


@dataclass
class Hand:
    landmarks: np.ndarray          # (21, 3): x, y normalized to the image, z relative depth
    world: np.ndarray | None       # (21, 3): meters, centered on the hand (MediaPipe world landmarks)
    score: float                   # handedness confidence
    handedness: str = ""


def palm_point(hand: Hand) -> tuple[float, float]:
    """Center of the four knuckles, mirrored so moving your hand right moves the pointer right.
    Averaging 4 points cuts landmark noise, and knuckles barely move when you pinch."""
    k = hand.landmarks[KNUCKLES]
    return 1.0 - float(k[:, 0].mean()), float(k[:, 1].mean())


def pinch_ratio(hand: Hand, aspect: float = 4 / 3) -> float:
    """Thumb-to-index gap divided by hand size: the same at any distance from the camera.

    3D world landmarks when available. The gap is the smaller of thumb tip to index tip and
    (slightly penalized) thumb joint to index tip, because in many hand positions the finger PADS
    touch, not the very tips. Hand size is robust_size(): several bones, so one badly estimated
    bone (common when the hand is turned) doesn't swing the ratio. Log 20260929-201849 showed the
    same touch reading 0.035 in one position and 0.20 in another with a single-bone size."""
    if hand.world is not None:
        w = hand.world
        gap = min(np.linalg.norm(w[THUMB_TIP] - w[INDEX_TIP]), 1.15 * np.linalg.norm(w[3] - w[INDEX_TIP]))
        return float(gap / robust_size(w))
    l = hand.landmarks[:, :2].copy()
    l[:, 0] *= aspect  # x is normalized to width, y to height: put them on one scale
    gap = np.linalg.norm(l[THUMB_TIP] - l[INDEX_TIP])
    size = np.linalg.norm(l[WRIST] - l[MIDDLE_MCP])
    return float(gap / size) if size > 1e-6 else float("inf")


# Bones that don't change when you pinch or turn your hand, and their typical length relative to
# wrist -> middle knuckle. Each gives its own estimate of hand size; the median ignores a bad one.
_BONES = [((0, 9), 1.0), ((0, 5), 0.95), ((0, 17), 0.85), ((5, 17), 0.72), ((5, 6), 0.42), ((9, 10), 0.46)]


def robust_size(w: np.ndarray) -> float:
    """Hand size (wrist to middle knuckle equivalent), from six bones' median estimate."""
    est = [np.linalg.norm(w[a] - w[b]) / typical for (a, b), typical in _BONES]
    return max(float(np.median(est)), 1e-6)


def finger_curl(hand: Hand) -> float | None:
    """How extended the middle and ring fingers are (fingertip to wrist, in hand sizes). A fist
    drops this sharply; a pinch barely changes it. None without 3D landmarks."""
    if hand.world is None:
        return None
    w = hand.world
    return float((np.linalg.norm(w[12] - w[WRIST]) + np.linalg.norm(w[16] - w[WRIST])) / 2 / robust_size(w))


def is_bunched(hand: Hand) -> bool:
    """All five fingertips together with the fingers mostly straight (the Jarvis panel gesture).
    Not a pinch: in a relaxed pinch the other fingers are CURLED, so their tips sit near their own
    knuckles. The first version only checked tip-to-thumb distance, which matched curled pinches
    and blocked real clicks (log 20260927-213438: 4 clicks registered in 2+ minutes)."""
    if hand.world is None:
        return False
    return bunched_world(hand.world)


def bunched_world(w: np.ndarray, together: float = 0.4, straight: float = 0.5) -> bool:
    """is_bunched on raw world landmarks (21, 3). `together`: max fingertip-to-thumb distance, and
    `straight`: min fingertip-to-knuckle distance, both in hand sizes. The pinch blocker uses the
    strict defaults; the panel gesture passes looser ones (it was hard to hit)."""
    size = max(float(np.linalg.norm(w[MIDDLE_MCP] - w[WRIST])), 1e-6)
    tips_together = all(np.linalg.norm(w[tip] - w[THUMB_TIP]) / size < together for tip in (8, 12, 16, 20))
    fingers_straight = all(np.linalg.norm(w[tip] - w[mcp]) / size > straight
                           for tip, mcp in ((12, 9), (16, 13), (20, 17)))
    return tips_together and fingers_straight


def flatten(hand: Hand | None) -> list[float]:
    """126 numbers (image + world landmarks) for the dataset recorder; NaN when no hand."""
    if hand is None:
        return [float("nan")] * 126
    world = hand.world if hand.world is not None else np.full((21, 3), np.nan)
    return hand.landmarks.reshape(-1).tolist() + world.reshape(-1).tolist()
