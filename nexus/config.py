"""All tunable numbers in one place. Defaults come from tuning the Swift version on real logs."""
from dataclasses import dataclass, field

from . import paths

# How much hand movement covers the whole screen, as a fraction of the camera's width
SENSITIVITY = {"low": 0.22, "medium": 0.17, "high": 0.13, "max": 0.10}


@dataclass
class Config:
    # Camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    model_path: str = field(default_factory=lambda: str(paths.hand_model()))
    sensitivity: str = "medium"  # "high" pinned the pointer to screen edges 23-56% of the time (log 20260928-145541)

    # Smoothing: One Euro filter on the palm position, in camera space
    min_cutoff: float = 0.8
    beta: float = 40.0

    # Stabilization, in screen px
    band_px: float = 6.0          # wobble smaller than this is ignored
    lock_speed_px: float = 40.0   # pointer slower than this (px/s)...
    lock_after: float = 0.15      # ...for this long locks in place
    lock_radius_px: float = 14.0  # until the hand clearly moves away

    # Tracking loss
    coast: float = 0.3            # s: dropouts this short are ignored (MediaPipe drops 1-9 frames often)
    fade: float = 0.35            # s: then the pointer "fades" and any pinch is released
    reset_after_lost: float = 0.6  # s: filters restart after this
    acquire_frames: int = 2       # a new hand must show up in 2 frames in a row (kills one-frame ghosts)
    max_jump_per_sec: float = 2.5  # camera widths/s: faster = landmark glitch, not a hand

    # Pinch (thumb-index gap / hand size, 3D). Measured on your hand (log 20260929-201849): clicks that
    # counted 0.035-0.14, light touches that didn't 0.14-0.20. See PinchDetector for the relative rule.
    pinch_instant: float = 0.13   # this small = touching, right away
    pinch_enter: float = 0.22     # or under this AND clearly closing (to 65% of a moment ago)
    pinch_exit: float = 0.25
    pinch_exit_frames: int = 2
    pinch_arm_delay: float = 0.2  # s after a hand appears before it can click
    open_ratio: float = 0.28      # fingers counted as "open" above this (aim point for clicks)
    rewind_max_s: float = 0.4
    rewind_max_px: float = 45.0
    # A pinch becomes a drag only on a clear pull. Log 20260928-145541: the wobble while pinching moved
    # the pointer 175-180 px, past the old 150 px limit, so 2 of 5 clicks turned into drags.
    drag_start_px: float = 70.0
    drag_min_hold: float = 0.35
    drag_fast_px: float = 350.0

    # Idle: no hand for a while -> only look at every Nth frame
    idle_after: float = 1.5
    idle_stride: int = 3

    # Cursor animation thread (runs at display rate, between camera frames)
    render_hz: int = 120
    tau_slow: float = 0.070       # slow hand: heavy smoothing
    tau_fast: float = 0.014       # fast hand: light smoothing
    slow_speed: float = 120.0     # px/s
    fast_speed: float = 900.0
    predict_from: float = 250.0   # no prediction below this speed (it just amplifies tremor)
    predict_full: float = 700.0
    predict_lead: float = 0.06    # s
    predict_max_px: float = 60.0

    @property
    def box_width(self) -> float:
        return SENSITIVITY[self.sensitivity]
