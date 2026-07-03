"""
biomechanics.py

Shared biomechanical feature computation used by:
  - offline pose extraction (src/pose_extraction.py)
  - the live OpenCV/MediaPipe app (src/live_app.py)

Keeping this logic in one module guarantees the live app computes features
*exactly* the same way the training pipeline did -- a common source of silent
train/serve skew in projects like this.

Feature set (grounded in the dissertation's literature review -- Zhao, 2024;
Hanen et al., 2025; Lee et al., 2024):
  - knee_angle_L / knee_angle_R : hip-knee-ankle angle (sagittal knee flexion)
  - hip_hinge_angle             : shoulder-hip-knee angle (hip hinge depth)
  - trunk_lean_angle            : shoulder-hip-ankle angle (proxy for lumbar /
                                   spinal alignment -- a true lumbar lordosis
                                   measurement needs marker-based mocap, which
                                   is exactly the gap this project addresses)
  - knee_symmetry               : abs(knee_angle_L - knee_angle_R)
  - knee_angular_velocity       : frame-to-frame change in mean knee angle
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import numpy as np

# MediaPipe Pose landmark indices (33-point model). Defined here so this
# module has no hard dependency on the mediapipe package itself -- useful
# for unit testing and for the synthetic data generator.
LM = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def angle_3pt(a, b, c) -> float:
    """Angle at vertex b formed by points a-b-c, in degrees. a/b/c are (x, y)
    or (x, y, z) iterables."""
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def landmarks_to_xy(landmarks) -> dict:
    """Convert a MediaPipe pose_landmarks.landmark list into a dict of
    {name: (x, y)} for the points this module needs. `landmarks` must support
    indexing with .x/.y attributes (mediapipe NormalizedLandmark) OR be a
    plain dict already shaped {name: (x, y)} (used by the synthetic
    generator / unit tests)."""
    if isinstance(landmarks, dict):
        return landmarks
    return {name: (landmarks[idx].x, landmarks[idx].y) for name, idx in LM.items()}


def compute_frame_features(landmarks) -> dict:
    """Compute the per-frame biomechanical feature vector from one frame's
    landmarks. Returns a flat dict -- this is the single source of truth for
    'what a frame's features are' used everywhere downstream."""
    p = landmarks_to_xy(landmarks)

    knee_L = angle_3pt(p["left_hip"], p["left_knee"], p["left_ankle"])
    knee_R = angle_3pt(p["right_hip"], p["right_knee"], p["right_ankle"])

    # Use the side-average for hip hinge / trunk lean to stay robust to
    # partial occlusion of one side.
    mid_shoulder = np.mean([p["left_shoulder"], p["right_shoulder"]], axis=0)
    mid_hip = np.mean([p["left_hip"], p["right_hip"]], axis=0)
    mid_knee = np.mean([p["left_knee"], p["right_knee"]], axis=0)
    mid_ankle = np.mean([p["left_ankle"], p["right_ankle"]], axis=0)

    hip_hinge = angle_3pt(mid_shoulder, mid_hip, mid_knee)
    trunk_lean = angle_3pt(mid_shoulder, mid_hip, mid_ankle)
    symmetry = abs(knee_L - knee_R)
    knee_mean = (knee_L + knee_R) / 2.0

    return {
        "knee_angle_L": knee_L,
        "knee_angle_R": knee_R,
        "knee_angle_mean": knee_mean,
        "hip_hinge_angle": hip_hinge,
        "trunk_lean_angle": trunk_lean,
        "knee_symmetry": symmetry,
    }


@dataclass
class RepSegmenter:
    """Frame-by-frame state machine that detects squat/deadlift repetition
    boundaries from the live knee-angle signal.

    Thresholds are deliberately conservative defaults inspired by the
    biomechanical ranges discussed in Hanen et al. (2025) and Lee et al.
    (2024) -- standing knee angle is close to 170-180 deg, a full-depth rep
    bottom is typically <100 deg. Tune per-subject if needed (camera angle
    and individual anthropometry shift these in practice).
    """
    descent_threshold: float = 160.0
    bottom_threshold: float = 110.0
    ascent_threshold: float = 150.0
    state: str = "top"
    frame_buffer: list = field(default_factory=list)

    def update(self, knee_angle_mean: float, frame_features: dict) -> dict | None:
        """Feed one frame's mean knee angle + full feature dict. Returns the
        completed rep's buffered frame-feature list when a rep finishes,
        otherwise None."""
        self.frame_buffer.append(frame_features)

        if self.state == "top" and knee_angle_mean < self.descent_threshold:
            self.state = "descending"

        elif self.state == "descending" and knee_angle_mean < self.bottom_threshold:
            self.state = "bottom"

        elif self.state == "bottom" and knee_angle_mean > self.ascent_threshold:
            self.state = "ascending"

        elif self.state == "ascending" and knee_angle_mean > self.descent_threshold:
            self.state = "top"
            completed_rep = self.frame_buffer
            self.frame_buffer = []
            return completed_rep

        return None


def aggregate_rep_features(rep_frames: list[dict]) -> dict:
    """Collapse a list of per-frame feature dicts (one completed rep) into
    the per-rep summary-statistic feature vector the classifiers are trained
    on: mean / std / range of each signal across the rep. This mirrors the
    temporal-statistics approach validated by Burns et al. (2023) and
    Reyes Leiva et al. (2025) in the literature review."""
    if not rep_frames:
        raise ValueError("Cannot aggregate an empty rep")

    keys = rep_frames[0].keys()
    out = {}
    for k in keys:
        series = np.array([f[k] for f in rep_frames], dtype=float)
        out[f"{k}_mean"] = float(series.mean())
        out[f"{k}_std"] = float(series.std())
        out[f"{k}_range"] = float(series.max() - series.min())
    out["n_frames"] = len(rep_frames)
    return out


FEATURE_COLUMNS = [
    f"{base}_{stat}"
    for base in ["knee_angle_L", "knee_angle_R", "knee_angle_mean",
                 "hip_hinge_angle", "trunk_lean_angle", "knee_symmetry"]
    for stat in ["mean", "std", "range"]
] + ["n_frames"]
