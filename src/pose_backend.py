"""
pose_backend.py

Thin wrapper around MediaPipe's current Tasks API (PoseLandmarker), giving
the rest of this project a simple, uniform interface so biomechanics.py and
every script that uses it (extract_real_dataset.py, pose_extraction.py,
live_app.py, the Django app) don't need to touch the Tasks API directly.

WHY THIS EXISTS: this project was originally built against mediapipe's
legacy `mp.solutions.pose` API. As of PyPI, mediapipe versions 0.10.22
through 0.10.29 have been removed from the index entirely (confirmed July
2026 -- `pip install mediapipe==0.10.21` fails with "no matching
distribution" even though it installs fine in some cached/mirrored
environments). Only 0.10.30+ is installable now, and those versions dropped
`mp.solutions` completely (`hasattr(mediapipe, "solutions")` is False) in
favour of the Tasks API used here.

WHAT CHANGES FOR YOU: nothing in biomechanics.py needs to change --
`compute_frame_features()` just needs a list of 33 landmarks with
.x/.y/.z attributes, indexed the same way as before. This module's
`PoseDetector.detect()` returns exactly that.

ONE-TIME MODEL DOWNLOAD: the Tasks API needs a small model file
(pose_landmarker_full.task, ~30MB) that isn't bundled in the pip package.
`ensure_model()` downloads it automatically to
models/pose_landmarker_full.task the first time you run anything that uses
this module. This needs a normal internet connection (this works fine on a
regular machine; it was blocked in the sandbox this project was partly
developed in, which is why the model isn't already sitting in models/).
If the automatic download fails (e.g. a restrictive network), download it
yourself from the URL below and save it to that exact path.
"""

import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_full.task"

# Simplified skeleton connections for the live overlay -- just the joints
# biomechanics.py actually uses (shoulders, hips, knees, ankles), which is
# also more directly relevant to what's being measured than the full
# 33-point body would be. Indices match MediaPipe's standard pose landmark
# numbering (unchanged between the old and new API).
POSE_CONNECTIONS = [
    (11, 12),   # shoulder - shoulder
    (11, 23), (12, 24),   # shoulder - hip
    (23, 24),   # hip - hip
    (23, 25), (25, 27),   # left hip - knee - ankle
    (24, 26), (26, 28),   # right hip - knee - ankle
]


def ensure_model() -> Path:
    """Download the pose landmarker model file if it isn't already present.
    Safe to call every time -- it's a no-op once the file exists."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Downloading pose landmarker model to {MODEL_PATH} (one-time, ~30MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Done.")
        except Exception as exc:
            raise RuntimeError(
                f"Could not download the pose landmarker model automatically ({exc}). "
                f"Download it yourself from {MODEL_URL} and save it to {MODEL_PATH}"
            ) from exc
    return MODEL_PATH


class PoseDetector:
    """Wraps PoseLandmarker in VIDEO running mode, which works fine for both
    offline video files and a live webcam loop -- it just needs frame
    timestamps to increase monotonically, which an internal counter handles
    automatically if you don't pass one yourself. Use as a context manager.

    Usage:
        with PoseDetector() as pose:
            landmarks = pose.detect(rgb_frame)   # rgb_frame: HxWx3 uint8 RGB
            if landmarks is not None:
                feats = compute_frame_features(landmarks)
    """

    def __init__(self, num_poses: int = 1):
        model_path = ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._ts = 0

    def detect(self, rgb_frame, timestamp_ms: int | None = None):
        """rgb_frame: HxWx3 uint8 RGB numpy array (NOT BGR -- convert with
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) first, same as the old API).
        Returns a list of 33 landmarks (.x/.y/.z/.visibility, normalised
        0-1 coords) for the first detected person, or None if nobody was
        detected in this frame."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if timestamp_ms is None:
            self._ts += 1
            timestamp_ms = self._ts
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def draw_landmarks(frame, landmarks, color=(200, 200, 200), thickness=2, radius=4):
    """Draws the simplified skeleton (see POSE_CONNECTIONS) directly with
    OpenCV -- replaces mp.solutions.drawing_utils, which no longer exists
    in mediapipe>=0.10.30 (the whole `mp.solutions` namespace was removed,
    not just the Pose model). Mutates and returns `frame`."""
    import cv2
    h, w = frame.shape[:2]

    def px(lm):
        return int(lm.x * w), int(lm.y * h)

    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, px(landmarks[a]), px(landmarks[b]), color, thickness, cv2.LINE_AA)
    for idx in {i for pair in POSE_CONNECTIONS for i in pair}:
        cv2.circle(frame, px(landmarks[idx]), radius, color, -1, cv2.LINE_AA)
    return frame
