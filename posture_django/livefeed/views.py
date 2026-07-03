"""
livefeed/views.py

Real-time posture analysis served over HTTP. Opens the local webcam,
runs MediaPipe Pose per frame (via the SAME biomechanics.py used for
training -- see sys.path insert below), segments reps, classifies each
completed rep with the trained Random Forest model, and streams the
annotated video as MJPEG (multipart/x-mixed-replace).

Pose detection goes through pose_backend.py (MediaPipe's current Tasks
API), not the older mp.solutions.pose API -- see that module's docstring
for why. The first run downloads a ~30MB model file automatically.
"""

import sys
import time
from pathlib import Path
from threading import Lock

import cv2
import joblib
import numpy as np
from django.http import StreamingHttpResponse
from django.shortcuts import render

# --- reuse the SAME feature-engineering code the offline pipeline trained on ---
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))
from biomechanics import (compute_frame_features, RepSegmenter,
                           aggregate_rep_features, FEATURE_COLUMNS)  # noqa: E402
from pose_backend import PoseDetector, draw_landmarks  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "random_forest.pkl"   # sub-ms inference, no GPU -- see setup_django_app.py docstring
SCALER_PATH = None  # only set this to MODELS_DIR / "scaler.pkl" if you switch MODEL_PATH to logistic_regression.pkl

_model = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH) if SCALER_PATH else None

CAMERA_INDEX = 0

_state_lock = Lock()
_state = {"rep_count": 0, "last_verdict": None, "last_confidence": 0.0}


def classify_rep(rep_frames):
    feats = aggregate_rep_features(rep_frames)
    x = np.array([[feats[c] for c in FEATURE_COLUMNS]])
    if _scaler is not None:
        x = _scaler.transform(x)
    pred = int(_model.predict(x)[0])
    proba = _model.predict_proba(x)[0][1] if hasattr(_model, "predict_proba") else float(pred)
    return pred, proba


def gen_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    segmenter = RepSegmenter()
    last_verdict = None
    last_verdict_time = 0
    VERDICT_DISPLAY_SECONDS = 4.0

    with PoseDetector() as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = pose.detect(rgb)

            if landmarks is not None:
                draw_landmarks(frame, landmarks)

                feats = compute_frame_features(landmarks)
                completed_rep = segmenter.update(feats["knee_angle_mean"], feats)

                if completed_rep is not None and len(completed_rep) >= 5:
                    pred, proba = classify_rep(completed_rep)
                    with _state_lock:
                        _state["rep_count"] += 1
                        _state["last_verdict"] = pred
                        _state["last_confidence"] = proba
                    last_verdict = (pred, proba)
                    last_verdict_time = time.time()

                cv2.putText(frame,
                            f"knee: {feats['knee_angle_mean']:.0f}deg  "
                            f"hip: {feats['hip_hinge_angle']:.0f}deg  "
                            f"trunk: {feats['trunk_lean_angle']:.0f}deg",
                            (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No person detected", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            with _state_lock:
                rep_count = _state["rep_count"]
            cv2.putText(frame, f"Reps: {rep_count}  |  state: {segmenter.state}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if last_verdict is not None and (time.time() - last_verdict_time) < VERDICT_DISPLAY_SECONDS:
                pred, proba = last_verdict
                color = (0, 0, 255) if pred else (0, 200, 0)
                label = f"{'UNSAFE POSTURE' if pred else 'SAFE POSTURE'}  ({proba:.0%} confidence)"
                cv2.rectangle(frame, (0, 45), (frame.shape[1], 85), color, -1)
                cv2.putText(frame, label, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

            ok2, buffer = cv2.imencode(".jpg", frame)
            if not ok2:
                continue
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


def index(request):
    return render(request, "livefeed/index.html")


def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                                  content_type="multipart/x-mixed-replace; boundary=frame")
