"""
live_app.py

The end-to-end goal: real-time posture safety feedback from a webcam feed.

Run this LOCALLY on a machine with a webcam. Requires: opencv-python,
mediapipe, joblib, scikit-learn/xgboost (whichever produced the model you
load), numpy.

Usage:
    python src/live_app.py --model models/random_forest.pkl --scaler models/scaler.pkl

Controls:
    q  - quit
    r  - reset rep counter / segmenter state

Design notes:
  - Uses the SAME biomechanics.py functions as pose_extraction.py, so the
    live app computes features identically to how the training data was
    built -- no train/serve skew.
  - Uses RandomForest/XGBoost by default (NOT the CNN-BiLSTM) for the live
    path: tree models predict in <1ms on CPU with no sequence-buffering
    latency, matching the real-time requirement discussed in the project.
    If you want to demo the CNN-BiLSTM live instead, pass --model_type keras
    and point --model at the .keras file; it will buffer each rep's frames
    and reconstruct the same fixed-length sequence shape used in training.
  - Pose detection goes through pose_backend.py (MediaPipe's current Tasks
    API), not the older mp.solutions.pose API -- see that module's
    docstring for why. The first run downloads a ~30MB model file
    automatically.
"""

import argparse
import sys
import time
from collections import deque

import cv2
import joblib
import numpy as np

from biomechanics import (compute_frame_features, RepSegmenter,
                           aggregate_rep_features, FEATURE_COLUMNS)
from pose_backend import PoseDetector, draw_landmarks


def load_sklearn_model(model_path, scaler_path=None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    return model, scaler


def classify_rep(model, scaler, rep_frames):
    feats = aggregate_rep_features(rep_frames)
    x = np.array([[feats[c] for c in FEATURE_COLUMNS]])
    if scaler is not None:
        x = scaler.transform(x)
    pred = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0][1] if hasattr(model, "predict_proba") else float(pred)
    return pred, proba, feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/random_forest.pkl")
    parser.add_argument("--scaler", default=None,
                         help="Pass models/scaler.pkl only if using logistic_regression.pkl "
                              "(tree models were trained on unscaled features -- see train_models.py)")
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()

    model, scaler = load_sklearn_model(args.model, args.scaler)

    # On Windows, OpenCV's default backend (MSMF) frequently fails to open a
    # working webcam even when the camera itself is fine -- CAP_DSHOW is far
    # more reliable there. Falls back to the default backend on other OSes
    # or if DSHOW itself can't open the device.
    if sys.platform == "win32":
        cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(args.camera_index)
    else:
        cap = cv2.VideoCapture(args.camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(
            f"ERROR: could not open camera index {args.camera_index}.\n"
            "  - Check Windows Settings > Privacy & security > Camera: "
            "\"Let apps access your camera\" and \"Let desktop apps access "
            "your camera\" must both be ON.\n"
            "  - Make sure no other app (Zoom, Teams, the Django server, "
            "another Python process, etc.) is currently holding the camera.\n"
            "  - If you have more than one camera, try --camera_index 1 "
            "(or 2, ...).",
            file=sys.stderr,
        )
        sys.exit(1)

    segmenter = RepSegmenter()
    rep_count = 0
    last_verdict = None       # (label, proba) of most recently completed rep
    last_verdict_time = 0
    VERDICT_DISPLAY_SECONDS = 4.0

    fps_buffer = deque(maxlen=30)

    with PoseDetector() as pose:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed -- exiting.")
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = pose.detect(rgb)

            if landmarks is not None:
                draw_landmarks(frame, landmarks)

                frame_feats = compute_frame_features(landmarks)
                completed_rep = segmenter.update(frame_feats["knee_angle_mean"], frame_feats)

                if completed_rep is not None and len(completed_rep) >= 5:
                    rep_count += 1
                    pred, proba, _ = classify_rep(model, scaler, completed_rep)
                    last_verdict = (pred, proba)
                    last_verdict_time = time.time()
                    print(f"Rep {rep_count}: {'UNSAFE' if pred else 'SAFE'} "
                          f"(p_unsafe={proba:.2f}, frames={len(completed_rep)})")

                # live angle readout regardless of rep state
                cv2.putText(frame, f"knee: {frame_feats['knee_angle_mean']:.0f}deg  "
                                    f"hip: {frame_feats['hip_hinge_angle']:.0f}deg  "
                                    f"trunk: {frame_feats['trunk_lean_angle']:.0f}deg",
                            (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No person detected", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # rep counter + state
            cv2.putText(frame, f"Reps: {rep_count}  |  state: {segmenter.state}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # last verdict banner (fades after VERDICT_DISPLAY_SECONDS)
            if last_verdict is not None and (time.time() - last_verdict_time) < VERDICT_DISPLAY_SECONDS:
                pred, proba = last_verdict
                color = (0, 0, 255) if pred else (0, 200, 0)
                label = f"{'UNSAFE POSTURE' if pred else 'SAFE POSTURE'}  ({proba:.0%} confidence)"
                cv2.rectangle(frame, (0, 45), (frame.shape[1], 85), color, -1)
                cv2.putText(frame, label, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)

            fps_buffer.append(1.0 / max(time.time() - t0, 1e-6))
            cv2.putText(frame, f"FPS: {np.mean(fps_buffer):.0f}",
                        (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imshow("Real-time posture analysis", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                segmenter.state = "top"
                segmenter.frame_buffer = []
                rep_count = 0
                last_verdict = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
