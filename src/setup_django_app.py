"""
setup_django_app.py

Scaffolds a Django project (posture_django/, a sibling of src/) that streams
your laptop webcam over HTTP with a live MediaPipe pose overlay and a
per-rep safe/unsafe verdict from the trained Random Forest model -- the
same real-time architecture as live_app.py, but served as a web page
instead of an OpenCV desktop window (multipart/x-mixed-replace MJPEG
streaming, no extra JS frameworks needed).

Run this ONCE locally (needs django in requirements.txt -- already added):
    cd src
    python setup_django_app.py

Then:
    cd ../posture_django
    python manage.py runserver
    # open http://127.0.0.1:8000/ in a browser

Design notes:
  - Reuses biomechanics.py from src/ directly (sys.path insert in views.py)
    -- same single source of truth as pose_extraction.py and live_app.py,
    so there is no train/serve skew between how features were computed for
    training vs. how they're computed live.
  - Uses random_forest.pkl by default for the same reason live_app.py does:
    sub-millisecond inference, no GPU, no sequence buffering -- the right
    choice for a real-time path. Change MODEL_PATH in views.py to try a
    different model.
  - This creates the Django project structure on YOUR filesystem when you
    run it -- nothing here needs internet access or admin rights beyond
    `pip install django` (already in requirements.txt).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent / "posture_django"

FILES = {}

# ---------------------------------------------------------------- manage.py
FILES["manage.py"] = '''#!/usr/bin/env python
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "posture_django.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Run: pip install django"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
'''

# --------------------------------------------------- posture_django/settings.py
FILES["posture_django/settings.py"] = '''from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Dev-only secret key -- do not deploy this project publicly as-is.
SECRET_KEY = "dev-only-not-for-production-change-before-any-real-deployment"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "livefeed",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "posture_django.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    },
]

WSGI_APPLICATION = "posture_django.wsgi.application"

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
'''

# ------------------------------------------------------- posture_django/urls.py
FILES["posture_django/urls.py"] = '''from django.urls import path, include

urlpatterns = [
    path("", include("livefeed.urls")),
]
'''

# ------------------------------------------------------- posture_django/wsgi.py
FILES["posture_django/wsgi.py"] = '''import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "posture_django.settings")

application = get_wsgi_application()
'''

FILES["posture_django/__init__.py"] = ""

# ------------------------------------------------------------- livefeed/__init__.py
FILES["livefeed/__init__.py"] = ""

# ------------------------------------------------------------------ livefeed/urls.py
FILES["livefeed/urls.py"] = '''from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("video_feed/", views.video_feed, name="video_feed"),
]
'''

# ----------------------------------------------------------------- livefeed/views.py
FILES["livefeed/views.py"] = '''"""
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
    # On Windows, OpenCV's default backend (MSMF) frequently fails to open a
    # working webcam even when the camera itself is fine -- CAP_DSHOW is far
    # more reliable there. Falls back to the default backend on other OSes.
    if sys.platform == "win32":
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(
            f"ERROR: could not open camera index {CAMERA_INDEX}. Check "
            "Windows Settings > Privacy & security > Camera (both "
            "'Let apps access your camera' and the desktop-apps toggle must "
            "be ON), and make sure no other app/process is holding the "
            "camera -- e.g. stop live_app.py first if you tried that too.",
            file=sys.stderr,
        )
        return

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
            yield (b"--frame\\r\\n"
                   b"Content-Type: image/jpeg\\r\\n\\r\\n" + frame_bytes + b"\\r\\n")

    cap.release()


def index(request):
    return render(request, "livefeed/index.html")


def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                                  content_type="multipart/x-mixed-replace; boundary=frame")
'''

# ---------------------------------------------------- livefeed/templates/livefeed/index.html
FILES["livefeed/templates/livefeed/index.html"] = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Real-Time Posture Analysis</title>
  <style>
    body { background: #111; color: #eee; font-family: system-ui, sans-serif; text-align: center; margin: 0; padding: 24px; }
    h1 { font-weight: 600; margin-bottom: 4px; }
    p.sub { color: #999; margin-top: 0; }
    img { max-width: 90%; border: 2px solid #333; border-radius: 10px; margin-top: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.4); }
    .legend { margin-top: 16px; font-size: 14px; color: #aaa; }
    .legend span { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }
  </style>
</head>
<body>
  <h1>Real-Time Posture Analysis</h1>
  <p class="sub">MediaPipe pose landmarks + Random Forest safe/unsafe classification, updated every completed rep.</p>
  <img src="{% url 'video_feed' %}" alt="Live posture analysis feed">
  <div class="legend">
    <span style="background:#2ecc71;"></span> Safe posture &nbsp;&nbsp;
    <span style="background:#e74c3c;"></span> Unsafe posture
  </div>
</body>
</html>
'''

# ------------------------------------------------------------------------- README
FILES["README.md"] = '''# posture_django

Real-time posture analysis served over HTTP, generated by
`../src/setup_django_app.py`. Streams your webcam with a live MediaPipe
pose overlay and a per-rep safe/unsafe verdict from the trained Random
Forest model (`../models/random_forest.pkl`).

## Run it

```
pip install -r ../requirements.txt   # if you haven't already (includes django)
python manage.py runserver
```

Then open http://127.0.0.1:8000/ in a browser on the SAME machine as the
webcam (the dev server binds to localhost; streaming to another device on
your network needs `runserver 0.0.0.0:8000` and a firewall rule -- fine for
a local dissertation demo, not intended as a public deployment).

## Notes

- Uses `random_forest.pkl` for the same reason `live_app.py` does:
  sub-millisecond inference, no GPU, no sequence buffering. Edit
  `livefeed/views.py` (`MODEL_PATH`) to try a different model.
- Reuses `../src/biomechanics.py` directly (see the `sys.path.insert` at
  the top of `livefeed/views.py`) so live feature computation is
  guaranteed identical to what the offline training pipeline used --
  no train/serve skew.
- This is a development server (`runserver`), fine for local
  demonstration/dissertation use. Do not deploy this as-is to a public
  network -- `SECRET_KEY`/`DEBUG=True`/`ALLOWED_HOSTS=["*"]` in
  `posture_django/settings.py` are dev-only settings.
'''


def main():
    for rel_path, content in FILES.items():
        full_path = PROJECT_ROOT / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        print(f"wrote {full_path.relative_to(PROJECT_ROOT.parent)}")

    print(f"\nDjango project created at: {PROJECT_ROOT}")
    print("Next steps:")
    print(f"  cd {PROJECT_ROOT.name}")
    print("  python manage.py runserver")
    print("  open http://127.0.0.1:8000/ in a browser")


if __name__ == "__main__":
    main()
