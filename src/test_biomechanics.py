"""
test_biomechanics.py

Sanity tests for the shared biomechanics module. Run with:
    python src/test_biomechanics.py

These don't require mediapipe/opencv -- they exercise the module using
plain (x, y) dict landmarks, which is exactly the fallback path
landmarks_to_xy() supports for testing.
"""

import numpy as np
from biomechanics import angle_3pt, compute_frame_features, RepSegmenter, aggregate_rep_features, FEATURE_COLUMNS


def test_angle_3pt_right_angle():
    # a=(0,1), b=(0,0), c=(1,0) -> 90 degree angle at vertex b
    a, b, c = (0, 1), (0, 0), (1, 0)
    result = angle_3pt(a, b, c)
    assert abs(result - 90.0) < 1e-6, f"expected 90deg, got {result}"


def test_angle_3pt_straight_line():
    # a=(-1,0), b=(0,0), c=(1,0) -> 180 degree (straight line)
    result = angle_3pt((-1, 0), (0, 0), (1, 0))
    assert abs(result - 180.0) < 0.05, f"expected ~180deg, got {result}"


def _standing_landmarks():
    """A roughly upright standing pose (knees ~ straight)."""
    return {
        "left_shoulder": (0.4, 0.2), "right_shoulder": (0.6, 0.2),
        "left_hip": (0.4, 0.5), "right_hip": (0.6, 0.5),
        "left_knee": (0.4, 0.75), "right_knee": (0.6, 0.75),
        "left_ankle": (0.4, 1.0), "right_ankle": (0.6, 1.0),
    }


def _squat_bottom_landmarks():
    """A deep squat: knees bent forward, hips dropped, trunk leaning forward."""
    return {
        "left_shoulder": (0.35, 0.35), "right_shoulder": (0.65, 0.35),
        "left_hip": (0.4, 0.65), "right_hip": (0.6, 0.65),
        "left_knee": (0.3, 0.8), "right_knee": (0.7, 0.8),
        "left_ankle": (0.4, 1.0), "right_ankle": (0.6, 1.0),
    }


def test_compute_frame_features_standing_has_larger_knee_angle_than_squat():
    standing = compute_frame_features(_standing_landmarks())
    squat = compute_frame_features(_squat_bottom_landmarks())
    assert standing["knee_angle_mean"] > squat["knee_angle_mean"], (
        "standing knee angle should be closer to 180deg (straight leg) than a squat bottom"
    )


def test_rep_segmenter_detects_one_full_rep():
    segmenter = RepSegmenter(descent_threshold=160, bottom_threshold=110, ascent_threshold=150)

    # simulate a knee-angle time series: standing -> descend -> bottom -> ascend -> standing
    angles = list(np.linspace(175, 175, 5)) + \
              list(np.linspace(175, 95, 10)) + \
              list(np.linspace(95, 95, 5)) + \
              list(np.linspace(95, 175, 10)) + \
              list(np.linspace(175, 175, 5))

    completed_reps = []
    for a in angles:
        fake_frame_features = {"knee_angle_mean": a, "dummy": a}
        rep = segmenter.update(a, fake_frame_features)
        if rep is not None:
            completed_reps.append(rep)

    assert len(completed_reps) == 1, f"expected exactly 1 completed rep, got {len(completed_reps)}"
    assert len(completed_reps[0]) > 0


def test_aggregate_rep_features_shape():
    frames = [compute_frame_features(_squat_bottom_landmarks()) for _ in range(10)]
    agg = aggregate_rep_features(frames)
    for col in FEATURE_COLUMNS:
        assert col in agg, f"missing expected column {col} in aggregated features"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")
