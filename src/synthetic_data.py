"""
synthetic_data.py

Generates a synthetic per-rep biomechanical feature dataset so the rest of
the pipeline (training, evaluation, SHAP, McNemar's test) can be built and
validated end-to-end *before* real video annotation is finished.

This directly addresses the labelling gap identified earlier: the Kaggle
Workout/Exercises Video dataset labels videos by exercise type, not by
posture safety, so a real "unsafe posture" ground truth has to be created
(expert annotation or threshold-derived). This module stands in for that
real, annotated dataset with literature-grounded synthetic values so you can:

  1. Validate the whole training/evaluation/SHAP pipeline works correctly
  2. Have a fallback to report on if real annotation is incomplete by the
     submission deadline
  3. Sanity-check that the modelling choices behave as expected (recall
     optimisation, SHAP attributing risk to the features that should matter)

IMPORTANT: this is NOT a substitute for real data in the final dissertation.
Swap data/processed/rep_features.csv for the output of pose_extraction.py
run over real, expert/threshold-labelled video as soon as it's available --
see README.md, section "Replacing synthetic data with real video data".

Biomechanical ranges are loosely grounded in the dissertation's literature
review (Hanen et al., 2025; Lee et al., 2024; Zhao, 2024):
  - Safe squat/deadlift: knee angle bottoms out ~80-100 deg, hip hinge
    ~40-70 deg, trunk lean close to neutral, knee symmetry small (<8 deg).
  - Unsafe: shallower or excessively rounded trunk lean (spinal flexion
    risk), larger knee valgus/varus asymmetry, faster uncontrolled angular
    velocity (here approximated via larger std/range).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from biomechanics import FEATURE_COLUMNS

RNG = np.random.default_rng(42)


def _rep(label: int) -> dict:
    """Simulate one rep's aggregated mean/std/range feature vector.
    label: 0 = safe, 1 = unsafe.

    NOTE ON DIFFICULTY: class distributions deliberately overlap (rather than
    being cleanly separated) and ~8% of rows get their label flipped below,
    mimicking realistic annotation noise (inter-rater disagreement on
    borderline reps). A dataset where every model scores 100% tells you
    nothing about which model generalises best -- real annotated data will
    not be this clean, so the synthetic set shouldn't be either.
    """

    if label == 0:  # safe posture
        knee_mean = RNG.normal(98, 11)
        knee_std = RNG.normal(19, 4)
        knee_range = RNG.normal(68, 10)
        hip_hinge_mean = RNG.normal(54, 9)
        hip_hinge_std = RNG.normal(13, 3)
        hip_hinge_range = RNG.normal(44, 7)
        trunk_lean_mean = RNG.normal(160, 9)   # close to neutral/upright
        trunk_lean_std = RNG.normal(6, 2)
        trunk_lean_range = RNG.normal(17, 5)
        symmetry_mean = np.abs(RNG.normal(4.5, 2.5))
        symmetry_std = np.abs(RNG.normal(2, 1))
        symmetry_range = np.abs(RNG.normal(6, 2.5))
    else:  # unsafe posture
        knee_mean = RNG.normal(105, 12)         # shallower depth, overlapping safe range
        knee_std = RNG.normal(21, 4.5)
        knee_range = RNG.normal(62, 11)
        hip_hinge_mean = RNG.normal(50, 10)
        hip_hinge_std = RNG.normal(15, 3.5)
        hip_hinge_range = RNG.normal(42, 8)
        trunk_lean_mean = RNG.normal(150, 11)   # more forward/rounded lean, overlaps safe
        trunk_lean_std = RNG.normal(9, 3)
        trunk_lean_range = RNG.normal(24, 7)
        symmetry_mean = np.abs(RNG.normal(7.5, 3.5))   # bigger L/R imbalance, overlaps safe
        symmetry_std = np.abs(RNG.normal(3, 1.5))
        symmetry_range = np.abs(RNG.normal(9.5, 4))

    knee_L_mean = knee_mean + RNG.normal(0, symmetry_mean / 2)
    knee_R_mean = knee_mean - RNG.normal(0, symmetry_mean / 2)

    row = {
        "knee_angle_L_mean": knee_L_mean, "knee_angle_L_std": knee_std + RNG.normal(0, 1),
        "knee_angle_L_range": knee_range + RNG.normal(0, 3),
        "knee_angle_R_mean": knee_R_mean, "knee_angle_R_std": knee_std + RNG.normal(0, 1),
        "knee_angle_R_range": knee_range + RNG.normal(0, 3),
        "knee_angle_mean_mean": knee_mean, "knee_angle_mean_std": knee_std,
        "knee_angle_mean_range": knee_range,
        "hip_hinge_angle_mean": hip_hinge_mean, "hip_hinge_angle_std": hip_hinge_std,
        "hip_hinge_angle_range": hip_hinge_range,
        "trunk_lean_angle_mean": trunk_lean_mean, "trunk_lean_angle_std": trunk_lean_std,
        "trunk_lean_angle_range": trunk_lean_range,
        "knee_symmetry_mean": symmetry_mean, "knee_symmetry_std": symmetry_std,
        "knee_symmetry_range": symmetry_range,
        "n_frames": RNG.integers(18, 45),
    }
    return row


def generate_dataset(n_safe: int = 420, n_unsafe: int = 230,
                      exercise_mix: bool = True) -> pd.DataFrame:
    """Generate a synthetic, class-imbalanced (safe > unsafe, mirroring the
    real-world imbalance Leckey et al. 2024 flag as typical) per-rep dataset.

    n_safe / n_unsafe defaults give ~35% positive class, which is imbalanced
    but workable with stratified CV + class weighting, similar to what you
    should expect from real annotated data.
    """
    rows = []
    labels = []
    exercises = []

    for _ in range(n_safe):
        rows.append(_rep(0)); labels.append(0)
    for _ in range(n_unsafe):
        rows.append(_rep(1)); labels.append(1)

    if exercise_mix:
        exercises = RNG.choice(["squat", "deadlift"], size=len(rows), p=[0.55, 0.45])
    else:
        exercises = ["squat"] * len(rows)

    df = pd.DataFrame(rows)
    df = df[FEATURE_COLUMNS]  # enforce canonical column order
    df["exercise"] = exercises
    df["label"] = labels

    # simulate ~8% annotation noise (borderline reps where expert raters
    # would plausibly disagree) -- flips a random subset of labels
    noise_frac = 0.08
    n_noisy = int(len(df) * noise_frac)
    noisy_idx = RNG.choice(len(df), size=n_noisy, replace=False)
    df.loc[noisy_idx, "label"] = 1 - df.loc[noisy_idx, "label"]

    # shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "rep_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic rep rows to {out_path}")
    print(df["label"].value_counts(normalize=True).rename("class_balance"))
