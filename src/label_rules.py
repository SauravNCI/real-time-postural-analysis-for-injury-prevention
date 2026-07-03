"""
label_rules.py

THE CENTRAL LIMITATION OF THIS PROJECT, MADE EXPLICIT.

The real dataset (data/<exercise>/*.mp4) -- like the source Kaggle Workout/
Exercises Video dataset it comes from -- labels videos by EXERCISE TYPE only
("this is a squat"). It carries no safe/unsafe posture ground truth, because
that label doesn't exist without either (a) an expert (physiotherapist /
strength coach) reviewing every rep, or (b) a defensible automatic proxy.

This module implements option (b): a transparent, tunable, DOCUMENTED
heuristic that flags a rep as "unsafe" when it is a biomechanical outlier
*relative to other reps of the same exercise* in the dataset, on the
specific signals the dissertation's literature review identifies as
injury-relevant:

  - trunk_lean_angle_mean : excessive forward/rounded trunk lean is linked
    to spinal flexion injury risk (Hanen et al. 2025; Lee et al. 2024)
  - knee_symmetry_mean    : large left/right knee angle imbalance is linked
    to valgus/varus loading asymmetry
  - trunk_lean_angle_std  : high within-rep trunk variability suggests a
    loss of postural control / bracing through the rep

WHY RELATIVE (per-exercise robust z-score) RATHER THAN FIXED ANGLE CUTOFFS:
the dissertation's cited literature reports injury-relevant RANGES, not a
single universal cutoff, and cutoffs are sensitive to camera angle/distance,
which varies a lot across a YouTube-sourced dataset like this one. Flagging
statistical outliers within each exercise class is a defensible, referenced
approach (e.g. anomaly-detection framings used in movement-screening
literature) and adapts automatically to this dataset's actual camera
conditions rather than guessing absolute degree thresholds that may not
transfer.

THIS IS A PLACEHOLDER, NOT A SUBSTITUTE FOR EXPERT LABELS. State this
explicitly in the dissertation's limitations section:
  "Ground-truth safe/unsafe labels were derived via an automated statistical
  outlier rule (see label_rules.py) in the absence of expert physiotherapist
  annotation, which was outside the scope/timeline of this project. Model
  performance figures should be read as evidence the *pipeline* works
  correctly, not as a validated clinical safety classifier."

If/when you get even a small expert-labelled subset (see
data/raw/LABELLING_PROTOCOL.md), pass it via --labels_csv to
pose_extraction.py instead -- real labels always take priority over this
heuristic when both are available (see merge logic in pose_extraction.py).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# Tunable thresholds -- in units of robust z-score (based on median +
# MAD-scaled deviation, so a single wild outlier can't distort the scale
# the way a mean/std-based z-score would).
DEFAULT_THRESHOLDS = {
    "trunk_lean_angle_mean": -1.25,   # flag if trunk lean is this many robust-SDs BELOW the
                                       # exercise median (smaller angle = more forward lean)
    "knee_symmetry_mean": 1.25,       # flag if knee L/R asymmetry is this many robust-SDs ABOVE median
    "trunk_lean_angle_std": 1.25,     # flag if within-rep trunk wobble is this many robust-SDs ABOVE median
}
# A rep is labelled unsafe (1) if at least this many of the criteria above fire.
MIN_VIOLATIONS_FOR_UNSAFE = 1


def _robust_z(series: pd.Series) -> pd.Series:
    """Median/MAD-based robust z-score. 1.4826 makes MAD consistent with
    the standard deviation for normally distributed data."""
    median = series.median()
    mad = (series - median).abs().median()
    scale = mad * 1.4826
    if scale < 1e-6:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - median) / scale


def auto_label(df: pd.DataFrame, thresholds: dict | None = None,
                min_violations: int = MIN_VIOLATIONS_FOR_UNSAFE) -> pd.DataFrame:
    """Add a 'label' column (0=safe, 1=unsafe) to a per-rep feature
    DataFrame, computed independently within each 'exercise' group so
    thresholds adapt to that exercise's own angle ranges/camera conditions.

    Also adds 'label_source' = 'auto_heuristic' and 'n_violations' so you
    can inspect/audit exactly why each rep was flagged -- include this
    audit trail in the dissertation appendix for transparency.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    df = df.copy()
    df["n_violations"] = 0
    df["label_source"] = "auto_heuristic"

    for exercise, group in df.groupby("exercise"):
        idx = group.index
        violations = pd.Series(0, index=idx)

        z_trunk_mean = _robust_z(group["trunk_lean_angle_mean"])
        violations += (z_trunk_mean <= thresholds["trunk_lean_angle_mean"]).astype(int)

        z_symmetry = _robust_z(group["knee_symmetry_mean"])
        violations += (z_symmetry >= thresholds["knee_symmetry_mean"]).astype(int)

        z_trunk_std = _robust_z(group["trunk_lean_angle_std"])
        violations += (z_trunk_std >= thresholds["trunk_lean_angle_std"]).astype(int)

        df.loc[idx, "n_violations"] = violations

    df["label"] = (df["n_violations"] >= min_violations).astype(int)
    return df


def merge_with_manual_labels(auto_df: pd.DataFrame, manual_labels_csv: str) -> pd.DataFrame:
    """If a manually/expert-labelled CSV exists (columns: source_video,
    label -- see LABELLING_PROTOCOL.md), overwrite the heuristic label for
    any video it covers. Real labels always win over the heuristic."""
    manual = pd.read_csv(manual_labels_csv)
    manual = manual.set_index("source_video")["label"].to_dict()

    df = auto_df.copy()
    mask = df["source_video"].isin(manual.keys())
    df.loc[mask, "label"] = df.loc[mask, "source_video"].map(manual)
    df.loc[mask, "label_source"] = "expert_manual"
    return df


if __name__ == "__main__":
    # Quick smoke test on a tiny synthetic frame to sanity-check the logic
    import sys
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "exercise": ["squat"] * n,
        "trunk_lean_angle_mean": rng.normal(155, 8, n),
        "trunk_lean_angle_std": rng.normal(7, 2, n),
        "knee_symmetry_mean": np.abs(rng.normal(5, 2.5, n)),
    })
    labelled = auto_label(df)
    print(labelled["label"].value_counts())
    print("Smoke test OK" if labelled["label"].nunique() > 1 else "WARNING: no variation in labels",
          file=sys.stderr)
