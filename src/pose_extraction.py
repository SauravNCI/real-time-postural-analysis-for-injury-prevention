"""
pose_extraction.py

NOTE: if your videos are laid out as data/<exercise>/*.mp4 (one folder per
exercise, no separate labels.csv) -- which is how the real dataset in this
project is actually organised -- use extract_real_dataset.py instead. This
script is for the alternate layout below (flat videos/ folder + a
labels.csv you maintain yourself), useful once you have real expert labels.

Run this LOCALLY (needs a real MediaPipe + OpenCV environment) over your
real video dataset to produce the real data/processed/rep_features.csv that
should eventually replace the synthetic one used during development.

Usage:
    python src/pose_extraction.py \
        --videos_dir data/raw/videos \
        --labels_csv data/raw/labels.csv \
        --out data/processed/rep_features.csv

Expected input layout:
    data/raw/videos/*.mp4
    data/raw/labels.csv  with columns: filename, exercise, label
        label: 1 = unsafe posture, 0 = safe posture
        (see README.md "Labelling protocol" for how to produce this file --
        this is the gap flagged in the dissertation review: the raw Kaggle
        dataset only labels exercise *type*, not posture safety, so you must
        add this label column yourself via expert review or threshold rules.)

Output: one row per detected repetition, in the same schema as
synthetic_data.py's generate_dataset(), so train_models.py works unchanged
on either source.

NOTE ON MEDIAPIPE: this uses the current Tasks API via pose_backend.py, not
the older mp.solutions.pose API (see that module's docstring for why -- the
short version: mediapipe>=0.10.30 removed mp.solutions entirely, and the
in-between versions that still had it are no longer installable from PyPI).
The first run will download a ~30MB model file automatically.
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

from biomechanics import compute_frame_features, RepSegmenter, aggregate_rep_features, FEATURE_COLUMNS
from pose_backend import PoseDetector


def extract_reps_from_video(video_path: str, pose: PoseDetector) -> list[dict]:
    """Run MediaPipe Pose over every frame of a video, segment into reps via
    the shared RepSegmenter, and return one aggregated feature dict per
    completed rep."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: could not open {video_path}", file=sys.stderr)
        return []

    segmenter = RepSegmenter()
    completed_reps = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = pose.detect(rgb)
        if landmarks is None:
            continue  # skip frames with no detected person (occlusion, edges)

        frame_feats = compute_frame_features(landmarks)
        completed = segmenter.update(frame_feats["knee_angle_mean"], frame_feats)
        if completed is not None and len(completed) >= 5:
            # discard implausibly short "reps" (likely segmentation noise)
            completed_reps.append(aggregate_rep_features(completed))

    cap.release()
    return completed_reps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--labels_csv", required=True,
                         help="CSV with columns: filename, exercise, label")
    parser.add_argument("--out", default="data/processed/rep_features.csv")
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_csv)
    required_cols = {"filename", "exercise", "label"}
    missing = required_cols - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels_csv missing required columns: {missing}")

    videos_dir = Path(args.videos_dir)
    all_rows = []

    with PoseDetector() as pose:
        for _, row in labels_df.iterrows():
            video_path = videos_dir / row["filename"]
            if not video_path.exists():
                print(f"  WARNING: {video_path} not found, skipping", file=sys.stderr)
                continue

            print(f"Processing {video_path.name} (exercise={row['exercise']}, label={row['label']}) ...")
            reps = extract_reps_from_video(str(video_path), pose)
            print(f"  -> {len(reps)} reps detected")

            for rep in reps:
                rep["exercise"] = row["exercise"]
                rep["label"] = row["label"]
                rep["source_video"] = row["filename"]
                all_rows.append(rep)

    if not all_rows:
        print("No reps extracted -- check video paths and detection thresholds.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(all_rows)
    out_cols = FEATURE_COLUMNS + ["exercise", "label", "source_video"]
    out_df = out_df[out_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rep rows from {labels_df['filename'].nunique()} videos to {out_path}")
    print(out_df["label"].value_counts(normalize=True).rename("class_balance"))


if __name__ == "__main__":
    main()
