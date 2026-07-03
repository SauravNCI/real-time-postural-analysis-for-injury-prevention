"""
extract_real_dataset.py

Walks the REAL dataset as you actually have it laid out:
    data/<exercise name>/*.mp4 | *.MOV
e.g. data/squat/squat_1.MOV, data/deadlift/deadlift_7.mp4, ...
(this is the raw Kaggle "Workout/Exercises Video" folder structure -- each
folder name IS the exercise label, there is no safe/unsafe column anywhere
in the source data).

For each video this script:
  1. Runs MediaPipe Pose over every frame (see pose_backend.py + biomechanics.py)
  2. Segments completed reps via the shared RepSegmenter
  3. Aggregates each rep into the mean/std/range feature vector the models
     train on (same schema as synthetic_data.py, so nothing downstream needs
     to change)
  4. ALSO saves a fixed-length (30-step) resampled per-frame sequence for
     each rep into data/processed/real_sequences.npz -- this replaces the
     pseudo-sequence reconstruction hack in train_models.py with real
     temporal data for the CNN-BiLSTM once you run this.
  5. If a video produces zero clean rep-cycles (common on unusual camera
     angles / partial clips), falls back to treating the whole video as one
     rep, so you don't silently lose entire videos of data.

Progress is checkpointed: already-processed videos (tracked by filename in
the output CSV) are skipped on re-run, so you can stop and resume freely --
useful since this dataset is ~800MB across 75 videos and processing all of
it in one sitting isn't required. Labels are re-applied on EVERY checkpoint
(not just at the very end), so even an interrupted run leaves a fully
usable, labelled CSV -- don't remove that from the loop below.

After extraction, labels are assigned via label_rules.py's automatic
heuristic (see that file for the full justification -- there is no expert
ground truth in this dataset, this is the documented, transparent stand-in
for it). Pass --labels_csv to override specific videos with real expert
labels once you have any (see data/raw/LABELLING_PROTOCOL.md).

NOTE ON MEDIAPIPE: this uses the current Tasks API via pose_backend.py, not
the older mp.solutions.pose API -- see that module's docstring for why. The
first run will download a ~30MB model file automatically.

Usage:
    python extract_real_dataset.py
    python extract_real_dataset.py --exercises squat deadlift "romanian deadlift"
    python extract_real_dataset.py --labels_csv ../data/raw/labels.csv
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from biomechanics import (compute_frame_features, RepSegmenter,
                           aggregate_rep_features, FEATURE_COLUMNS)
from label_rules import auto_label, merge_with_manual_labels
from pose_backend import PoseDetector

DEFAULT_EXERCISES = ["squat", "deadlift", "romanian deadlift"]
SEQ_STEPS = 30  # fixed-length resample target for CNN-BiLSTM sequences
SEQ_FEATURE_BASES = ["knee_angle_L", "knee_angle_R", "knee_angle_mean",
                      "hip_hinge_angle", "trunk_lean_angle", "knee_symmetry"]


def resample_sequence(rep_frames, n_steps: int = SEQ_STEPS) -> np.ndarray:
    """Linearly resample a variable-length list of per-frame feature dicts
    to a fixed n_steps x n_features array (real data equivalent of
    train_models.py's _reconstruct_pseudo_sequence, but from genuine
    per-frame measurements instead of a synthetic sine approximation)."""
    n_frames = len(rep_frames)
    src_t = np.linspace(0, 1, n_frames)
    tgt_t = np.linspace(0, 1, n_steps)
    seq = np.zeros((n_steps, len(SEQ_FEATURE_BASES)))
    for i, base in enumerate(SEQ_FEATURE_BASES):
        values = np.array([f[base] for f in rep_frames], dtype=float)
        seq[:, i] = np.interp(tgt_t, src_t, values)
    return seq


def extract_reps_from_video(video_path: Path, pose: PoseDetector):
    """Run MediaPipe over every frame, segment reps, and return a list of
    (agg_features_dict, raw_frame_sequence) tuples, one per completed rep.
    Falls back to treating the whole video as a single rep if the
    RepSegmenter never completes a full cycle (common on non-side-on camera
    angles in a YouTube-sourced dataset)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: could not open {video_path.name}", file=sys.stderr)
        return []

    segmenter = RepSegmenter()
    completed = []
    all_frames = []  # fallback buffer: every frame with a detected person

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = pose.detect(rgb)
        if landmarks is None:
            continue

        feats = compute_frame_features(landmarks)
        all_frames.append(feats)

        rep = segmenter.update(feats["knee_angle_mean"], feats)
        if rep is not None and len(rep) >= 5:
            completed.append(rep)

    cap.release()

    if not completed and len(all_frames) >= 8:
        # Fallback: whole video as one rep (better than losing the video).
        completed.append(all_frames)

    return [(aggregate_rep_features(r), resample_sequence(r)) for r in completed]


def find_videos(data_root: Path, exercise: str):
    folder = data_root / exercise
    if not folder.exists():
        return []
    exts = {".mp4", ".mov", ".mkv", ".avi"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../data",
                         help="Folder containing one subfolder per exercise (default: ../data)")
    parser.add_argument("--exercises", nargs="+", default=DEFAULT_EXERCISES,
                         help="Which exercise subfolders to process")
    parser.add_argument("--out", default="../data/processed/rep_features_real.csv")
    parser.add_argument("--seq_out", default="../data/processed/real_sequences.npz")
    parser.add_argument("--labels_csv", default=None,
                         help="Optional expert-labelled CSV (source_video,label) -- "
                              "overrides the auto heuristic for videos it covers")
    parser.add_argument("--max_videos", type=int, default=None,
                         help="Optional cap, useful for a quick test run")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out)
    seq_out_path = Path(args.seq_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip videos already present in the output CSV.
    already_done = set()
    existing_rows = []
    existing_seqs = []
    if out_path.exists():
        prev = pd.read_csv(out_path)
        already_done = set(prev["source_video"].unique())
        existing_rows = prev.to_dict("records")
        if seq_out_path.exists():
            existing_seqs = list(np.load(seq_out_path)["sequences"])
            if len(existing_seqs) != len(existing_rows):
                print("  WARNING: sequence/row count mismatch on resume, "
                      "re-extracting sequences is recommended", file=sys.stderr)

    videos = []
    for ex in args.exercises:
        vids = find_videos(data_root, ex)
        print(f"{ex}: found {len(vids)} videos")
        videos.extend((ex, v) for v in vids)

    todo = [(ex, v) for ex, v in videos if v.name not in already_done]
    if args.max_videos:
        todo = todo[: args.max_videos]
    print(f"\n{len(already_done)} videos already processed (resuming), "
          f"{len(todo)} to go this run.\n")

    rows = list(existing_rows)
    seqs = list(existing_seqs)

    with PoseDetector() as pose:
        for i, (exercise, video_path) in enumerate(todo, 1):
            t0 = time.time()
            reps = extract_reps_from_video(video_path, pose)
            dt = time.time() - t0
            print(f"[{i}/{len(todo)}] {exercise}/{video_path.name}: "
                  f"{len(reps)} reps ({dt:.1f}s)")

            for agg, seq in reps:
                agg["exercise"] = exercise
                agg["source_video"] = video_path.name
                rows.append(agg)
                seqs.append(seq)

            # Checkpoint after every video so a stopped run loses nothing.
            # Labels are re-applied on every checkpoint (not just at the end)
            # so an interrupted run still leaves a fully usable, labelled CSV.
            out_df = pd.DataFrame(rows)
            cols = FEATURE_COLUMNS + ["exercise", "source_video"]
            out_df = out_df[[c for c in cols if c in out_df.columns]]
            out_df = auto_label(out_df)
            if args.labels_csv:
                out_df = merge_with_manual_labels(out_df, args.labels_csv)
            out_df.to_csv(out_path, index=False)
            if seqs:
                np.savez(seq_out_path, sequences=np.stack(seqs))

    if not rows:
        print("No reps extracted -- check --data_root and video files.", file=sys.stderr)
        sys.exit(1)

    # ---- final labelling pass (idempotent -- matches the per-checkpoint pass above) ----
    final_df = pd.DataFrame(rows)
    cols = FEATURE_COLUMNS + ["exercise", "source_video"]
    final_df = final_df[[c for c in cols if c in final_df.columns]]
    final_df = auto_label(final_df)
    if args.labels_csv:
        final_df = merge_with_manual_labels(final_df, args.labels_csv)
    final_df.to_csv(out_path, index=False)

    print(f"\nWrote {len(final_df)} rep rows ({final_df['source_video'].nunique()} videos) to {out_path}")
    print(final_df["label"].value_counts(normalize=True).rename("class_balance"))
    print(final_df.groupby("exercise")["label"].value_counts(normalize=True))
    print(f"Sequences saved to {seq_out_path} (shape: {np.load(seq_out_path)['sequences'].shape})")


if __name__ == "__main__":
    main()
