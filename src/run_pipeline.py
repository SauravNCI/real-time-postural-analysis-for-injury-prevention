"""
run_pipeline.py

Runs the full offline pipeline in the correct order. Useful both for the
synthetic-data dry run and as a template for the real-data run.

Usage:
    python src/run_pipeline.py --use_synthetic     # regenerate + use synthetic data
    python src/run_pipeline.py --data ../data/processed/rep_features_real.csv \\
                                --seq_data ../data/processed/real_sequences.npz
"""

import argparse
import subprocess
import sys
from pathlib import Path

STEPS = ["train_models.py", "evaluate.py", "shap_analysis.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/rep_features.csv")
    parser.add_argument("--seq_data", default=None,
                         help="Optional real_sequences.npz for a genuine (non-reconstructed) "
                              "CNN-BiLSTM -- see train_models.py --seq_data")
    parser.add_argument("--use_synthetic", action="store_true",
                         help="Regenerate synthetic_data.py output before training")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent

    if args.use_synthetic:
        print("=" * 60)
        print("STEP 0: generating synthetic dataset")
        print("=" * 60)
        subprocess.run([sys.executable, "synthetic_data.py"], cwd=src_dir, check=True)

    for i, step in enumerate(STEPS, start=1):
        print("\n" + "=" * 60)
        print(f"STEP {i}: {step}")
        print("=" * 60)
        cmd = [sys.executable, step]
        if step == "train_models.py":
            cmd += ["--data", str((Path.cwd() / args.data).resolve())
                    if not Path(args.data).is_absolute() else args.data]
            if args.seq_data:
                cmd += ["--seq_data", str((Path.cwd() / args.seq_data).resolve())
                        if not Path(args.seq_data).is_absolute() else args.seq_data]
        subprocess.run(cmd, cwd=src_dir, check=True)

    print("\n" + "=" * 60)
    print("Pipeline complete. See outputs/reports/results_table.md and "
          "outputs/figures/ for the full comparison.")
    print("=" * 60)


if __name__ == "__main__":
    main()
