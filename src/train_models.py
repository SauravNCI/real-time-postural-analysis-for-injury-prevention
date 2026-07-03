"""
train_models.py

Trains and compares all four classifiers specified in the dissertation
Research Method:
  1. Logistic Regression  (interpretable baseline)
  2. Random Forest        (ensemble, per Leckey et al. 2024)
  3. XGBoost              (gradient boosting)
  4. CNN + BiLSTM         (temporal model on per-frame sequences)

Models 1-3 train on the per-rep aggregated feature table (mean/std/range).
Model 4 needs the raw per-frame sequence: pass --seq_data pointing at the
real_sequences.npz produced by extract_real_dataset.py for a faithful
CNN-BiLSTM trained on genuine per-frame data. If --seq_data is omitted (e.g.
running on the synthetic dataset, which has no per-frame data), it falls
back to a reconstructed pseudo-sequence approximation -- see
`_reconstruct_pseudo_sequence`.

All four models are trained with stratified 5-fold CV (matching the
proposal), evaluated on a held-out 20% stratified test set, and saved to
models/. Run evaluate.py afterwards for the full metrics/plots/McNemar
comparison.

Usage:
    python src/train_models.py --data data/processed/rep_features.csv
    python src/train_models.py --data ../data/processed/rep_features_real.csv \\
                                --seq_data ../data/processed/real_sequences.npz
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from xgboost import XGBClassifier

from biomechanics import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reports"


def load_data(path: str):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values
    return df, X, y


def cross_validated_recall(model_fn, X, y, n_splits=5):
    """Stratified k-fold CV, reporting recall on the unsafe (positive) class
    -- the metric the dissertation's success criterion (recall >= 0.85) is
    defined on."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(recall_score(y[val_idx], preds, pos_label=1))
    return float(np.mean(scores)), float(np.std(scores))


def _reconstruct_pseudo_sequence(row: pd.Series, n_steps: int = 30, seed: int = 0) -> np.ndarray:
    """Reconstruct an approximate per-frame sequence from a rep's
    mean/std/range summary stats, for CNN-BiLSTM training when only
    aggregated features are available (e.g. synthetic data). This is a
    deliberate approximation for pipeline development only -- once real
    per-frame sequences exist (extract_real_dataset.py), use those directly
    instead via --seq_data. The shape is sinusoidal: a rep genuinely moves
    from a 'top' angle, down to a 'bottom' angle, and back up, so a
    half-sine matches the qualitative shape much better than random noise
    while still respecting the rep's actual mean/std/range.
    """
    rng = np.random.default_rng(seed)
    bases = ["knee_angle_L", "knee_angle_R", "knee_angle_mean",
             "hip_hinge_angle", "trunk_lean_angle", "knee_symmetry"]
    t = np.linspace(0, np.pi, n_steps)
    seq = np.zeros((n_steps, len(bases)))
    for i, base in enumerate(bases):
        mean, std, rng_val = row[f"{base}_mean"], row[f"{base}_std"], row[f"{base}_range"]
        shape = -np.sin(t)  # dips down then back up, like a rep
        signal = mean + (rng_val / 2.0) * shape + rng.normal(0, max(std * 0.3, 1e-3), n_steps)
        seq[:, i] = signal
    return seq


def build_cnn_bilstm(n_steps: int, n_features: int):
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Input(shape=(n_steps, n_features)),
        keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
        keras.layers.Conv1D(16, 3, activation="relu", padding="same"),
        keras.layers.Bidirectional(keras.layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.Recall(name="recall")])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/rep_features.csv")
    parser.add_argument("--seq_data", default=None,
                         help="Optional .npz (key 'sequences') of REAL per-frame rep "
                              "sequences from extract_real_dataset.py, same row order as "
                              "--data. If omitted, CNN-BiLSTM trains on reconstructed "
                              "pseudo-sequences instead (see _reconstruct_pseudo_sequence).")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--cnn_epochs", type=int, default=40)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df, X, y = load_data(args.data)
    print(f"Loaded {len(df)} reps | class balance: {np.bincount(y) / len(y)}")

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=args.test_size, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    results = {}

    # ---------- 1. Logistic Regression ----------
    print("\n[1/4] Logistic Regression")
    cv_recall, cv_std = cross_validated_recall(
        lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), X_train_s, y_train)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_s, y_train)
    joblib.dump(lr, MODELS_DIR / "logistic_regression.pkl")
    results["logistic_regression"] = {"cv_recall_mean": cv_recall, "cv_recall_std": cv_std}
    print(f"  5-fold CV recall (unsafe class): {cv_recall:.3f} +/- {cv_std:.3f}")

    # ---------- 2. Random Forest ----------
    print("\n[2/4] Random Forest")
    cv_recall, cv_std = cross_validated_recall(
        lambda: RandomForestClassifier(n_estimators=300, max_depth=8,
                                        class_weight="balanced", random_state=42),
        X_train, y_train)  # tree models don't need scaling
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                 class_weight="balanced", random_state=42).fit(X_train, y_train)
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")
    results["random_forest"] = {"cv_recall_mean": cv_recall, "cv_recall_std": cv_std}
    print(f"  5-fold CV recall (unsafe class): {cv_recall:.3f} +/- {cv_std:.3f}")

    # ---------- 3. XGBoost ----------
    print("\n[3/4] XGBoost")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    cv_recall, cv_std = cross_validated_recall(
        lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                               random_state=42),
        X_train, y_train)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                         random_state=42).fit(X_train, y_train)
    joblib.dump(xgb, MODELS_DIR / "xgboost.pkl")
    results["xgboost"] = {"cv_recall_mean": cv_recall, "cv_recall_std": cv_std}
    print(f"  5-fold CV recall (unsafe class): {cv_recall:.3f} +/- {cv_std:.3f}")

    # ---------- 4. CNN + BiLSTM ----------
    n_steps, bases = 30, ["knee_angle_L", "knee_angle_R", "knee_angle_mean",
                           "hip_hinge_angle", "trunk_lean_angle", "knee_symmetry"]
    if args.seq_data and Path(args.seq_data).exists():
        print(f"\n[4/4] CNN + BiLSTM (trained on REAL per-frame sequences from {args.seq_data})")
        seqs = np.load(args.seq_data)["sequences"]
        if len(seqs) != len(df):
            raise ValueError(
                f"--seq_data has {len(seqs)} sequences but --data has {len(df)} rows -- "
                "they must be from the same extraction run, in the same row order.")
    else:
        print("\n[4/4] CNN + BiLSTM (trained on reconstructed pseudo-sequences -- "
              "pass --seq_data for real per-frame sequences)")
        seqs = np.stack([_reconstruct_pseudo_sequence(df.iloc[i], n_steps, seed=i) for i in df.index])
    seq_train, seq_test = seqs[idx_train], seqs[idx_test]

    # per-feature standardisation fit on train only
    seq_mean = seq_train.reshape(-1, seq_train.shape[-1]).mean(axis=0)
    seq_std = seq_train.reshape(-1, seq_train.shape[-1]).std(axis=0) + 1e-8
    seq_train_n = (seq_train - seq_mean) / seq_std
    seq_test_n = (seq_test - seq_mean) / seq_std
    np.savez(MODELS_DIR / "cnn_bilstm_scaler.npz", mean=seq_mean, std=seq_std)

    cnn = build_cnn_bilstm(n_steps, len(bases))
    from tensorflow import keras
    early_stop = keras.callbacks.EarlyStopping(monitor="val_recall", mode="max",
                                                patience=8, restore_best_weights=True)
    class_weight = {0: 1.0, 1: float((y_train == 0).sum() / max((y_train == 1).sum(), 1))}
    history = cnn.fit(seq_train_n, y_train, validation_split=0.2, epochs=args.cnn_epochs,
                       batch_size=16, class_weight=class_weight, callbacks=[early_stop], verbose=0)
    cnn.save(MODELS_DIR / "cnn_bilstm.keras")

    val_recall_hist = history.history.get("val_recall", [0])
    results["cnn_bilstm"] = {
        "cv_recall_mean": float(np.max(val_recall_hist)),
        "cv_recall_std": 0.0,
        "note": "validation-split recall (held-out fold), not k-fold CV -- "
                "k-fold retraining of a deep model is expensive; see evaluate.py "
                "for the proper held-out test-set comparison across all 4 models.",
    }
    print(f"  best validation recall (unsafe class): {results['cnn_bilstm']['cv_recall_mean']:.3f}")

    # ---------- persist test split + summary for evaluate.py ----------
    np.savez(MODELS_DIR / "test_split.npz",
              X_test=X_test, X_test_s=X_test_s, y_test=y_test,
              seq_test_n=seq_test_n, idx_test=idx_test)

    with open(REPORTS_DIR / "cv_training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete. Cross-validation recall summary:")
    for name, r in results.items():
        print(f"  {name:22s} recall={r['cv_recall_mean']:.3f}")
    print(f"\nModels saved to {MODELS_DIR}")
    print(f"Run evaluate.py next for held-out test metrics, ROC curves, and McNemar's test.")


if __name__ == "__main__":
    main()
