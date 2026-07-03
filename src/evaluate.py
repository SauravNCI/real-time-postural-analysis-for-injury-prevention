"""
evaluate.py

Loads the models + held-out test split saved by train_models.py and produces
the full evaluation the dissertation commits to:
  - accuracy, precision, recall, F1 per model on the 20% held-out test set
  - confusion matrices
  - ROC curves with AUC
  - McNemar's test between each pair of models (statistical significance
    of performance differences)
  - a markdown comparison table + all figures saved to outputs/

Usage:
    python src/evaluate.py
"""

import json
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc,
                              ConfusionMatrixDisplay)
from statsmodels.stats.contingency_tables import mcnemar

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reports"


def load_test_split():
    d = np.load(MODELS_DIR / "test_split.npz")
    return d["X_test"], d["X_test_s"], d["y_test"], d["seq_test_n"]


def get_predictions():
    X_test, X_test_s, y_test, seq_test_n = load_test_split()
    preds, probs = {}, {}

    lr = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    preds["logistic_regression"] = lr.predict(X_test_s)
    probs["logistic_regression"] = lr.predict_proba(X_test_s)[:, 1]

    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    preds["random_forest"] = rf.predict(X_test)
    probs["random_forest"] = rf.predict_proba(X_test)[:, 1]

    xgb = joblib.load(MODELS_DIR / "xgboost.pkl")
    preds["xgboost"] = xgb.predict(X_test)
    probs["xgboost"] = xgb.predict_proba(X_test)[:, 1]

    from tensorflow import keras
    cnn = keras.models.load_model(MODELS_DIR / "cnn_bilstm.keras")
    cnn_prob = cnn.predict(seq_test_n, verbose=0).ravel()
    probs["cnn_bilstm"] = cnn_prob
    preds["cnn_bilstm"] = (cnn_prob >= 0.5).astype(int)

    return y_test, preds, probs


def compute_metrics_table(y_test, preds):
    rows = []
    for name, y_pred in preds.items():
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "meets_recall_target (>=0.85)": recall_score(y_test, y_pred, zero_division=0) >= 0.85,
        })
    return rows


def plot_confusion_matrices(y_test, preds):
    fig, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 4))
    for ax, (name, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["safe", "unsafe"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    path = FIG_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_roc_curves(y_test, probs):
    plt.figure(figsize=(6, 6))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (recall)")
    plt.title("ROC curves -- unsafe posture classification")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    path = FIG_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def mcnemar_comparison(y_test, preds):
    """Pairwise McNemar's test between every pair of models, as committed to
    in the dissertation evaluation plan. Tests whether two models'
    disagreements with each other are symmetric (null: no significant
    difference in error rates)."""
    results = []
    for m1, m2 in combinations(preds.keys(), 2):
        correct1 = (preds[m1] == y_test)
        correct2 = (preds[m2] == y_test)
        # 2x2 contingency table of (m1 correct, m2 correct)
        both_correct = int(np.sum(correct1 & correct2))
        only1_correct = int(np.sum(correct1 & ~correct2))
        only2_correct = int(np.sum(~correct1 & correct2))
        both_wrong = int(np.sum(~correct1 & ~correct2))
        table = [[both_correct, only1_correct], [only2_correct, both_wrong]]
        result = mcnemar(table, exact=(only1_correct + only2_correct) < 25)
        results.append({
            "model_a": m1, "model_b": m2,
            "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "significant_at_0.05": bool(result.pvalue < 0.05),
        })
    return results


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    y_test, preds, probs = get_predictions()

    metrics = compute_metrics_table(y_test, preds)
    print("\n=== Held-out test set metrics ===")
    header = f"{'model':22s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s}  meets_target"
    print(header)
    for m in metrics:
        print(f"{m['model']:22s} {m['accuracy']:.3f}  {m['precision']:.3f}  "
              f"{m['recall']:.3f}  {m['f1']:.3f}   {m['meets_recall_target (>=0.85)']}")

    cm_path = plot_confusion_matrices(y_test, preds)
    roc_path = plot_roc_curves(y_test, probs)
    print(f"\nSaved confusion matrices -> {cm_path}")
    print(f"Saved ROC curves -> {roc_path}")

    mcnemar_results = mcnemar_comparison(y_test, preds)
    print("\n=== McNemar's test (pairwise significance of model differences) ===")
    for r in mcnemar_results:
        sig = "significant" if r["significant_at_0.05"] else "not significant"
        print(f"  {r['model_a']:20s} vs {r['model_b']:20s}  p={r['p_value']:.4f}  ({sig})")

    best_model = max(metrics, key=lambda m: m["recall"])["model"]
    print(f"\nBest model by recall on unsafe class: {best_model}")

    with open(REPORT_DIR / "test_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "mcnemar": mcnemar_results,
                   "best_model_by_recall": best_model}, f, indent=2)

    # markdown table for direct inclusion in the dissertation
    md_lines = ["| Model | Accuracy | Precision | Recall | F1 | Meets recall >=0.85 |",
                "|---|---|---|---|---|---|"]
    for m in metrics:
        md_lines.append(f"| {m['model']} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
                         f"{m['recall']:.3f} | {m['f1']:.3f} | "
                         f"{'Yes' if m['meets_recall_target (>=0.85)'] else 'No'} |")
    with open(REPORT_DIR / "results_table.md", "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nSaved JSON + markdown results table to {REPORT_DIR}")


if __name__ == "__main__":
    main()
