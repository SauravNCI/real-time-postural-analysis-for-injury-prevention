"""
shap_analysis.py

Runs SHAP TreeExplainer on the best-performing tree-based model (Random
Forest or XGBoost -- picked automatically by test-set recall, matching the
dissertation's evaluation criteria) to identify which biomechanical features
drive "unsafe posture" predictions.

Produces:
  - outputs/figures/shap_summary.png       (global feature importance + direction)
  - outputs/figures/shap_waterfall_0.png   (local explanation for one test rep)
  - outputs/reports/shap_feature_ranking.md

Usage:
    python src/shap_analysis.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from biomechanics import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
REPORT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reports"


def pick_best_tree_model():
    """Pick RF vs XGBoost by test-set recall (the tree-based models only --
    SHAP TreeExplainer needs a tree model; the CNN-BiLSTM would need
    DeepExplainer/GradientExplainer, deliberately out of scope per the
    dissertation's plan to apply SHAP to 'the best-performing tree-based
    model')."""
    with open(REPORT_DIR / "test_metrics.json") as f:
        report = json.load(f)
    tree_metrics = [m for m in report["metrics"] if m["model"] in ("random_forest", "xgboost")]
    best = max(tree_metrics, key=lambda m: m["recall"])
    name = best["model"]
    model = joblib.load(MODELS_DIR / f"{name}.pkl")
    return name, model


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    name, model = pick_best_tree_model()
    print(f"Running SHAP on best tree-based model: {name}")

    d = np.load(MODELS_DIR / "test_split.npz")
    X_test = d["X_test"]
    X_test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)
    # Some sklearn/xgboost + shap version combos return a list [class0, class1]
    # or a (n_samples, n_features, n_classes) array for binary classifiers --
    # normalise to the positive-class (unsafe) 2D array either way.
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # ---- global summary plot ----
    plt.figure()
    shap.summary_plot(sv, X_test_df, show=False)
    plt.tight_layout()
    summary_path = FIG_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP summary plot -> {summary_path}")

    # ---- global feature ranking table (mean |SHAP value|) ----
    mean_abs_shap = np.abs(sv).mean(axis=0)
    ranking = (pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs_shap})
               .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    md_lines = ["| Rank | Feature | Mean |SHAP value| |", "|---|---|---|"]
    for i, row in ranking.head(10).iterrows():
        md_lines.append(f"| {i + 1} | {row['feature']} | {row['mean_abs_shap']:.4f} |")
    with open(REPORT_DIR / "shap_feature_ranking.md", "w") as f:
        f.write(f"SHAP global feature ranking -- model: {name}\n\n" + "\n".join(md_lines))
    print(f"Top 5 features by mean |SHAP value|:")
    print(ranking.head(5).to_string(index=False))

    # ---- local waterfall for one test rep (index 0) ----
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(np.shape(expected_value)) else expected_value

    exp = shap.Explanation(values=sv[0], base_values=expected_value,
                            data=X_test_df.iloc[0].values, feature_names=FEATURE_COLUMNS)
    plt.figure()
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    waterfall_path = FIG_DIR / "shap_waterfall_0.png"
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved local SHAP waterfall (test rep 0) -> {waterfall_path}")


if __name__ == "__main__":
    main()
