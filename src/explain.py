"""
src/explain.py - SHAP Explainability & Feature Importance
==========================================================
Uses SHAP TreeExplainer (optimised for tree-based models like XGBoost) to
produce global and local explanations of the house-price predictions.

Plots generated:
    1. SHAP Summary (beeswarm) plot   -- shows which features matter most
       and how their values push predictions up or down.
    2. SHAP Dependence plot           -- shows how the top feature's value
       relates to its SHAP value (i.e. its effect on price).
    3. Feature Importance bar chart   -- simple bar chart from XGBoost's
       built-in gain-based importance.

Usage:
    python src/explain.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    get_logger, load_model, load_feature_names,
    PLOTS_DIR, DATA_DIR,
)

log = get_logger("explain")


def explain() -> None:
    log.info("Loading model and test data ...")
    model = load_model()
    feature_cols = load_feature_names()

    splits_dir = os.path.join(DATA_DIR, "splits")
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    X_test = test_df[feature_cols]

    # ------------------------------------------------------------------
    # 1. SHAP TreeExplainer
    #    TreeExplainer is the best choice for XGBoost because it computes
    #    exact Shapley values in polynomial time (vs exponential for
    #    KernelExplainer), making it both fast and precise.
    # ------------------------------------------------------------------
    log.info("Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # ------------------------------------------------------------------
    # 2. SHAP Summary Plot (beeswarm)
    #    Each dot is one data point. X-axis = SHAP value (impact on
    #    prediction). Colour = feature value (red = high, blue = low).
    #    Features are sorted by overall importance. This tells us:
    #      - WHICH features matter most (top rows)
    #      - HOW feature values affect the price (colour + direction)
    # ------------------------------------------------------------------
    log.info("Generating SHAP summary plot ...")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot - Feature Impact on Price", fontsize=14, fontweight="bold")
    plt.tight_layout()
    summary_path = os.path.join(PLOTS_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info("Saved: %s", summary_path)

    # ------------------------------------------------------------------
    # 3. SHAP Dependence Plot (top feature)
    #    Shows the relationship between one feature's value (x-axis)
    #    and its SHAP value (y-axis). Interaction effects are shown
    #    by colouring with the most correlated feature.
    #    This helps answer: "How does increasing X affect the price?"
    # ------------------------------------------------------------------
    # Find top feature by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs_shap))
    top_feature = feature_cols[top_idx]
    log.info("Top feature by mean |SHAP|: %s", top_feature)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(top_idx, shap_values, X_test, show=False, ax=ax)
    ax.set_title(f"SHAP Dependence: {top_feature}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    dep_path = os.path.join(PLOTS_DIR, "shap_dependence.png")
    fig.savefig(dep_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", dep_path)

    # ------------------------------------------------------------------
    # 4. XGBoost Feature Importance (gain-based)
    #    Gain = average reduction in loss when a feature is used to
    #    split. Higher gain = more useful feature. This is simpler than
    #    SHAP but gives a quick overview.
    # ------------------------------------------------------------------
    log.info("Generating feature importance bar chart ...")
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    # Map back to readable names
    imp_df = pd.DataFrame([
        {"feature": feature_cols[int(k.replace("f", ""))], "gain": v}
        if k.startswith("f") else {"feature": k, "gain": v}
        for k, v in importance.items()
    ]).sort_values("gain", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"], imp_df["gain"], color="#e8915a")
    ax.set_xlabel("Gain (avg loss reduction)", fontsize=12)
    ax.set_title("XGBoost Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    imp_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    fig.savefig(imp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", imp_path)

    # ------------------------------------------------------------------
    # 5. Save SHAP explainer for Streamlit
    # ------------------------------------------------------------------
    import joblib
    joblib.dump(explainer, os.path.join(PLOTS_DIR, "..", "shap_explainer.pkl"))
    log.info("Saved SHAP explainer for Streamlit.")

    log.info("")
    log.info("=" * 55)
    log.info("EXPLAINABILITY COMPLETE")
    log.info("  Plots saved to: %s", PLOTS_DIR)
    log.info("=" * 55)


if __name__ == "__main__":
    explain()
