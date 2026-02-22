"""
src/evaluate.py - Model Evaluation
====================================
Computes regression metrics (RMSE, MAE, R2) on train/val/test sets and
generates evaluation plots.

Plots generated:
    1. Predicted vs Actual scatter (test set)
    2. Residual histogram (test set)

Usage:
    python src/evaluate.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    get_logger, load_model, load_feature_names,
    save_metrics, PLOTS_DIR, DATA_DIR, METRICS_JSON_PATH,
    OUTPUTS_DIR,
)

log = get_logger("evaluate")

# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, label: str) -> dict:
    """Return RMSE, MAE, R2 as a dict."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    log.info("  %s  RMSE=%15.0f  MAE=%15.0f  R2=%.4f", label.ljust(6), rmse, mae, r2)
    return {"rmse": round(float(rmse), 2),
            "mae":  round(float(mae), 2),
            "r2":   round(float(r2), 4)}


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(y_true, y_pred, save_path: str) -> None:
    """
    Scatter plot of predicted vs actual prices.
    -- Points close to the diagonal line indicate good predictions.
    -- Spread away from the line shows prediction error.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=15, color="#4a90d9")
    # Perfect prediction line
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price (LKR)", fontsize=12)
    ax.set_ylabel("Predicted Price (LKR)", fontsize=12)
    ax.set_title("Predicted vs Actual House Prices", fontsize=14, fontweight="bold")
    ax.legend()
    ax.ticklabel_format(style="plain")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", save_path)


def plot_residual_histogram(y_true, y_pred, save_path: str) -> None:
    """
    Histogram of residuals (actual - predicted).
    -- A bell-shaped curve centred at 0 means errors are random (good).
    -- Skew or heavy tails indicate systematic under/over-prediction.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals, bins=50, color="#6cc070", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Actual - Predicted) LKR", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Residual Distribution", fontsize=14, fontweight="bold")
    ax.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_transform_info():
    """Load transform info to check if log transform was used."""
    transform_path = os.path.join(DATA_DIR, "models", "transform_info.json")
    if os.path.exists(transform_path):
        with open(transform_path) as f:
            return json.load(f)
    return {"log_transform": False}


def evaluate() -> None:
    log.info("Loading model and data ...")
    model = load_model()
    feature_cols = load_feature_names()
    splits_dir = os.path.join(DATA_DIR, "splits")
    
    # Check if log transform was used
    transform_info = load_transform_info()
    use_log = transform_info.get("log_transform", False)
    if use_log:
        log.info("Model trained with log1p transform - applying expm1 to predictions")

    metrics_all = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(splits_dir, f"{split}.csv"))
        X = df[feature_cols]
        y = df["price"]
        preds = model.predict(X)
        
        # Convert back from log scale if needed
        if use_log:
            preds = np.expm1(preds)
        
        metrics_all[split] = compute_metrics(y, preds, split)

    # Save metrics
    save_metrics(metrics_all)
    log.info("Saved metrics to %s", METRICS_JSON_PATH)

    # Save as CSV table too
    rows = []
    for split, m in metrics_all.items():
        rows.append({"split": split, **m})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUTS_DIR, "metrics_table.csv"), index=False
    )

    # ---- Plots (test set) ----------------------------------------------------
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    X_test = test_df[feature_cols]
    y_test = test_df["price"]
    y_pred = model.predict(X_test)
    
    # Convert back from log scale if needed
    if use_log:
        y_pred = np.expm1(y_pred)

    plot_predicted_vs_actual(
        y_test, y_pred,
        os.path.join(PLOTS_DIR, "predicted_vs_actual.png")
    )
    plot_residual_histogram(
        y_test, y_pred,
        os.path.join(PLOTS_DIR, "residual_histogram.png")
    )

    log.info("")
    log.info("=" * 55)
    log.info("EVALUATION COMPLETE")
    log.info("=" * 55)


if __name__ == "__main__":
    evaluate()
