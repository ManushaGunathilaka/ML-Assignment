"""
src/train.py - XGBoost Model Training with Optuna Hyperparameter Tuning
========================================================================
Trains an XGBoost regressor on the preprocessed house-price data.

Why XGBoost?
    1. Gradient-boosted trees excel at tabular regression with mixed feature types
    2. Built-in handling of missing values
    3. Native support for early stopping on a validation set
    4. Compatible with SHAP TreeExplainer (fast, exact explanations)
    5. Outperforms basic models (Linear Regression, Decision Tree) because it
       builds an ensemble of weak learners, each correcting the errors of
       the previous one, and applies regularisation to prevent overfitting.

Why Optuna (over RandomizedSearchCV)?
    Optuna uses Bayesian optimisation (Tree-structured Parzen Estimator) which
    is smarter than random search: it learns from prior trials to focus on
    promising hyperparameter regions, generally converging faster.

Usage:
    python src/train.py --data processed.csv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    get_logger, save_model, load_feature_names,
    MODEL_PATH, DATA_DIR,
)

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

log = get_logger("train")

# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: minimise validation RMSE."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1500),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "random_state":      42,
        "tree_method":       "hist",      # fast histogram-based algorithm
        "verbosity":         0,
    }

    model = xgb.XGBRegressor(**params, early_stopping_rounds=50)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_path: str, n_trials: int = 50) -> None:
    # ---- Load splits ---------------------------------------------------------
    splits_dir = os.path.join(DATA_DIR, "splits")
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(splits_dir, "val.csv"))

    feature_cols = load_feature_names()

    X_train = train_df[feature_cols]
    y_train = train_df["price"]
    X_val   = val_df[feature_cols]
    y_val   = val_df["price"]

    log.info("Training data:    %d rows x %d features", *X_train.shape)
    log.info("Validation data:  %d rows x %d features", *X_val.shape)

    # ---- Optuna hyperparameter tuning ----------------------------------------
    log.info("Starting Optuna search (%d trials) ...", n_trials)

    study = optuna.create_study(direction="minimize", study_name="xgb-house-price")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    log.info("Best trial RMSE: %.2f", study.best_value)
    log.info("Best params: %s", best)

    # ---- Retrain final model with best params + early stopping ---------------
    best["random_state"] = 42
    best["tree_method"] = "hist"
    best["verbosity"] = 0

    final_model = xgb.XGBRegressor(**best, early_stopping_rounds=50)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ---- Save model ----------------------------------------------------------
    save_model(final_model, MODEL_PATH)
    log.info("Model saved to %s", MODEL_PATH)

    # Quick sanity check
    val_preds = final_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    log.info("Final model validation RMSE: {:,.0f} LKR".format(val_rmse))

    log.info("")
    log.info("=" * 55)
    log.info("TRAINING COMPLETE")
    log.info("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--data",     default="processed.csv", help="Processed CSV")
    parser.add_argument("--n_trials", type=int, default=50,    help="Optuna trials")
    args = parser.parse_args()
    train(args.data, args.n_trials)


if __name__ == "__main__":
    main()
