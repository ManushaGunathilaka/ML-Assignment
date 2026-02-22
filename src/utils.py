"""
src/utils.py - Shared helpers for the ML pipeline
===================================================
Centralises paths, logging setup, and model I/O so every other module
can simply ``from src.utils import *``.
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths (relative to repo root)
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
DATA_DIR = ROOT_DIR  # CSV lives in the repo root

for d in (MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a pre-configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.json")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
METRICS_JSON_PATH = os.path.join(OUTPUTS_DIR, "metrics.json")
METRICS_CSV_PATH = os.path.join(OUTPUTS_DIR, "metrics_table.csv")


def save_model(model, path: str = MODEL_PATH) -> None:
    """Save an XGBoost model."""
    model.save_model(path)


def load_model(path: str = MODEL_PATH):
    """Load an XGBoost model."""
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def save_encoders(encoders: dict, path: str = ENCODERS_PATH) -> None:
    joblib.dump(encoders, path)


def load_encoders(path: str = ENCODERS_PATH) -> dict:
    return joblib.load(path)


def save_feature_names(names: list, path: str = FEATURE_NAMES_PATH) -> None:
    with open(path, "w") as f:
        json.dump(names, f)


def load_feature_names(path: str = FEATURE_NAMES_PATH) -> list:
    with open(path) as f:
        return json.load(f)


def save_metrics(metrics: dict, path: str = METRICS_JSON_PATH) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str = METRICS_JSON_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def load_transform_info() -> dict:
    """Load transform info to check if log transform was used during training."""
    transform_path = os.path.join(MODELS_DIR, "transform_info.json")
    if os.path.exists(transform_path):
        with open(transform_path) as f:
            return json.load(f)
    return {"log_transform": False}
