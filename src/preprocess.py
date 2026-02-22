"""
src/preprocess.py - Data Cleaning & Feature Engineering
========================================================
Loads the raw houses_data.csv, cleans it, engineers features, encodes
categoricals, removes outliers, and splits into 70/15/15 train/val/test.

Usage:
    python src/preprocess.py --input houses_data.csv --output processed.csv
"""

import argparse
import re
import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Allow imports when run from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    get_logger, save_encoders, save_feature_names,
    MODELS_DIR, DATA_DIR,
)

log = get_logger("preprocess")

# ---------------------------------------------------------------------------
# 1. Parsing helpers
# ---------------------------------------------------------------------------

def parse_house_size(val: str) -> float:
    """
    Extract numeric sq-ft from strings like '3000  sq ft' or ' sq ft'.
    Returns NaN when the field is blank / unparseable.
    """
    if not isinstance(val, str):
        return np.nan
    val = val.strip()
    # Match leading digits (with optional decimals)
    m = re.match(r"([\d,]+\.?\d*)", val)
    if m:
        return float(m.group(1).replace(",", ""))
    return np.nan  # e.g. " sq ft" -> NaN


def parse_land_size(val: str) -> float:
    """
    Extract numeric perches from strings like '10.5 Perches' or ' Perches'.
    Returns NaN when blank / unparseable.
    """
    if not isinstance(val, str):
        return np.nan
    val = val.strip()
    m = re.match(r"([\d,]+\.?\d*)", val)
    if m:
        return float(m.group(1).replace(",", ""))
    return np.nan


def parse_price(val) -> float:
    """
    Handle currency formats: plain float, 'Rs 1,200,000', 'Lakh', 'Mn'.
    The scraped data is already numeric, but this is here for robustness.
    """
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return np.nan
    val = val.strip().replace(",", "").replace("Rs", "").replace("LKR", "").strip()
    try:
        num = float(re.sub(r"[^\d.]", "", val))
    except ValueError:
        return np.nan
    low = val.lower()
    if "mn" in low or "million" in low:
        num *= 1_000_000
    elif "lakh" in low:
        num *= 100_000
    return num

# ---------------------------------------------------------------------------
# 2. Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(input_path: str, output_path: str) -> None:
    log.info("Loading %s ...", input_path)
    df = pd.read_csv(input_path)
    log.info("Raw shape: %s", df.shape)

    # ---- Auto-detect target column -------------------------------------------
    target_col = "price"
    if target_col not in df.columns:
        candidates = [c for c in df.columns if "price" in c.lower()]
        if candidates:
            target_col = candidates[0]
            log.info("Target column auto-detected as '%s'", target_col)
        else:
            raise ValueError("Cannot find a 'price' column. Available: " +
                             str(df.columns.tolist()))
    log.info("Target column: '%s'", target_col)

    # ---- Parse numeric fields ------------------------------------------------
    df["price"] = df[target_col].apply(parse_price)
    df["house_size_sqft"] = df["house_size"].apply(parse_house_size)
    df["land_size_perches"] = df["land_size"].apply(parse_land_size)

    # ---- Drop columns that are not useful for prediction ---------------------
    drop_cols = ["id", "title", "description", "posted_date",
                 "address", "price_type", "house_size", "land_size"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    log.info("Dropped non-predictive columns: %s", drop_cols)

    # ---- Handle missing values -----------------------------------------------
    # Fill missing numeric features with median
    for col in ["house_size_sqft", "land_size_perches"]:
        median_val = df[col].median()
        n_missing = df[col].isna().sum()
        df[col].fillna(median_val, inplace=True)
        log.info("  %s: filled %d NaN with median %.1f", col, n_missing, median_val)

    # Fill missing categoricals
    for col in ["district", "city"]:
        if col in df.columns:
            df[col].fillna("Unknown", inplace=True)

    # ---- Remove price outliers (IQR method) ----------------------------------
    before = len(df)
    q1 = df["price"].quantile(0.01)
    q3 = df["price"].quantile(0.99)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df = df[(df["price"] >= max(lower, 0)) & (df["price"] <= upper)].copy()
    log.info("Outlier removal: %d -> %d rows (removed %d)",
             before, len(df), before - len(df))

    # Also remove rows with price <= 0
    df = df[df["price"] > 0].copy()

    # ---- Remove rows with 0 bedrooms (likely data errors) --------------------
    df = df[df["bedrooms"] > 0].copy()

    # ---- Encode categoricals -------------------------------------------------
    encoders = {}
    for col in ["district", "city"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            log.info("  Label-encoded '%s' (%d classes)", col, len(le.classes_))

    # Convert bool to int
    if "negotiable" in df.columns:
        df["negotiable"] = df["negotiable"].astype(int)

    # ---- Split: 70% train, 15% validation, 15% test -------------------------
    feature_cols = [c for c in df.columns if c != "price"]
    X = df[feature_cols]
    y = df["price"]

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    # Second split: 50/50 temp -> 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    log.info("Split sizes:  train=%d  val=%d  test=%d",
             len(X_train), len(X_val), len(X_test))

    # ---- Save everything -----------------------------------------------------
    # Save full processed dataset
    df.to_csv(output_path, index=False)
    log.info("Saved processed data to %s (%d rows, %d cols)",
             output_path, len(df), len(df.columns))

    # Save splits as CSVs
    splits_dir = os.path.join(DATA_DIR, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    pd.concat([X_train, y_train], axis=1).to_csv(
        os.path.join(splits_dir, "train.csv"), index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(
        os.path.join(splits_dir, "val.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(
        os.path.join(splits_dir, "test.csv"), index=False)

    log.info("Saved train/val/test splits to %s/", splits_dir)

    # Save encoders and feature names
    save_encoders(encoders)
    save_feature_names(feature_cols)
    log.info("Saved encoders and feature names to models/")

    # ---- Print summary -------------------------------------------------------
    log.info("")
    log.info("="*55)
    log.info("PREPROCESSING COMPLETE")
    log.info("  Features: %s", feature_cols)
    log.info("  Target:   price")
    log.info("  Rows:     %d", len(df))
    log.info("="*55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess house data")
    parser.add_argument("--input",  default="houses_data.csv", help="Raw CSV path")
    parser.add_argument("--output", default="processed.csv",   help="Output CSV path")
    args = parser.parse_args()
    preprocess(args.input, args.output)


if __name__ == "__main__":
    main()
