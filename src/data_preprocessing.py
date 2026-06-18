"""
data_preprocessing.py
---------------------
Load the Breast Cancer Wisconsin dataset from scikit-learn, convert it to a
pandas DataFrame, perform basic checks, and save a processed CSV for modeling.

No feature scaling is applied here — scaling is done inside training pipelines
after the train/test split to prevent data leakage.

Input:  sklearn.datasets.load_breast_cancer (built-in, no download required)
Output: data/processed/breast_cancer_processed.csv

Usage (from repo root):
    python src/data_preprocessing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "breast_cancer_processed.csv"
TARGET_COLUMN = "target"


def load_breast_cancer_data() -> pd.DataFrame:
    """
    Load the Wisconsin Breast Cancer dataset from scikit-learn.

    Returns a DataFrame with 30 numeric features and a binary target column:
      0 = malignant, 1 = benign
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[TARGET_COLUMN] = data.target
    return df


def check_missing_values(df: pd.DataFrame) -> int:
    """Report missing values and return the total count."""
    missing_per_col = df.isnull().sum()
    total_missing = int(missing_per_col.sum())

    if total_missing == 0:
        print("[check] No missing values found.")
    else:
        print("[check] Missing values per column:")
        print(missing_per_col[missing_per_col > 0])
        print(f"[check] Total missing values: {total_missing}")

    return total_missing


def print_summary(df: pd.DataFrame) -> None:
    """Print a clean summary of the processed dataset."""
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    class_labels = {0: "malignant", 1: "benign"}

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Rows:    {len(df)}")
    print(f"Columns: {df.shape[1]} ({df.shape[1] - 1} features + 1 target)")
    print("\nClass balance:")
    for label, count in class_counts.items():
        pct = 100 * count / len(df)
        print(f"  {label} ({class_labels[label]}): {count} ({pct:.1f}%)")
    print(f"\nOutput:  {OUTPUT_FILE.resolve()}")
    print("=" * 50)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[load]  Loading Breast Cancer Wisconsin dataset from scikit-learn ...")
    df = load_breast_cancer_data()
    print(f"[load]  Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    check_missing_values(df)

    # Save raw (unscaled) features + target — scaling happens during training
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[save]  Processed data -> {OUTPUT_FILE}")

    print_summary(df)
    print("\n[done]  Preprocessing complete.")


if __name__ == "__main__":
    main()
