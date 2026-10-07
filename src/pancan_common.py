"""Shared loading and cleaning logic for the TCGA pan-cancer RNA-seq analysis.

Used by both pancan_preprocessing.py (inspection and the browsable example
subset) and pancan_train_models.py (the real run), so the two never drift
on what "cleaned" means.
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

RAW_DIR = Path("data/raw/tcga_pancan/TCGA-PANCAN-HiSeq-801x20531")
DATA_CSV = RAW_DIR / "data.csv"
LABELS_CSV = RAW_DIR / "labels.csv"


def load_raw():
    """Load the full 801 x 20531 expression matrix and its class labels."""
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found. Download the dataset first, see README 'How to run it'."
        )
    X = pd.read_csv(DATA_CSV, index_col=0)
    y = pd.read_csv(LABELS_CSV, index_col=0)["Class"]
    return X, y


def drop_zero_variance(X, threshold=0.0):
    """Drop genes with zero (or near zero) variance across all 801 samples.

    This is label independent, so doing it on the full dataset before any
    train/test split is not a leakage risk, unlike the per-class feature
    selection in pancan_train_models.py, which is fit on the training fold
    only.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    kept_columns = X.columns[selector.get_support()]
    return pd.DataFrame(X_filtered, index=X.index, columns=kept_columns)
