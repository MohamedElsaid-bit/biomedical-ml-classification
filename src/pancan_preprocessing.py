"""Inspect the raw TCGA pan-cancer RNA-seq download and write a small,
browsable example subset. Does not persist the full cleaned matrix; the
training script re-derives it (cheap: one CSV read and a variance filter).

Run from the repo root: python3 src/pancan_preprocessing.py
"""
from pathlib import Path

from pancan_common import drop_zero_variance, load_raw

EXAMPLE_SUBSET_PATH = Path("data/processed/pancan_example_subset.csv")
N_EXAMPLE_SAMPLES = 15
N_EXAMPLE_GENES = 30


def main():
    X, y = load_raw()
    print(f"Raw matrix: {X.shape[0]} samples x {X.shape[1]} genes")
    print(f"Missing values: {int(X.isna().sum().sum())}")
    print("Class counts:")
    print(y.value_counts())

    X_clean = drop_zero_variance(X)
    n_dropped = X.shape[1] - X_clean.shape[1]
    print(f"\nDropped {n_dropped} zero variance genes, {X_clean.shape[1]} remain")

    EXAMPLE_SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    example = X.iloc[:N_EXAMPLE_SAMPLES, :N_EXAMPLE_GENES].copy()
    example.insert(0, "Class", y.iloc[:N_EXAMPLE_SAMPLES].values)
    example.to_csv(EXAMPLE_SUBSET_PATH)
    print(
        f"\nWrote a {N_EXAMPLE_SAMPLES} sample x {N_EXAMPLE_GENES} gene "
        f"browsable example (with labels) to {EXAMPLE_SUBSET_PATH}"
    )
    print(
        "This subset is for browsing only; it is far too small and too few "
        "genes to reproduce the real results, see README 'How to run it' "
        "for the full download."
    )


if __name__ == "__main__":
    main()
