"""
train_models.py
---------------
Train three classifiers on the preprocessed breast cancer dataset:
  1. Logistic Regression
  2. Random Forest
  3. Support Vector Machine (SVM)

Uses stratified train/test split and 5-fold stratified cross-validation on the
training set only. StandardScaler is applied inside sklearn Pipelines so it is
fit on training folds only — no data leakage.

Input:  data/processed/breast_cancer_processed.csv
Output: results/metrics/cv_results.csv
        results/metrics/*.pkl (trained models)
        results/metrics/X_test.csv, y_test.csv (held-out test set)

Usage (from repo root):
    python src/train_models.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_FILE = Path("data/processed/breast_cancer_processed.csv")
METRICS_DIR = Path("results/metrics")
TARGET_COLUMN = "target"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def get_project_root() -> Path:
    """Return the repository root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def load_data(filepath: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed CSV and split into features (X) and target (y)."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    print(f"[load]  X shape: {X.shape}")
    print(f"[load]  Class distribution: {y.value_counts().sort_index().to_dict()}")
    return X, y


def define_models() -> dict:
    """
    Return unfitted estimators wrapped in Pipelines where scaling is needed.

    Logistic Regression and SVM are scale-sensitive, so they include StandardScaler.
    Random Forest is tree-based and does not require scaling.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def run_cross_validation(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Run stratified k-fold CV on the training set and return results as DataFrame."""
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    rows = []
    for name, model in models.items():
        print(f"\n[cv]    {name} - {CV_FOLDS}-fold stratified CV ...")
        scores = cross_validate(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )
        row = {"Model": name}
        for metric in scoring:
            key = f"test_{metric}"
            row[f"cv_{metric}_mean"] = round(float(np.mean(scores[key])), 4)
            row[f"cv_{metric}_std"] = round(float(np.std(scores[key])), 4)
            print(
                f"        {metric}: {row[f'cv_{metric}_mean']:.4f} "
                f"+/- {row[f'cv_{metric}_std']:.4f}"
            )
        rows.append(row)

    return pd.DataFrame(rows)


def fit_final_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """Fit each model on the full training split."""
    fitted = {}
    for name, model in models.items():
        print(f"[fit]   Training final {name} ...")
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def save_artifacts(
    fitted_models: dict,
    cv_results: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_dir: Path,
) -> None:
    """Save CV results, trained models, and the held-out test set."""
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cv_path = metrics_dir / "cv_results.csv"
    cv_results.to_csv(cv_path, index=False)
    print(f"\n[save]  CV results -> {cv_path}")

    for name, model in fitted_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        model_path = metrics_dir / filename
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[save]  Model      -> {model_path}")

    X_test.to_csv(metrics_dir / "X_test.csv", index=False)
    y_test.to_csv(metrics_dir / "y_test.csv", index=False)
    print(f"[save]  Test set   -> {metrics_dir / 'X_test.csv'}, y_test.csv")


def main() -> None:
    root = get_project_root()
    processed_path = root / PROCESSED_FILE
    metrics_dir = root / METRICS_DIR

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Run data_preprocessing.py first."
        )

    X, y = load_data(processed_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\n[split] Train: {len(X_train)}  |  Test: {len(X_test)}")

    models = define_models()
    cv_results = run_cross_validation(models, X_train, y_train)
    fitted_models = fit_final_models(models, X_train, y_train)
    save_artifacts(fitted_models, cv_results, X_test, y_test, metrics_dir)

    print("\n[done]  Training complete. Run evaluate_models.py next.")


if __name__ == "__main__":
    main()
