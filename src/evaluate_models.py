"""
evaluate_models.py
------------------
Evaluate trained models on the held-out test set and generate metrics and figures.

Metrics: accuracy, precision, recall, F1-score, ROC/AUC, confusion matrix
Figures: per-model confusion matrices, ROC curve comparison, RF feature importance

Input:  results/metrics/*.pkl, X_test.csv, y_test.csv
Output: results/metrics/test_results.csv
        results/figures/*.png

Usage (from repo root):
    python src/evaluate_models.py
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})


def get_project_root() -> Path:
    """Return the repository root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def load_test_data(metrics_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test set saved during training."""
    X_test = pd.read_csv(metrics_dir / "X_test.csv")
    y_test = pd.read_csv(metrics_dir / "y_test.csv").squeeze()
    print(f"[load]  Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    return X_test, y_test


def load_models(metrics_dir: Path) -> dict:
    """Load pickled models from disk."""
    models = {}
    for name, filename in MODEL_FILES.items():
        path = metrics_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. Run train_models.py first."
            )
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"[load]  {name} <- {filename}")
    return models


def compute_test_metrics(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Calculate classification metrics for each model on the test set."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        row = {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
            "Recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
            "F1": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "ROC_AUC": round(roc_auc, 4),
        }
        rows.append(row)

        print(f"\n-- {name} --")
        print(f"  Accuracy:  {row['Accuracy']}")
        print(f"  Precision: {row['Precision']}")
        print(f"  Recall:    {row['Recall']}")
        print(f"  F1:        {row['F1']}")
        print(f"  ROC/AUC:   {row['ROC_AUC']}")

    return pd.DataFrame(rows)


def plot_confusion_matrices(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path,
) -> None:
    """Save one confusion matrix PNG per model."""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3.5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["malignant (0)", "benign (1)"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name}", fontsize=11, pad=10)
        plt.tight_layout()

        filename = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        fig.savefig(figures_dir / filename, bbox_inches="tight")
        plt.close(fig)
        print(f"[save]  {filename}")


def plot_roc_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path,
) -> None:
    """Plot overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    fig.savefig(figures_dir / "roc_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("[save]  roc_curves.png")


def plot_feature_importance(
    models: dict,
    feature_names: list,
    figures_dir: Path,
) -> None:
    """Plot top-15 Random Forest feature importances."""
    rf = models.get("Random Forest")
    if rf is None or not hasattr(rf, "feature_importances_"):
        print("[skip]  Random Forest not available for feature importance.")
        return

    importances = rf.feature_importances_
    top_n = 15
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_features[::-1], top_importances[::-1], color="#2E86AB")
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)")
    ax.set_title("Top 15 Feature Importances — Random Forest")
    plt.tight_layout()

    fig.savefig(figures_dir / "feature_importance_rf.png", bbox_inches="tight")
    plt.close(fig)
    print("[save]  feature_importance_rf.png")


def main() -> None:
    root = get_project_root()
    metrics_dir = root / METRICS_DIR
    figures_dir = root / FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = load_test_data(metrics_dir)
    models = load_models(metrics_dir)
    feature_names = X_test.columns.tolist()

    print("\n=== Test Set Evaluation ===")
    results = compute_test_metrics(models, X_test, y_test)

    results_path = metrics_dir / "test_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\n[save]  Test results -> {results_path}")

    print("\n=== Generating Figures ===")
    plot_confusion_matrices(models, X_test, y_test, figures_dir)
    plot_roc_curves(models, X_test, y_test, figures_dir)
    plot_feature_importance(models, feature_names, figures_dir)

    print("\n[done]  Evaluation complete.")
    print(f"        Metrics -> {metrics_dir}/")
    print(f"        Figures -> {figures_dir}/")
    print("\nModel comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
