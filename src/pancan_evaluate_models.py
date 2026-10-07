"""Evaluate the fitted TCGA pan-cancer models on the held-out test set:
confusion matrix, per-class precision/recall/F1, a PCA projection of the
selected genes, and the Random Forest's top features.

Run from the repo root, after pancan_train_models.py:
    python3 src/pancan_evaluate_models.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

RESULTS_DIR = Path("results/pancan")
MODEL_NAMES = ["logistic_regression", "random_forest", "svm_linear"]
CLASS_ORDER = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]


def main():
    figures_dir = RESULTS_DIR / "figures"
    tables_dir = RESULTS_DIR / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(RESULTS_DIR / "metrics" / "X_test.csv", index_col=0)
    y_test = pd.read_csv(RESULTS_DIR / "metrics" / "y_test.csv", index_col=0)["Class"]
    X_train = pd.read_csv(RESULTS_DIR / "metrics" / "X_train.csv", index_col=0)
    y_train = pd.read_csv(RESULTS_DIR / "metrics" / "y_train.csv", index_col=0)["Class"]

    test_rows = []
    for name in MODEL_NAMES:
        model = joblib.load(RESULTS_DIR / "metrics" / f"model_{name}.joblib")
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True)
        test_rows.append({
            "model": name,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
        })
        pd.DataFrame(report).transpose().to_csv(tables_dir / f"classification_report_{name}.csv")

        cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_ORDER).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name.replace('_', ' ').title()}: test set confusion matrix")
        fig.tight_layout()
        fig.savefig(figures_dir / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)

        print(f"{name}: test accuracy {report['accuracy']:.3f}, macro F1 {report['macro avg']['f1-score']:.3f}")

    pd.DataFrame(test_rows).to_csv(RESULTS_DIR / "metrics" / "test_results.csv", index=False)

    # PCA projection of the 500 selected genes, train + test together, for
    # visualization only (not part of the modeling pipeline).
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls in CLASS_ORDER:
        mask = (y_all == cls).values
        ax.scatter(coords[mask, 0], coords[mask, 1], label=cls, s=14, alpha=0.7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA of the 500 ANOVA-selected genes, all 801 samples")
    ax.legend(title="Tumor type", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "pca_scatter.png", dpi=150)
    plt.close(fig)
    print(f"\nPC1+PC2 explained variance: {sum(pca.explained_variance_ratio_[:2])*100:.1f}%")

    # Random Forest feature importance among the 500 selected genes.
    rf_model = joblib.load(RESULTS_DIR / "metrics" / f"model_random_forest.joblib")
    importances = pd.Series(
        rf_model.named_steps["clf"].feature_importances_, index=X_test.columns
    ).sort_values(ascending=False)
    importances.head(15).to_csv(tables_dir / "feature_importance_rf_top15.csv", header=["importance"])

    fig, ax = plt.subplots(figsize=(6, 5))
    top15 = importances.head(15).iloc[::-1]
    ax.barh(top15.index, top15.values, color="#2a78d6")
    ax.set_xlabel("Random Forest Gini importance")
    ax.set_title("Top 15 genes by importance (anonymized IDs, see README)")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance_rf.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote figures to {figures_dir} and tables to {tables_dir}")


if __name__ == "__main__":
    main()
