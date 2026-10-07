"""Train classifiers on the real TCGA pan-cancer RNA-seq dataset: 801 tumor
samples, 5 cancer types, 20,531 gene expression features.

Unlike the Wisconsin benchmark (30 hand engineered features, 569 samples),
here the feature count vastly exceeds the sample count, so univariate
feature selection (fit on the training fold only, to avoid leakage) stands
in for the domain engineered features of the first analysis.

Run from the repo root: python3 src/pancan_train_models.py
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pancan_common import drop_zero_variance, load_raw

RESULTS_DIR = Path("results/pancan")
K_FEATURES = 500
RANDOM_STATE = 42


def build_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ]),
        "svm_linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def main():
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    X, y = load_raw()
    X = drop_zero_variance(X)
    print(f"After dropping zero variance genes: {X.shape[1]} genes remain")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # A handful of genes that varied globally can still be constant within
    # this particular 640-sample training fold; drop those first so the
    # ANOVA F-test below never divides by a zero within-group variance.
    fold_variance = VarianceThreshold(threshold=0.0)
    X_train = pd.DataFrame(
        fold_variance.fit_transform(X_train), index=X_train.index,
        columns=X_train.columns[fold_variance.get_support()],
    )
    X_test = X_test[X_train.columns]
    print(f"After dropping genes constant within the training fold: {X_train.shape[1]} genes remain")

    # Univariate feature selection, fit on the training fold only.
    selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    selected_genes = X_train.columns[selector.get_support()]
    scores = selector.scores_[selector.get_support()]

    pd.DataFrame({"gene": selected_genes, "anova_f_score": scores}) \
        .sort_values("anova_f_score", ascending=False) \
        .to_csv(RESULTS_DIR / "tables" / "selected_genes_anova.csv", index=False)
    print(f"Selected top {K_FEATURES} genes by ANOVA F-test (fit on training set only)")

    X_train_sel = pd.DataFrame(X_train_sel, index=X_train.index, columns=selected_genes)
    X_test_sel = pd.DataFrame(X_test_sel, index=X_test.index, columns=selected_genes)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1_macro", "roc_auc_ovr"]

    cv_rows = []
    fitted_models = {}
    for name, pipeline in build_models().items():
        cv_results = cross_validate(
            pipeline, X_train_sel, y_train, cv=cv, scoring=scoring, n_jobs=1
        )
        row = {"model": name}
        for metric in scoring:
            key = f"test_{metric}"
            row[f"{metric}_mean"] = cv_results[key].mean()
            row[f"{metric}_std"] = cv_results[key].std()
        cv_rows.append(row)
        print(f"{name}: CV accuracy {row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f}")

        pipeline.fit(X_train_sel, y_train)
        fitted_models[name] = pipeline

    pd.DataFrame(cv_rows).to_csv(RESULTS_DIR / "metrics" / "cv_results.csv", index=False)

    X_test_sel.to_csv(RESULTS_DIR / "metrics" / "X_test.csv")
    y_test.to_csv(RESULTS_DIR / "metrics" / "y_test.csv")
    X_train_sel.to_csv(RESULTS_DIR / "metrics" / "X_train.csv")
    y_train.to_csv(RESULTS_DIR / "metrics" / "y_train.csv")

    import joblib
    for name, pipeline in fitted_models.items():
        joblib.dump(pipeline, RESULTS_DIR / "metrics" / f"model_{name}.joblib")

    with open(RESULTS_DIR / "metrics" / "run_config.json", "w") as f:
        json.dump({
            "k_features": K_FEATURES,
            "random_state": RANDOM_STATE,
            "n_genes_after_variance_filter": int(X.shape[1]),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        }, f, indent=2)

    print(f"\nWrote CV results, fitted models, and train/test splits to {RESULTS_DIR}/metrics/")


if __name__ == "__main__":
    main()
