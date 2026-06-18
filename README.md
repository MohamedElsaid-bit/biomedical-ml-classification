# Biomedical Classification with Machine Learning

A supervised machine learning pipeline for binary classification of breast tumor biopsies. Three classifiers — Logistic Regression, Random Forest, and SVM — are trained, cross-validated, and evaluated on the Wisconsin Breast Cancer dataset built into scikit-learn.

> **Key Result:** Best model achieved approximately **98% held-out test accuracy** on the Wisconsin Breast Cancer dataset.

> **Disclaimer:** This is an academic portfolio project. Results are for educational and demonstration purposes only and are not intended for clinical use.

---

## Why This Project

Biomedical classification problems require careful validation: false positives and false negatives both carry real costs. This project demonstrates a reproducible, leakage-free ML workflow — from data loading through model comparison — using techniques common in bioinformatics and scientific data engineering roles.

No external data download is required. The dataset ships with scikit-learn, so the full pipeline runs out of the box.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Wisconsin Breast Cancer (Diagnostic) |
| **Source** | `sklearn.datasets.load_breast_cancer` |
| **Samples** | 569 |
| **Features** | 30 numeric cytology measurements |
| **Target** | Binary — 0 = malignant, 1 = benign |
| **Class balance** | 37.3% malignant / 62.7% benign |

Features describe cell nucleus properties (radius, texture, perimeter, area, smoothness, etc.) derived from digitized fine needle aspirate images.

---

## Repository Structure

```
biomedical-ml-classification/
|
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   |-- raw/              # Optional: place external CSVs here
|   \-- processed/        # breast_cancer_processed.csv (generated)
|
|-- notebooks/
|   \-- 01_model_development.ipynb
|
|-- src/
|   |-- data_preprocessing.py
|   |-- train_models.py
|   \-- evaluate_models.py
|
|-- results/
|   |-- figures/          # Confusion matrices, ROC curves, feature importance
|   \-- metrics/          # cv_results.csv, test_results.csv
|
\-- reports/
    \-- project_summary.md
```

---

## Methods

### 1. Preprocessing (`src/data_preprocessing.py`)
- Load Breast Cancer Wisconsin dataset from scikit-learn
- Convert to pandas DataFrame with a `target` column
- Check for missing values (none in this dataset)
- Save unscaled features to CSV — **no scaling at this stage**

### 2. Training (`src/train_models.py`)
- 80/20 stratified train/test split (`random_state=42`)
- 5-fold stratified cross-validation on the **training set only**
- `StandardScaler` inside sklearn `Pipeline` for Logistic Regression and SVM (fit on training folds only — no leakage)
- Random Forest trained without scaling (tree-based, scale-invariant)
- Save CV metrics and trained models

### 3. Evaluation (`src/evaluate_models.py`)
- Evaluate all models on the held-out test set
- Metrics: accuracy, precision, recall, F1, ROC/AUC
- Generate confusion matrices, ROC comparison, and Random Forest feature importance plots

---

## Models Tested

| Model | Notes |
|---|---|
| Logistic Regression | Interpretable linear baseline; scaled via Pipeline |
| Random Forest | Ensemble method; provides Gini feature importances |
| SVM (RBF kernel) | Strong on high-dimensional data; scaled via Pipeline |

All models use default or simple hyperparameters (no tuning — see Limitations).

---

## How to Run

### 1. Clone and install

```bash
git clone https://github.com/MohamedElsaid-bit/biomedical-ml-classification.git
cd biomedical-ml-classification
pip install -r requirements.txt
```

### 2. Run the pipeline (from repo root)

```bash
python src/data_preprocessing.py
python src/train_models.py
python src/evaluate_models.py
```

### 3. Or explore interactively

```bash
jupyter notebook notebooks/01_model_development.ipynb
```

---

## Example Results

### Cross-Validation (Training Set, 5-fold)

| Model | CV Accuracy | CV F1 | CV ROC/AUC |
|---|---|---|---|
| Logistic Regression | 0.978 +/- 0.010 | 0.983 +/- 0.008 | 0.996 +/- 0.005 |
| Random Forest | 0.963 +/- 0.018 | 0.970 +/- 0.015 | 0.990 +/- 0.008 |
| SVM | 0.969 +/- 0.015 | 0.976 +/- 0.011 | 0.996 +/- 0.005 |

### Test Set Performance

| Model | Accuracy | Precision | Recall | F1 | ROC/AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.983 | 0.983 | 0.983 | 0.983 | 0.995 |
| Random Forest | 0.956 | 0.956 | 0.956 | 0.956 | 0.994 |
| SVM | 0.983 | 0.983 | 0.983 | 0.983 | 0.995 |

Logistic Regression and SVM tied for best test performance. Random Forest was slightly lower but still strong and provides interpretable feature importances.

### Figures

| File | Description |
|---|---|
| `results/figures/confusion_matrix_*.png` | Per-model confusion matrices |
| `results/figures/roc_curves.png` | Overlaid ROC curves with AUC |
| `results/figures/feature_importance_rf.png` | Top 15 Random Forest features |

---

## Skills Demonstrated

| Category | Tools / Concepts |
|---|---|
| Language | Python 3 |
| Data manipulation | pandas, NumPy |
| Machine learning | scikit-learn (Pipeline, cross-validation) |
| Models | Logistic Regression, Random Forest, SVM |
| Validation | Stratified 5-fold CV, train/test split |
| Evaluation | Accuracy, Precision, Recall, F1, ROC/AUC, Confusion Matrix |
| Feature analysis | Random Forest Gini importance |
| Visualization | matplotlib |
| Reproducibility | Fixed random seed, modular scripts, version-pinned requirements |
| Best practices | No data leakage (scaler fit on train only) |

---

## Reproducibility

- **Built-in dataset:** Data comes from `sklearn.datasets.load_breast_cancer` — no external download, API key, or manual file placement required.
- **Runs out of the box:** `pip install -r requirements.txt` followed by the three pipeline scripts is enough to reproduce all results.
- **No data leakage:** `StandardScaler` is never fit on the full dataset. It lives inside sklearn `Pipeline` objects and is fit only on training folds during cross-validation and on the training split for the final model.
- **Pinned dependencies:** Package versions in `requirements.txt` match the environment used to generate the committed example outputs.
- **CI validation:** A GitHub Actions workflow (`.github/workflows/run_pipeline.yml`) runs the full pipeline on every push and pull request using Python 3.11.

---

## Limitations

- **No hyperparameter tuning.** Models use defaults. `RandomizedSearchCV` would likely improve results.
- **Single built-in dataset.** Performance may not generalize to other cohorts or collection protocols.
- **Feature importance is not causation.** Gini importance reflects predictive utility, not biological causality.
- **No SHAP explainability.** SHAP values would provide more reliable feature attribution.
- **Mild class imbalance.** Weighted metrics are used; SMOTE or class weighting could be explored.

---

## Future Improvements

- [ ] Add `RandomizedSearchCV` for hyperparameter optimization
- [ ] Implement SHAP values for model explainability
- [ ] Add SMOTE or class-weight balancing
- [ ] Explore XGBoost or LightGBM as additional baselines
- [ ] Validate on an independent external cohort

---

## Interview Talking Points

1. **Why no scaling in preprocessing?** Scaling must happen after the train/test split. Fitting `StandardScaler` on the full dataset would leak test-set statistics into training. I use sklearn `Pipeline` so the scaler is fit only on training folds during CV and on the full training set for the final model.

2. **Why stratified splits?** A random split could give the test set a different class ratio than training — especially harmful on smaller datasets.

3. **Why cross-validate on training data only?** The test set is held out entirely until final evaluation to prevent leakage into model selection.

4. **Why three different models?** Logistic Regression is an interpretable baseline. Random Forest handles non-linear interactions and gives feature importances. SVM is strong on high-dimensional data. Comparing them shows you understand trade-offs, not just one algorithm.

5. **What do ROC curves tell you?** AUC summarizes performance across all decision thresholds. Near 1.0 means the model consistently ranks positive samples above negatives — useful when class imbalance makes accuracy misleading.

---

## About

Built by [Mohamed Elsaid](https://github.com/MohamedElsaid-bit) as part of a bioinformatics portfolio targeting Computational Biology, Bioinformatics, and Scientific Data Engineering roles.

M.S. Bioinformatics candidate, Johns Hopkins University | B.S. Biochemistry | Former Associate Scientist, Pfizer / Catalent
