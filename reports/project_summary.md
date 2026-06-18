# Project Summary: Biomedical Classification with Machine Learning

## Objective

Build a reproducible, end-to-end supervised learning pipeline to classify breast tumor biopsies as malignant or benign. Compare three standard classifiers and evaluate them with cross-validation and a held-out test set.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Wisconsin Breast Cancer (Diagnostic) |
| **Source** | `sklearn.datasets.load_breast_cancer` |
| **Samples** | 569 |
| **Features** | 30 numeric cytology measurements |
| **Target** | 0 = malignant (212, 37.3%), 1 = benign (357, 62.7%) |

No missing values. No external download required.

---

## Methods

### Preprocessing
- Loaded dataset from scikit-learn into a pandas DataFrame
- Verified no missing values
- Saved unscaled features + target to `data/processed/breast_cancer_processed.csv`
- Feature scaling deferred to training pipelines (prevents data leakage)

### Models

| Model | Key Settings | Scaling |
|---|---|---|
| Logistic Regression | `max_iter=1000`, `solver=lbfgs` | StandardScaler in Pipeline |
| Random Forest | `n_estimators=100` | None (tree-based) |
| SVM | `kernel=rbf`, `probability=True` | StandardScaler in Pipeline |

### Validation
- 80/20 stratified train/test split (`random_state=42`)
- 5-fold stratified cross-validation on training set only
- Final evaluation on held-out test set (114 samples)

---

## Model Comparison

### Cross-Validation (Training Set)

| Model | Accuracy | F1 | ROC/AUC |
|---|---|---|---|
| Logistic Regression | 0.978 +/- 0.010 | 0.983 +/- 0.008 | 0.996 +/- 0.005 |
| Random Forest | 0.963 +/- 0.018 | 0.970 +/- 0.015 | 0.990 +/- 0.008 |
| SVM | 0.969 +/- 0.015 | 0.976 +/- 0.011 | 0.996 +/- 0.005 |

### Test Set

| Model | Accuracy | Precision | Recall | F1 | ROC/AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.983 | 0.983 | 0.983 | 0.983 | 0.995 |
| Random Forest | 0.956 | 0.956 | 0.956 | 0.956 | 0.994 |
| SVM | 0.983 | 0.983 | 0.983 | 0.983 | 0.995 |

**Best performers:** Logistic Regression and SVM tied on the test set (98.3% accuracy, 0.995 AUC). Random Forest was slightly lower (95.6%) but provides Gini-based feature importances for biological interpretation.

**Key observations:**
- All three models achieved strong performance (AUC > 0.99), consistent with the well-separated nature of this dataset
- `worst perimeter`, `worst concave points`, and `worst radius` ranked among the top Random Forest features — consistent with known cytological markers of malignancy
- No signs of severe overfitting; CV and test metrics are closely aligned

---

## Limitations

- Hyperparameters were not tuned (no grid or randomized search)
- Results are specific to this dataset; generalizability to other cohorts is unknown
- Feature importance scores reflect impurity reduction, not causal biological relationships
- No external validation set from an independent study

---

## Next Steps

1. Add `RandomizedSearchCV` for hyperparameter tuning (especially `C` for SVM/LR, `max_depth` for RF)
2. Implement SHAP values for more reliable feature attribution
3. Test on additional biomedical datasets (e.g., gene expression, clinical trial data)
4. Add class-weight tuning or SMOTE if working with more imbalanced datasets
