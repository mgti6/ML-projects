# Credit Scoring — Give Me Some Credit (Kaggle)

## 🎯 Objective
Predict the probability that a borrower will experience serious financial distress
(payment delinquency of 90+ days) within the next two years, using the Kaggle
"Give Me Some Credit" dataset.

---

## 📊 Dataset
- Source: Kaggle — Give Me Some Credit
- Target variable: `SeriousDlqin2yrs` (binary, imbalanced: 93% / 7%)
- Shape: 150 000 rows × 11 features
- Main missing values:
  - `MonthlyIncome` (~19.8%)
  - `NumberOfDependents` (~2.6%)

---

## 🔍 Exploratory Data Analysis (EDA)
- Target distribution and class imbalance check
- Univariate distributions of continuous and discrete features
- Bivariate analysis: default rate by feature (custom binning for discrete vs.
  continuous variables), Mann-Whitney/Welch t-tests, Cohen's d
- Correlation analysis with the target (mostly weak, linear correlations ≤ 0.13)
- Identification of sentinel error codes (96/98) in delinquency count columns

Notebook: `notebooks/give_me_some_credit.ipynb`

---

## 🧠 Feature Engineering & Preprocessing
Custom `ColumnTransformer` pipeline handling each data quality issue explicitly:
- **Sentinel codes (96/98)** on delinquency counts → replaced with NaN, flagged
  via `SimpleImputer(add_indicator=True)`, then median-imputed
- **`RevolvingUtilizationOfUnsecuredLines`** → winsorized at the 99th percentile
  (learned on train) to cap extreme "normal" outliers
- **`DebtRatio` + `MonthlyIncome`** → custom transformer: cap, `log1p`, missing-income
  flag, and neutralization of `DebtRatio` for rows with unknown income (to avoid
  a spurious second distribution mode caused by a broken denominator)
- Remaining numerical features → median imputation + standard scaling

All fitted exclusively on the train set to avoid leakage.

---

## 🤖 Modeling
| Model | ROC AUC (test) | PR AUC (test) |
|---|---|---|
| Logistic Regression (`class_weight='balanced'`) | 0.860 | 0.389 |
| Logistic Regression (GridSearchCV: C, penalty, solver) | 0.860 | ~0.382 (no significant gain over baseline) |
| HistGradientBoostingClassifier (tuned) | **0.868** | **0.404** |
| Stacking (LR + HGB + RandomForest) | 0.868 | 0.399 (no improvement over HGB alone) |

Methodology:
- Stratified train/test split
- Cross-validated hyperparameter search (`GridSearchCV`, `scoring="average_precision"`)
- Precision/recall trade-off analysis to pick an operating threshold based on
  business cost (missed default vs. wrongly flagged client), rather than the
  default 0.5 cutoff

Notebook: `notebooks/give_me_some_credit.ipynb`

---

## 📈 Results
- Best model: **HistGradientBoostingClassifier**, ROC AUC 0.868 / PR AUC 0.404 —
  in line with top scores on the original Kaggle leaderboard for this dataset.
- Logistic Regression hyperparameter tuning (C, penalty, solver) produced
  negligible gains — performance is capped by the linear model's inability to
  capture feature interactions, not by regularization strength.
- Stacking multiple models did not outperform HGB alone: the weaker linear
  model doesn't provide complementary information over the same 10 features.
- Chosen operating threshold trades recall for precision depending on the
  target use case (e.g. threshold ≈ 1.27 on HGB's decision function →
  precision 0.40 / recall 0.50, vs. a high-recall threshold → precision 0.22 /
  recall 0.75).

---

## 🛠️ Tools & Libraries
- Python
- pandas, numpy, scipy
- scikit-learn (`ColumnTransformer`, custom transformers, `GridSearchCV`,
  `HistGradientBoostingClassifier`, `StackingClassifier`)
- matplotlib, seaborn

---

## 📌 Key Takeaways
- ROC AUC can look deceptively strong on imbalanced data; PR AUC is more
  representative of real-world usefulness on the minority class.
- Precision and recall trade off along a fixed curve — the right operating
  threshold is a business decision (cost of a missed default vs. cost of a
  false alarm), not a modeling one.
- Custom per-feature preprocessing (sentinel flags, winsorizing, log-transform
  with neutralization) matters more here than model choice or hyperparameter
  tuning once a reasonable model family is picked.
- Gains beyond ~0.86-0.87 ROC AUC on this dataset are marginal — consistent
  with historical Kaggle leaderboard results, suggesting the ceiling is set by
  the available features rather than by algorithm choice.

---
