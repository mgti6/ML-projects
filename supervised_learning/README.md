# Supervised Learning

## Projects

### Credit Scoring — Give Me Some Credit (Kaggle)
`credit_scoring/`
- Custom preprocessing for sentinel error codes, winsorizing, and log-transform with neutralization
- Logistic Regression vs. HistGradientBoostingClassifier vs. stacking
- Precision/recall trade-off analysis for threshold selection
- Best: HGB, ROC AUC 0.868 / PR AUC 0.404

### Titanic — Binary Classification
`titanic/`
- Exploratory data analysis and feature engineering
- Model selection and evaluation

### Linear Regression from Scratch
`linear_regression_from_scratch/`
- Manual gradient descent implementation
- Comparison with the analytical solution

### California Housing — Regression Pipeline
`california_housing.ipynb`

End-to-end sklearn pipeline predicting California house prices (dataset: 20 640 districts).

**Feature engineering**
- Ratio features: bedrooms/rooms, rooms/household, people/household
- Log transform on skewed variables (population, income, etc.)
- Geographic clustering with KMeans + RBF kernel (10 clusters)
- One-hot encoding for `ocean_proximity`

**Models tested**

| Model | CV RMSE |
|---|---|
| Linear Regression | ~69 800 $ |
| Decision Tree | ~66 400 $ (overfitting) |
| Random Forest | ~46 900 $ |
| Random Forest (tuned) | ~42 100 $ |

**Hyperparameter tuning**
- GridSearchCV: best params `n_clusters=15`, `max_features=6`
- RandomizedSearchCV: best params `n_clusters=45`, `max_features=9` → RMSE **42 107 $**
