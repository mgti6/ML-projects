# California Housing – Supervised Learning Project

## Objective
The objective of this project is to predict median house values in California districts using supervised machine learning models.
The project follows a complete data science workflow, from exploratory data analysis to model fine-tuning and export.

Source: *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron)

---

## Dataset
- Source: California Housing dataset (StatLib repository)
- Target variable: `median_house_value`
- Shape: 20,640 rows × 10 columns
- Variable types:
  - Numerical variables: 9
  - Categorical variable: 1 (`ocean_proximity`)
- Main missing values:
  - `total_bedrooms` (~207 missing values)

---

## Exploratory Data Analysis (EDA)

Main steps:
- Shape and type analysis
- Missing values analysis
- Stratified train/test split based on `median_income` to preserve income distribution
- Geographic visualization of housing prices using scatter plots (latitude/longitude)
- Correlation analysis:
  - `median_income` is the strongest predictor of `median_house_value`
  - `latitude` and `longitude` are also correlated with price
- Feature engineering insights:
  - `rooms_per_house`, `bedrooms_ratio`, `people_per_house` added to improve signal

Notebook: `notebooks/california_housing.ipynb`

---

## Feature Engineering & Preprocessing

A full scikit-learn pipeline is built to handle all preprocessing steps cleanly and without data leakage:

- **Missing values**: imputed with the median using `SimpleImputer`
- **Categorical encoding**: `ocean_proximity` encoded with `OneHotEncoder`
- **Feature scaling**: `StandardScaler` applied to numerical features
- **Custom features**:
  - `rooms_per_house`, `bedrooms_ratio`, `people_per_house` via `FunctionTransformer`
  - `ClusterSimilarity`: custom transformer using KMeans to measure geographic similarity to cluster centroids
- **Target transformation**: log-scaling of `median_house_value` via `TransformedTargetRegressor`

The full preprocessing is assembled with `ColumnTransformer` and `make_pipeline`.

---

## Cross-Validation

To reliably estimate model performance and detect overfitting:

- `cross_val_score` with `cv=10` is used on the training set
- Models evaluated:
  - **Linear Regression**: RMSE ~ 68,000 — underfitting
  - **Decision Tree**: RMSE ~ 0 on train, ~66,000 on validation — overfitting
  - **Random Forest**: RMSE ~ 17,500 on train, ~46,900 on validation — still overfitting but best performer

Cross-validation surfaces the gap between training and validation error, which guides model selection and regularization.

---

## Model Fine-Tuning

Two hyperparameter search strategies are applied to the Random Forest:

- **GridSearchCV**: exhaustive search over a parameter grid
- **RandomizedSearchCV**: random sampling over distributions — faster and often finds better results

Best hyperparameters are extracted from `rnd_search.best_estimator_`.
Feature importances are extracted and ranked to understand which features drive predictions.

---

## Evaluation on the Test Set

The final model is evaluated on the held-out test set:
- RMSE computed with `root_mean_squared_error`
- 95% confidence interval estimated using `scipy.stats`

---

## Model Export with Joblib

The final trained model (full pipeline) is saved to disk using `joblib`:

```python
import joblib
joblib.dump(final_model, "my_california_housing_model.pkl")
```

The saved file includes the entire pipeline (preprocessing + model), so predictions can be made directly on raw data after loading.

---

## Tools & Libraries
- Python
- pandas, numpy, scipy
- scikit-learn
- matplotlib

---

## Key Takeaways
- Stratified sampling is critical when the target distribution is skewed
- Pipelines prevent data leakage and make the workflow reproducible
- Cross-validation is essential to detect overfitting early
- RandomizedSearchCV is a practical alternative to GridSearchCV for large search spaces
- Joblib allows exporting the full pipeline for deployment or later reuse
