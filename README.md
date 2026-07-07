# ML Projects

A collection of machine learning projects covering supervised, unsupervised, and deep learning.

## Structure

| Folder | Description |
|---|---|
| `supervised_learning/` | Regression and classification projects |
| `unsupervised_learning/` | Clustering projects |
| `deep/` | Neural networks built from scratch with NumPy |

## Projects

### Supervised

**Credit Scoring** — Predicting borrower default risk (Kaggle "Give Me Some Credit")  
`supervised_learning/credit_scoring/`
- Custom sklearn preprocessing: sentinel error codes, winsorizing, log-transform with neutralization
- Logistic Regression, HistGradientBoostingClassifier, and stacking compared
- Precision/recall trade-off analysis for business-driven threshold selection
- Best ROC AUC 0.868 / PR AUC 0.404 (HistGradientBoostingClassifier)

**Titanic** — Binary classification predicting passenger survival  
`supervised_learning/titanic/`
- Exploratory data analysis and feature engineering
- Model selection and evaluation

**Linear Regression from Scratch** — Gradient descent implementation without sklearn  
`supervised_learning/linear_regression_from_scratch/`
- Manual implementation of gradient descent
- Comparison with the analytical solution

**California Housing** — End-to-end regression pipeline predicting house prices  
`supervised_learning/california_housing.ipynb`
- Custom sklearn transformers (`ClusterSimilarity`, ratio features, log features)
- Full `ColumnTransformer` pipeline with geographic, numerical and categorical features
- Models tested: Linear Regression, Decision Tree, Random Forest
- Hyperparameter tuning: GridSearchCV and RandomizedSearchCV
- Best RMSE ~42 000 $ (Random Forest, RandomizedSearch)

### Unsupervised

**Customer Segmentation** — K-Means clustering on the Online Retail II dataset  
`unsupervised_learning/customer_clustering/`
- RFM feature engineering (Recency, Frequency, Monetary)
- Optimal k selection with elbow method and silhouette score

### Deep Learning

**Artificial Neuron** — Single neuron built from scratch with NumPy  
`deep/neural_artificial.ipynb`
- Gradient descent, sigmoid activation, decision boundary visualization
