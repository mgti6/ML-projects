# Titanic – Supervised Learning Project

## 🎯 Objective
The objective of this project is to predict passenger survival on the Titanic using supervised machine learning models.
The project follows a complete data science workflow, from exploratory data analysis to model training and evaluation.

---

## 📊 Dataset
- Source: Kaggle Titanic Dataset
- Target variable: `survived`
- Initial shape: 891 rows × 15 columns
- Variable types:
  - Categorical variables: 9
  - Numerical variables: 6
- Main missing values:
  - `deck` (~77%)
  - `age` (~20%)

---

## 🔍 Exploratory Data Analysis (EDA)
The EDA focuses on understanding the structure of the data and identifying key drivers of survival.

Main steps:
- Shape and type analysis
- Missing values analysis
- Target distribution analysis
- Univariate analysis (histograms, categorical distributions)
- Bivariate analysis:
  - Relationship between survival and passenger class
  - Impact of gender, age, fare, and family-related variables
- Correlation analysis:
  - Strong correlation between `sibsp` and `parch`
  - Negative correlation between `pclass` and `age`
  - No strong linear correlation between `age` and `survived`

Notebook: `titanic_EDA.ipynb`

---

## 🧠 Feature Engineering & Preprocessing
Key preprocessing steps include:
- Dropping non-informative variables
- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Creating a clean and reproducible preprocessing pipeline using `scikit-learn`

A pipeline-based approach is used to ensure robustness and avoid data leakage.

---

## 🤖 Modeling
Several supervised learning models are trained and compared:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)

Methodology:
- Train / test split
- Pipeline-based training
- Model comparison using accuracy
- Selection of the best-performing model

Notebook: `final_titanic.ipynb`

---

## 📈 Results
- The Random Forest model achieved the best overall performance.
- Survival is mainly driven by:
  - Gender
  - Passenger class
  - Fare
  - Family-related variables

This confirms both domain intuition and insights obtained during EDA.

---

## 🛠️ Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## 📌 Key Takeaways
- Importance of structured EDA before modeling
- Benefit of pipelines for clean ML workflows
- Feature engineering has a significant impact on model performance

---

