# Diabetes Prediction Using Health Data

This project predicts whether a person is **Diabetic** or **Non-Diabetic** using health-related features such as blood glucose, blood pressure, heart rate, body temperature, SPO2, sweating, and shivering.

## Project Objective

The main goal is to build a clean machine learning workflow that includes data cleaning, training-only EDA, model comparison, hyperparameter tuning, final evaluation, and model saving for future prediction.

## Dataset Features

The dataset contains the following columns:

- Age
- Gender
- Blood Glucose Reading
- Diastolic Blood Pressure
- Diastolic Blood Pressure Level
- Systolic Blood Pressure
- Heart Rate
- Body Temperature
- SPO2
- Sweating (Y/N)
- Shivering (Y/N)
- Diabetic/NonDiabetic (D/N)

Target encoding:

```text
N = 0 = Non-Diabetic
D = 1 = Diabetic
```

## Workflow

```text
Data Cleaning
↓
Target Encoding
↓
Train-Test Split
↓
Training-Only EDA
↓
Preprocessing Pipeline
↓
Model Comparison using Cross-Validation
↓
Final Model Selection
↓
GridSearchCV Tuning
↓
Final Test Evaluation
↓
Model Saving and Sample Prediction
```

## Data Cleaning

Invalid values were removed, including impossible ages, negative blood glucose readings, SPO2 values above 100, invalid binary values, and exact duplicate rows.

## EDA Approach

To avoid test-set leakage, statistical EDA was performed only on the training data after the train-test split.

The EDA included:

- Target distribution check
- Countplots for categorical features
- Histograms and boxplots for numerical features
- Chi-square test for categorical features
- Cramér’s V for categorical effect size
- Welch’s t-test for numerical features
- Cohen’s d for numerical effect size
- Correlation heatmap

## Model Training

Several models were compared using stratified cross-validation:

- Logistic Regression
- Random Forest
- Gradient Boosting
- AdaBoost
- SVC

Because the dataset was highly imbalanced, balanced metrics were used. Models that support `class_weight` used `class_weight='balanced'`. Gradient Boosting and AdaBoost used balanced `sample_weight` during fitting.

## Final Model

Gradient Boosting was selected as the final model based on cross-validation performance. It was then tuned using GridSearchCV.

Final test performance:

```text
Accuracy: 0.9997
Balanced Accuracy: 0.9998
Confusion Matrix:
[[60, 0],
 [1, 3009]]
```

The model made only one error on the test set.

## Model Persistence

The final trained pipeline was saved as:

```text
diabetes_non_diabetes_pipeline.pkl
```

This file contains both preprocessing steps and the trained model, so it can be loaded later for new predictions without retraining.

## Sample Prediction

A custom sample can be passed into the saved pipeline to predict whether the person is Diabetic or Non-Diabetic. The model returns both the predicted class and prediction probabilities.

## Important Limitation

The dataset is highly imbalanced, and the test set contains only 60 Non-Diabetic samples. Therefore, the strong minority-class performance should be interpreted carefully. More real-world Non-Diabetic samples would be needed to confirm generalization.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Conclusion

This project follows a complete machine learning workflow, from cleaning and training-only EDA to model selection, tuning, final evaluation, and model persistence. The final Gradient Boosting pipeline achieved very high performance and can be reused for future predictions.
