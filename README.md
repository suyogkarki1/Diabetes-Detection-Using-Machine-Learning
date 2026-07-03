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

## Model 2 — Without Blood Glucose Reading

### Why a second model?

Blood Glucose Reading is the strongest signal in the dataset, and by clinical definition
it's the value doctors actually use to diagnose diabetes. Using it in a model isn't wrong,
but it raises a fair question: is the model detecting diabetes, or just reading off the
diagnostic value it was trained on?

Model 2 answers a different, more practical question: **can diabetes risk be estimated
from vitals and symptoms alone, without a lab glucose test?** This mirrors a realistic
screening scenario — a nurse checking blood pressure, heart rate, temperature, and SPO2
at a routine visit, before any lab work is ordered.

A feature importance check on Model 1 also confirmed Blood Glucose Reading was **not**
dominating the model's decisions (it ranked outside the top 5 features), which meant it
was a reasonable, well-motivated experiment rather than an attempt to fix a leakage problem.

### What changed from Model 1

- **Removed:** `Blood Glucose Reading` entirely, from both features and cleaning steps
  that reference it directly.
- **Everything else kept identical:** same cleaning rules, same train/test split logic,
  same preprocessing pipeline structure, same CV/tuning/evaluation workflow.
- **Feature engineering added** for both models, using fixed clinical thresholds/formulas
  (safe to compute before the split, since none of them are derived from data statistics):

```text
Age_Group        — bucketed by fixed age ranges (Child/Teen, Adult, Middle Age, Senior)
HR_Category      — bucketed by standard clinical heart-rate bands
Pulse_Pressure   — Systolic - Diastolic (standard clinical formula)
MAP              — Mean Arterial Pressure (standard clinical formula)
Low_SPO2_Flag    — SPO2 < 95 (standard low-oxygen threshold)
Fever_Flag       — Body Temperature > 100.4°F (standard fever threshold)
Symptom_Count    — sum of Sweating and Shivering flags
```

### Model 2 results

Cross-validation (train only, GradientBoosting, imbalance-corrected via sample weighting):

```text
Balanced Accuracy: 0.9728
Recall:            0.9969
Precision:         0.9990
PR-AUC:            0.9999890
```

Final one-shot test evaluation:

```text
Balanced Accuracy: 0.9902
              precision  recall  f1-score  support
Non-Diabetic     0.88     0.98     0.93      59
Diabetic         1.00     1.00     1.00    2986
```

### Model 1 vs. Model 2

| Metric | Model 1 (with glucose) | Model 2 (without glucose) |
|---|---|---|
| CV Balanced Accuracy | 0.9847 | 0.9728 |
| Final Test Balanced Accuracy | 0.9998 | 0.9902 |

### What this shows

Removing the direct glucose reading costs only about **1–2 points of balanced accuracy**,
not a collapse in performance. That's the key finding: the vitals and symptoms alone carry
real, substantial signal about diabetes status — they aren't just noise sitting around one
dominant feature. This makes Model 2 a legitimate, honestly-tested answer to a more realistic
question than Model 1 answers: *"can this be screened for before a lab test, using signals
a nurse can check in five minutes?"*

### Model 2 Persistence

The Model 2 pipeline was saved separately from Model 1, so both remain available for
comparison and reuse:

```text
diabetes_non_diabetes_pipeline_2.pkl
```

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
