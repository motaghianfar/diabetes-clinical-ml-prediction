# Report: Machine Learning Prediction of Diabetes from Clinical Data

## 1. Project overview

This project predicts diabetes status using patient clinical measurements from the Pima Indians Diabetes Dataset. The project belongs to biomedical artificial intelligence and clinical bioinformatics because it uses health-related measurements to build predictive models. The goal is educational: to learn how clinical data can be cleaned, modeled, evaluated, and interpreted using machine-learning methods.

## 2. Dataset

The dataset contains 768 patient records and 9 columns. Eight columns are clinical input features, and one column is the target outcome.

Input features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target:

- Outcome: 0 means no diabetes, 1 means diabetes

## 3. Data cleaning

Although the raw file did not contain explicit missing values, several clinical columns had impossible zero values. These zero values were treated as missing measurements in the following columns:

| column        |   zero_count_before_cleaning |   median_used_for_imputation |   zero_count_after_cleaning |
|:--------------|-----------------------------:|-----------------------------:|----------------------------:|
| Glucose       |                            5 |                        117   |                           0 |
| BloodPressure |                           35 |                         72   |                           0 |
| SkinThickness |                          227 |                         29   |                           0 |
| Insulin       |                          374 |                        125   |                           0 |
| BMI           |                           11 |                         32.3 |                           0 |

Each impossible zero was replaced with a missing value and then filled using the median value of that column. Median imputation was used because the median is less affected by extreme values than the mean.

## 4. Machine-learning workflow

The cleaned dataset was split into training and test sets. The training set was used to train the models, and the test set was used to evaluate how well the models performed on unseen patient records.

Three classification models were trained:

- Logistic Regression
- Random Forest
- Support Vector Machine

Logistic Regression and Support Vector Machine used feature scaling because these models are sensitive to feature magnitude. Random Forest did not require scaling because it is based on decision trees.

## 5. Model evaluation

The models were evaluated using accuracy, precision, recall/sensitivity, specificity, F1-score, ROC-AUC, and confusion matrix values.

| model                  |   accuracy |   precision |   recall_sensitivity |   specificity |   f1_score |   roc_auc |   true_negative |   false_positive |   false_negative |   true_positive |
|:-----------------------|-----------:|------------:|---------------------:|--------------:|-----------:|----------:|----------------:|-----------------:|-----------------:|----------------:|
| Random Forest          |   0.746753 |    0.666667 |             0.555556 |          0.85 |   0.606061 |  0.816019 |              85 |               15 |               24 |              30 |
| Logistic Regression    |   0.707792 |    0.6      |             0.5      |          0.82 |   0.545455 |  0.812963 |              82 |               18 |               27 |              27 |
| Support Vector Machine |   0.74026  |    0.652174 |             0.555556 |          0.84 |   0.6      |  0.796389 |              84 |               16 |               24 |              30 |

The best model by ROC-AUC was **Random Forest**, with ROC-AUC = **0.816** and accuracy = **0.747**.

## 6. ROC curve interpretation

The ROC curve compares model performance across different classification thresholds. A model with a curve closer to the upper-left corner performs better. ROC-AUC summarizes this performance as a single value. A value near 0.5 means random guessing, while a value closer to 1.0 indicates stronger discrimination between diabetes and non-diabetes cases.

The ROC curve figure was saved as:

`figures/roc_curves.png`

## 7. Feature importance

The Random Forest model was used to estimate feature importance because it performed best by ROC-AUC.

Top features:

| feature                  |   importance |
|:-------------------------|-------------:|
| Glucose                  |    0.261296  |
| BMI                      |    0.165522  |
| Age                      |    0.128272  |
| DiabetesPedigreeFunction |    0.119705  |
| Insulin                  |    0.0968906 |

The most important predictor was **Glucose**, followed by BMI and Age. This is clinically reasonable because blood glucose is directly related to diabetes diagnosis and risk.

The feature importance figure was saved as:

`figures/random_forest_feature_importance.png`

## 8. Responsible clinical interpretation

This project is for education and should not be used as a medical diagnosis tool. Real clinical AI systems require larger and more diverse datasets, clinical validation, bias assessment, calibration, privacy review, and expert medical oversight. The model can help demonstrate how machine learning works, but it cannot replace professional healthcare judgment.

## 9. Main outputs

- Raw dataset: `data/raw/pima_indians_diabetes.csv`
- Cleaned dataset: `data/processed/pima_indians_diabetes_clean.csv`
- Notebook: `notebooks/diabetes_prediction_workflow.ipynb`
- ROC curve: `figures/roc_curves.png`
- Feature importance graph: `figures/random_forest_feature_importance.png`
- Model performance table: `results/model_performance_metrics.csv`
- Feature importance table: `results/random_forest_feature_importance.csv`
