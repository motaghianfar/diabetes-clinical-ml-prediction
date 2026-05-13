# Machine Learning Prediction of Diabetes from Clinical Data

## Project topic

This project predicts diabetes using patient clinical measurements from the Pima Indians Diabetes Dataset.

## Objective

The objective is to build and evaluate classification models that predict whether a patient is likely to have diabetes based on clinical variables such as glucose, BMI, blood pressure, insulin, age, and diabetes pedigree function.

## Area

Biomedical AI / Clinical Bioinformatics

## Dataset

Name: Pima Indians Diabetes Dataset  
Type: Clinical  
Source: UCI Machine Learning Repository dataset, accessed through a public CSV copy for reproducible download.

The dataset contains 768 patient records and 9 columns:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

The target variable is Outcome:

- 0 = no diabetes
- 1 = diabetes

## Project workflow

Load data -> Clean data -> Split data -> Train models -> Evaluate models -> Explain model

## Tools used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- SHAP
- JupyterLab

## Models trained

Three classification models were trained:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine

## Best model

The best model by ROC-AUC was:

- Model: Random Forest
- Accuracy: 0.747
- ROC-AUC: 0.816

## Main results

Model performance results are saved in:

- results/model_performance_metrics.csv
- results/model_performance_metrics.xlsx

## Important features

The Random Forest model found the most important clinical predictors to be:

| feature                  |   importance |
|:-------------------------|-------------:|
| Glucose                  |    0.261296  |
| BMI                      |    0.165522  |
| Age                      |    0.128272  |
| DiabetesPedigreeFunction |    0.119705  |
| Insulin                  |    0.0968906 |

The full feature-importance table is saved in:

- results/random_forest_feature_importance.csv
- results/random_forest_feature_importance.xlsx

## Figures

The project includes the following figures:

- figures/roc_curves.png
- figures/random_forest_feature_importance.png

## Repository structure

Project5/
- data/
  - raw/
  - processed/
- figures/
- models/
- notebooks/
- reports/
- results/
- src/
- README.md
- requirements.txt
- environment.yml
- .gitignore

## How to reproduce this project

### 1. Clone the repository

git clone https://github.com/motaghianfar/Project5.git

cd Project5

### 2. Create the Conda environment

conda env create -f environment.yml

conda activate diabetes-ml-py311

Alternative using pip:

python -m pip install -r requirements.txt

### 3. Run the full pipeline

python src/01_download_data.py

python src/02_clean_data.py

python src/03_train_models.py

python src/04_evaluate_models.py

python src/05_feature_importance.py

### 4. Run the notebook

jupyter lab notebooks/diabetes_prediction_workflow.ipynb

## Outputs

This project produces:

- ROC curves
- Accuracy and evaluation table
- Feature importance graph
- Jupyter notebook
- Markdown report
- Reproducible scripts

## Clinical interpretation

This project is for education and should not be used as a medical diagnosis tool. Real clinical AI systems require larger datasets, external validation, fairness assessment, calibration, privacy review, and expert medical supervision.

## Author

GitHub: https://github.com/motaghianfar/
