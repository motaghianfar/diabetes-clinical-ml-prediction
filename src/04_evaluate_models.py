from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading test data...")
X_test = pd.read_csv(RESULTS_DIR / "X_test.csv")
y_test = pd.read_csv(RESULTS_DIR / "y_test.csv").squeeze()

model_files = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
    "Support Vector Machine": MODELS_DIR / "support_vector_machine.joblib",
}

metrics_rows = []

plt.figure(figsize=(8, 6))

for model_name, model_path in model_files.items():
    print(f"Evaluating: {model_name}")

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensitivity = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    metrics_rows.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Diabetes Prediction Models")
plt.legend()
plt.tight_layout()

roc_path = FIGURES_DIR / "roc_curves.png"
plt.savefig(roc_path, dpi=300)
plt.close()

metrics_df = pd.DataFrame(metrics_rows)
metrics_df = metrics_df.sort_values(by="roc_auc", ascending=False)

metrics_csv_path = RESULTS_DIR / "model_performance_metrics.csv"
metrics_xlsx_path = RESULTS_DIR / "model_performance_metrics.xlsx"

metrics_df.to_csv(metrics_csv_path, index=False)
metrics_df.to_excel(metrics_xlsx_path, index=False)

print("")
print("Model performance metrics:")
print(metrics_df)

print("")
print(f"Saved metrics CSV to: {metrics_csv_path}")
print(f"Saved metrics Excel table to: {metrics_xlsx_path}")
print(f"Saved ROC curve figure to: {roc_path}")

best_model = metrics_df.iloc[0]["model"]
best_auc = metrics_df.iloc[0]["roc_auc"]

print("")
print(f"Best model by ROC-AUC: {best_model} with ROC-AUC = {best_auc:.3f}")
