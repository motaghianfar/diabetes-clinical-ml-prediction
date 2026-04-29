from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading training feature names...")
X_train = pd.read_csv(RESULTS_DIR / "X_train.csv")

print("Loading best model: Random Forest...")
model = joblib.load(MODELS_DIR / "random_forest.joblib")

feature_names = X_train.columns.tolist()
importance_values = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
}).sort_values(by="importance", ascending=False)

csv_path = RESULTS_DIR / "random_forest_feature_importance.csv"
xlsx_path = RESULTS_DIR / "random_forest_feature_importance.xlsx"
fig_path = FIGURES_DIR / "random_forest_feature_importance.png"

importance_df.to_csv(csv_path, index=False)
importance_df.to_excel(xlsx_path, index=False)

print("")
print("Feature importance table:")
print(importance_df)

plt.figure(figsize=(8, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance for Diabetes Prediction")
plt.tight_layout()
plt.savefig(fig_path, dpi=300)
plt.close()

print("")
print(f"Saved feature importance CSV to: {csv_path}")
print(f"Saved feature importance Excel table to: {xlsx_path}")
print(f"Saved feature importance figure to: {fig_path}")
