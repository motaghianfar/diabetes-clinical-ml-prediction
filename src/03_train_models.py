from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "pima_indians_diabetes_clean.csv"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

print("")
print("Feature matrix shape:")
print(X.shape)

print("")
print("Target vector shape:")
print(y.shape)

print("")
print("Class distribution:")
print(y.value_counts())

# Stratify keeps the diabetes/non-diabetes ratio similar in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("")
print("Training set shape:")
print(X_train.shape)

print("")
print("Test set shape:")
print(X_test.shape)

print("")
print("Training target distribution:")
print(y_train.value_counts())

print("")
print("Test target distribution:")
print(y_test.value_counts())

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "support_vector_machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(probability=True, random_state=42))
    ])
}

print("")
print("Training models...")

for model_name, model in models.items():
    print(f"Training: {model_name}")
    model.fit(X_train, y_train)

    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    print(f"Saved model to: {model_path}")

# Save the split data so evaluation uses the exact same test patients.
X_train.to_csv(RESULTS_DIR / "X_train.csv", index=False)
X_test.to_csv(RESULTS_DIR / "X_test.csv", index=False)
y_train.to_csv(RESULTS_DIR / "y_train.csv", index=False)
y_test.to_csv(RESULTS_DIR / "y_test.csv", index=False)

print("")
print("Saved train/test split files to results/")

print("")
print("Model training completed successfully.")
