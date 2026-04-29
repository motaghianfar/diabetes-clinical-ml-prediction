from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Public CSV copy of the Pima Indians Diabetes Dataset.
# Original dataset: UCI Machine Learning Repository / Pima Indians Diabetes Dataset.
DATA_URL = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"

print("Downloading Pima Indians Diabetes Dataset CSV...")

df = pd.read_csv(DATA_URL)

output_path = RAW_DIR / "pima_indians_diabetes.csv"
df.to_csv(output_path, index=False)

print(f"Saved dataset to: {output_path}")

print("")
print("Dataset shape:")
print(df.shape)

print("")
print("Columns:")
print(df.columns.tolist())

print("")
print("First 5 rows:")
print(df.head())

print("")
print("Missing values per column:")
print(df.isna().sum())

print("")
print("Target distribution:")
print(df["Outcome"].value_counts())
