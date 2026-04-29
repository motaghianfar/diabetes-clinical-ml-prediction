from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_DIR / "data" / "raw" / "pima_indians_diabetes.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading raw dataset...")
df = pd.read_csv(RAW_PATH)

print("")
print("Raw dataset shape:")
print(df.shape)

# In the Pima diabetes dataset, zero values in these columns are commonly
# interpreted as missing clinical measurements.
zero_as_missing_cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

print("")
print("Zero counts before cleaning:")
zero_counts_before = (df[zero_as_missing_cols] == 0).sum()
print(zero_counts_before)

# Replace impossible zeros with NaN.
df_clean = df.copy()
df_clean[zero_as_missing_cols] = df_clean[zero_as_missing_cols].replace(0, np.nan)

print("")
print("Missing values after replacing impossible zeros with NaN:")
missing_after_zero_replacement = df_clean.isna().sum()
print(missing_after_zero_replacement)

# Fill missing values with the median of each column.
# Median is robust because it is less affected by extreme values than the mean.
median_values = df_clean[zero_as_missing_cols].median()
df_clean[zero_as_missing_cols] = df_clean[zero_as_missing_cols].fillna(median_values)

print("")
print("Median values used for imputation:")
print(median_values)

print("")
print("Missing values after median imputation:")
print(df_clean.isna().sum())

print("")
print("Zero counts after cleaning:")
zero_counts_after = (df_clean[zero_as_missing_cols] == 0).sum()
print(zero_counts_after)

# Save cleaned dataset.
clean_path = PROCESSED_DIR / "pima_indians_diabetes_clean.csv"
df_clean.to_csv(clean_path, index=False)

# Save a small cleaning summary table for the final report.
cleaning_summary = pd.DataFrame({
    "column": zero_as_missing_cols,
    "zero_count_before_cleaning": zero_counts_before.values,
    "median_used_for_imputation": median_values.values,
    "zero_count_after_cleaning": zero_counts_after.values,
})

summary_path = RESULTS_DIR / "cleaning_summary.csv"
cleaning_summary.to_csv(summary_path, index=False)

print("")
print(f"Saved cleaned dataset to: {clean_path}")
print(f"Saved cleaning summary to: {summary_path}")

print("")
print("Cleaning summary:")
print(cleaning_summary)
