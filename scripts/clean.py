from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT/"raw_data"/"heart_disease_combined.csv"
CLEAN_PATH = PROJECT_ROOT/ 'clean_data'/ 'heart_disease_clened.csv'
target_col = 'target'

def clean_data(df: pd.DataFrame) ->pd.DataFrame:
    print(f"[clean] Starting shape: {df.shape}")

    df['target'] = (df['target'] > 0).astype(int)
    cols_to_drop = ['num_major_vessels']
    df_cleaned = df.drop(columns=[c for c in cols_to_drop if c in df])

    df_cleaned = df_cleaned.dropna(subset = [target_col])
    numeric_cols = [
        "age", "resting_blood_pressure", "cholesterol", "fasting_blood_sugar",
        "rest_ecg", "max_heart_rate", "exercise_induced_angina",
        "oldpeak", "slope", "thalassemia"
    ]
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

    print(f"[clean] Missing values per column:\n{df_cleaned[numeric_cols].isna().sum()}")

    print(f"[clean] Final shape: {df_cleaned.shape}")
    return df_cleaned

if __name__ == "__main__":
    df_raw = pd.read_csv(RAW_PATH, na_values=['-9', '-9.0', '?'])
    df_clean = clean_data(df_raw)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"[clean] Cleaned data saved to {CLEAN_PATH}")