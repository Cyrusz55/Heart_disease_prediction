from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
from database.db_connection import get_engine

RAW_CSV_PATH = PROJECT_ROOT/"raw_data"/"heart_disease_combined.csv"
TARGET_TABLE = "heart_disease_raw"
RAW_COLUMNS = [

    "age",
    "sex",
    "chest_pains",
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "rest_ecg",
    "max_heart_rate",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "num_major_vessels",
    "thalassemia",
    "target",
]
def ingest_raw_data():
    engine = get_engine()
    df = pd.read_csv(RAW_CSV_PATH, usecols = RAW_COLUMNS, low_memory = False)
    df = df.dropna(subset = ['target'])
    print(f"[Ingest] loaded {len(df)} rows from {RAW_CSV_PATH}")
    print(f"[Ingest] Columns: {list(df.columns)}")

    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {TARGET_TABLE} CASCADE")

    df.to_sql(
        TARGET_TABLE,
        con = engine,
        if_exists = 'replace',
        index = False,
        chunksize = 1000,
        method = 'multi',
    )

    print("[ingest] Raw data loaded into 'geart disease raw' table")

if __name__ == "__main__":
    ingest_raw_data()