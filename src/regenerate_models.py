"""
Regenerates scaler_X.pkl and scaler_y.pkl from the dataset.
Run once to fix corrupted pickle files.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ROOT = Path(__file__).parent.parent

FEATURE_COLS = [
    'wind_dir_sin','wind_dir_cos','hour_sin','hour_cos',
    'month_sin','month_cos','u_wind','v_wind',
    'pressure_tendency','temp_trend','dewpoint_depression',
    'cooling_rate','monsoon_flag','sea_breeze_phase',
    'wind_speed','gust','pressure','visibility','temp','dewpoint'
]
REG_TARGETS = ['temp','wind_speed','gust','pressure','visibility','u_wind','v_wind']

print("Loading data...")
df = pd.read_csv(ROOT / "data/features/vabb_metar_features_updated.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"Loaded: {len(df)} rows")

split_idx = int(len(df) * 0.8)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(df[FEATURE_COLS].values[:split_idx])
scaler_y.fit(df[REG_TARGETS].values[:split_idx])

with open(ROOT / "models/scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)
with open(ROOT / "models/scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

print("scaler_X.pkl saved")
print("scaler_y.pkl saved")

# Verify
with open(ROOT / "models/scaler_X.pkl", "rb") as f:
    test = pickle.load(f)
print(f"Verification OK — scaler_X type: {type(test).__name__}")
print(f"scaler_X feature count: {test.n_features_in_}")

# CI mode — skip if models already exist and --ci flag not passed
if __name__ == "__main__" and "ci" not in sys.argv:
    # Already runs full regeneration above
    pass
