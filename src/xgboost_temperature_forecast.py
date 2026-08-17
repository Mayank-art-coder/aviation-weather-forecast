"""
src/xgboost_temperature_forecast.py
Standalone temperature specialist — isolated from TFT shared trunk.

Architecture: XGBoost regressor predicting 6-hour temperature CHANGE
(delta_temp_target = T(t+6h) - T(t)) rather than absolute temperature.
Inference reconstructs: T_pred = T(t) + delta_pred.

Causal feature policy: all rolling/lag statistics use temp_lag_1
(NOT current temp) to prevent leakage / algebraic target recovery.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

BASE_DIR = Path(__file__).parent
csv_path = BASE_DIR.parent / "Vabb_Metar_Data" / "vabb_metar_features_v3.csv"

FEATURE_COLS = [
    'temp_lag_1', 'temp_lag_1_roll_mean_3h', 'temp_lag_1_roll_std_3h',
    'solar_phase', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'monsoon_flag', 'sea_breeze_phase', 'sensor_outage_mask',
    'pressure_tendency', 'dewpoint_depression'
]
TARGET = 'delta_temp_target'

print("Loading data...")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Clean NaN/gap rows first (same as TFT pipeline — keeps test set identical)
df = df.dropna(subset=FEATURE_COLS + [TARGET, 'gap_flag']).reset_index(drop=True)
df = df[df['gap_flag'] == 0].reset_index(drop=True)
print(f"Total usable rows: {len(df)}")

split_idx = int(len(df) * 0.8)
X = df[FEATURE_COLS].values
y = df[TARGET].values
current_temp = df['temp'].values
outage = df['sensor_outage_mask'].values

X_train_full, X_test = X[:split_idx], X[split_idx:]
y_train_full, y_test = y[:split_idx], y[split_idx:]
temp_test = current_temp[split_idx:]

# Outage mask applied ONLY to training data — test set stays full/unfiltered
# so RMSE is directly comparable to the TFT's test set
train_mask = outage[:split_idx] == 0
X_train = X_train_full[train_mask]
y_train = y_train_full[train_mask]
print(f"Training rows after outage mask: {len(X_train)} (test set unaffected: {len(X_test)})")

print("Training XGBoost temperature specialist...")
model = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

delta_pred = model.predict(X_test)
temp_pred  = temp_test + delta_pred        # reconstruction
temp_true  = temp_test + y_test

rmse_delta = np.sqrt(mean_squared_error(y_test, delta_pred))
mae_delta  = mean_absolute_error(y_test, delta_pred)
rmse_temp  = np.sqrt(mean_squared_error(temp_true, temp_pred))
mae_temp   = mean_absolute_error(temp_true, temp_pred)

print(f"\n── XGBoost Temperature Specialist Results (6hr horizon) ──")
print(f"Delta target  — MAE: {mae_delta:.3f}  RMSE: {rmse_delta:.3f}")
print(f"Reconstructed absolute temp — MAE: {mae_temp:.3f}  RMSE: {rmse_temp:.3f}")

# Persistence baseline for comparison (T+6h = T now, i.e. delta=0)
rmse_persist = np.sqrt(mean_squared_error(temp_true, temp_test))
print(f"Persistence baseline RMSE: {rmse_persist:.3f}")
print(f"Skill vs persistence: {100*(1 - rmse_temp/rmse_persist):.1f}%")

model_path = BASE_DIR.parent / "models" / "xgb_temperature_model_6hr.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\nSaved: {model_path}")
