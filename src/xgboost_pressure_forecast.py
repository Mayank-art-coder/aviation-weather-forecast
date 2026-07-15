import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score

# df = pd.read_csv('vabb_metar_features_updated.csv')
from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).parent
csv_path = BASE_DIR.parent / "Vabb_Metar_Data" / "vabb_metar_features_updated.csv"
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

FEATURE_COLS = [
    'wind_dir_sin','wind_dir_cos','hour_sin','hour_cos',
    'month_sin','month_cos','u_wind','v_wind',
    'pressure_tendency','temp_trend','dewpoint_depression',
    'cooling_rate','monsoon_flag','sea_breeze_phase',
    'wind_speed','gust','pressure','visibility','temp','dewpoint'
]

TARGETS  = ['pressure']
HORIZONS = {1:2, 2:4, 3:6, 4:8, 5:10, 6:12}

# ── REGRESSION: ALL TARGETS ────────────────────────────
results = []

for target in TARGETS:
    for hour, shift in HORIZONS.items():
        y = df[target].shift(-shift)
        X = df[FEATURE_COLS].copy()
        mask = (df['gap_flag'] == 0) & y.notna()
        X, y = X[mask], y[mask]
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        results.append({'horizon_hr': hour, 'variable': target,
                        'MAE': round(mae,3), 'RMSE': round(rmse,3)})
        # Save raw predictions with timestamps for ensemble
        target_ts = df['timestamp'].shift(-shift)
        target_ts = target_ts[mask]

        pred_df = pd.DataFrame({
          'timestamp': target_ts.iloc[split:].values,
          'actual_pressure': y_test.values,
          'predicted_pressure': preds,
          'horizon_hr': hour
          })
        pred_df.to_csv(f'xgb_pressure_preds_{hour}hr.csv', index=False)
        print(f"{target} | {hour}hr | RMSE: {rmse:.3f} | MAE: {mae:.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv('xgboost_pressure_results.csv', index=False)
print("Saved: xgboost_pressure_results.csv")