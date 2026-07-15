import pandas as pd
import numpy as np

# ── LOAD ALL THREE ──────────────────────────────────────
tft_preds = pd.read_csv('tft_predictions_with_timestamps.csv')
xgb_preds = pd.read_csv('xgb_pressure_preds_6hr.csv')
fog_preds = pd.read_csv('fog_predictions.csv')

# ── FIX TIMESTAMPS ─────────────────────────────────────
tft_preds['timestamp'] = pd.to_datetime(tft_preds['timestamp'])
xgb_preds['timestamp'] = pd.to_datetime(xgb_preds['timestamp'])
fog_preds['timestamp'] = pd.to_datetime(fog_preds['timestamp'])

# ── MERGE TFT + XGB PRESSURE ───────────────────────────
merged = tft_preds.merge(
    xgb_preds[['timestamp', 'predicted_pressure']],
    on='timestamp',
    how='inner',
    suffixes=('_tft', '_xgb')
)

# Replace TFT pressure with XGBoost pressure (XGB wins here)
merged['predicted_pressure'] = merged['predicted_pressure_xgb']
merged = merged.drop(columns=['predicted_pressure_tft',
                               'predicted_pressure_xgb'])

# ── MERGE FOG CLASSIFIER ───────────────────────────────
merged = merged.merge(
    fog_preds[['timestamp', 'fog_probability',
               'fog_alert_0.5', 'fog_alert_0.1']],
    on='timestamp',
    how='left'   # left join — fog predictions may have fewer rows
)

# Fill missing fog probability with 0 (no alert)
merged['fog_probability'] = merged['fog_probability'].fillna(0)
merged['fog_alert_0.5']   = merged['fog_alert_0.5'].fillna(0).astype(int)
merged['fog_alert_0.1']   = merged['fog_alert_0.1'].fillna(0).astype(int)

# ── SAVE ───────────────────────────────────────────────
merged.to_csv('final_forecast.csv', index=False)

print(f"TFT rows:      {len(tft_preds)}")
print(f"XGBoost rows:  {len(xgb_preds)}")
print(f"Fog rows:      {len(fog_preds)}")
print(f"Final matched: {len(merged)}")
print(f"\nColumns: {merged.columns.tolist()}")
print(f"\nSample:")
print(merged[['timestamp','predicted_temp','predicted_pressure',
              'predicted_visibility','fog_probability',
              'fog_alert_0.1']].head(5))