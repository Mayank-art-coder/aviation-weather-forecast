import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import joblib


#--load data
BASE_DIR = Path(__file__).parent
csv_path = BASE_DIR.parent / "data" / "features" / "vabb_metar_features_updated.csv"
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Fog-specific features only
FOG_FEATURES = [
    'dewpoint_depression',   # most important — temp close to dewpoint = fog
    'temp_trend',            # cooling → fog formation
    'pressure_tendency',     # falling pressure = weather change
    'visibility',            # current visibility (leading indicator)
    'hour_sin', 'hour_cos',  # fog mostly at night/early morning
    'month_sin', 'month_cos',# fog mostly Nov-Feb
    'wind_speed',            # calm wind = fog more likely
    'monsoon_flag'           # monsoon vs dry season
]

# Target: will visibility drop below 1000m in next 6 hours?
SHIFT = 12  # 6 hours ahead

y = df['visibility'].shift(-SHIFT)
X = df[FOG_FEATURES].copy()
mask = (df['gap_flag'] == 0) & y.notna()
X, y = X[mask], y[mask]

# Binary target
y_binary = (y < 1000).astype(int)

print(f"Total fog events: {y_binary.sum()} / {len(y_binary)}")
print(f"Imbalance ratio: {(y_binary==0).sum() / (y_binary==1).sum():.1f}:1")

# Temporal split
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y_binary.iloc[:split], y_binary.iloc[split:]

# Random Forest with balanced class weight
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# # Calibrate probabilities
# calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv='prefit')
# calibrated_rf.fit(X_test, y_test)

# Evaluate at multiple thresholds
# probs = calibrated_rf.predict_proba(X_test)[:, 1]
probs = rf.predict_proba(X_test)[:, 1]

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE — Normal: {(y_train_res==0).sum()}  Fog: {(y_train_res==1).sum()}")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_res, y_train_res)

# Direct probabilities — no calibration
probs = rf.predict_proba(X_test)[:, 1]

print(f"\n{'Threshold':>10}  {'Recall':>8}  {'Precision':>10}  {'F1':>6}")
best_f1, best_thresh = 0, 0.3
for thresh in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]:
    pred_t = (probs >= thresh).astype(int)
    rec  = recall_score(y_test, pred_t, zero_division=0)
    prec = precision_score(y_test, pred_t, zero_division=0)
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    flag = " ← best F1" if f1 > best_f1 else ""
    print(f"{thresh:>10.2f}  {rec:>8.3f}  {prec:>10.3f}  {f1:>6.3f}{flag}")
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

best_preds = (probs >= best_thresh).astype(int)
print(f"\nBest threshold: {best_thresh}")
print(classification_report(y_test, best_preds,
      target_names=['Normal vis', 'Low vis (<1000m)']))
print("Confusion matrix:")
print(confusion_matrix(y_test, best_preds))


joblib.dump(rf, BASE_DIR.parent / "models" / "fog_classifier_rf.pkl")
print("Saved: fog_classifier_rf.pkl")

# Save fog predictions with timestamps for ensemble
fog_df = pd.DataFrame({
    'timestamp': df['timestamp'][mask].iloc[split:].values,
    'actual_fog': y_test.values,
    'fog_probability': probs,
    'fog_alert_0.5': (probs >= 0.50).astype(int),
    'fog_alert_0.1': (probs >= 0.10).astype(int)
})
fog_df.to_csv('fog_predictions.csv', index=False)
print("Saved: fog_predictions.csv")

# # ======================================================
# # DEBUG SECTION
# # ======================================================

# print("\n=== TEST SET DISTRIBUTION ===")
# print(y_test.value_counts())

# print("\n=== PROBABILITY DISTRIBUTION ===")
# print(pd.Series(probs).describe())

# print("\n=== PROBABILITY CHECKS ===")
# print("Max probability :", probs.max())
# print("Prob > 0.10     :", (probs > 0.10).sum())
# print("Prob > 0.05     :", (probs > 0.05).sum())
# print("Prob > 0.01     :", (probs > 0.01).sum())
# print("Prob > 0.001    :", (probs > 0.001).sum())

# print("\nTop 20 probabilities:")
# print(np.sort(probs)[-20:])

# # ======================================================

# print(f"\n{'Threshold':>10}  {'Recall':>8}  {'Precision':>10}  {'F1':>6}")
# best_f1, best_thresh = 0, 0.3
# for thresh in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]:
#     pred_t = (probs >= thresh).astype(int)
#     rec  = recall_score(y_test, pred_t, zero_division=0)
#     prec = precision_score(y_test, pred_t, zero_division=0)
#     f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
#     flag = " ← best F1" if f1 > best_f1 else ""
#     print(f"{thresh:>10.2f}  {rec:>8.3f}  {prec:>10.3f}  {f1:>6.3f}{flag}")
#     if f1 > best_f1:
#         best_f1, best_thresh = f1, thresh

# best_preds = (probs >= best_thresh).astype(int)
# print(f"\nBest threshold: {best_thresh}")
# print(classification_report(y_test, best_preds,
#       target_names=['Normal vis', 'Low vis (<1000m)']))
# print("Confusion matrix:")
# print(confusion_matrix(y_test, best_preds))

# # Feature importance
# import pandas as pd
# fi = pd.DataFrame({
#     'feature': FOG_FEATURES,
#     'importance': rf.feature_importances_
# }).sort_values('importance', ascending=False)
# print("\nFeature importance:")
# print(fi.to_string(index=False))
