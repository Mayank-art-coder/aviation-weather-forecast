import pandas as pd
import numpy as np

df = pd.read_csv('vabb_metar_cleaned_v2.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# ── CYCLICAL ENCODING ──────────────────────────────
# Wind direction (0-360 circular)
df['wind_dir_sin'] = np.sin(np.radians(df['wind_dir']))
df['wind_dir_cos'] = np.cos(np.radians(df['wind_dir']))

# Hour of day (0-23 circular)
df['hour'] = df['timestamp'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Month (1-12 circular)
df['month'] = df['timestamp'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# ── U/V WIND COMPONENTS (NEW) ──────────────────────
# Encodes BOTH direction and speed as a physical vector
# sin/cos above only captures direction — u/v captures the full wind vector
# Critical for model to understand wind magnitude + direction together
df['u_wind'] = df['wind_speed'] * np.sin(np.radians(df['wind_dir']))  # east-west
df['v_wind'] = df['wind_speed'] * np.cos(np.radians(df['wind_dir']))  # north-south

# ── DERIVED FEATURES ───────────────────────────────
# Pressure tendency (how fast pressure is changing)
df['pressure_tendency'] = df['pressure'].diff(periods=6)  # 6 x 30min = 3 hours

# Temperature trend
df['temp_trend'] = df['temp'].diff(periods=6)

# Dewpoint depression (gap between temp and dewpoint = humidity indicator)
df['dewpoint_depression'] = df['temp'] - df['dewpoint']

# Gust excess (how much gust exceeds sustained wind) not needed yet.
#df['gust_excess'] = df['gust'] - df['wind_speed']

# Cooling rate (NEW) ───────────────────────────────
# Rate of temperature drop per 30-min interval
# Negative value = cooling = fog formation risk rising
# From Paper 1 (Castillo) — key physics feature for visibility prediction
df['cooling_rate'] = df['temp'].diff(periods=1)

# ── VISIBILITY VELOCITY + FOG PERSISTENCE (NEW) ────────────────
df['visibility_velocity'] = df['visibility'].diff(periods=1) * 2
df['dewpoint_depression_velocity'] = df['dewpoint_depression'].diff(periods=1) * 2

low_vis = (df['visibility'] < 1000).astype(int)
hours_since_low_vis = (
    low_vis[::-1].groupby(low_vis[::-1].cumsum()).cumcount()[::-1] * 0.5
)
df['fog_persistence_memory'] = np.exp(-hours_since_low_vis / 4)

# ── WIND KE + SHEAR (NEW) ───────────────────────────────────────
df['wind_ke'] = 0.5 * (df['u_wind']**2 + df['v_wind']**2)
df['wind_ke_volatility'] = df['wind_ke'].rolling(12).std()
df['wind_ke_divergence'] = df['wind_ke'].rolling(4).mean() - df['wind_ke'].rolling(48).mean()
df['u_shear_3h'] = df['u_wind'] - df['u_wind'].shift(6)
df['v_shear_3h'] = df['v_wind'] - df['v_wind'].shift(6)

# ── DELTA TEMPERATURE TARGET (NEW) ──────────────────────────────
df['delta_temp'] = df['temp'].shift(-12) - df['temp']

# ── MONSOON FLAG (NEW) ─────────────────────────────
# Mumbai monsoon: June–September (months 6,7,8,9)
# Without this, model cannot distinguish June from December
# Critical for Mumbai — monsoon regime is meteorologically distinct
df['monsoon_flag'] = df['timestamp'].dt.month.isin([6, 7, 8, 9]).astype(int)

# ── SEA BREEZE PHASE (NEW) ─────────────────────────
# CSMI Airport sits directly on Arabian Sea coast
# Sea breeze onset ~0900 IST = 0330 UTC → hour >= 3
# Sea breeze reversal ~2000 IST = 1430 UTC → hour < 14
# 1 = onshore sea breeze active, 0 = offshore / land breeze
df['sea_breeze_phase'] = (
    (df['timestamp'].dt.hour >= 3) &
    (df['timestamp'].dt.hour < 14)
).astype(int)

# ── GAP FLAG ───────────────────────────────────────
df['time_gap'] = df['timestamp'].diff().dt.total_seconds() / 60
df['gap_flag'] = (df['time_gap'] > 180).astype(int)

# ── CLEAN UP ───────────────────────────────────────
# Drop raw wind_dir degrees (replaced by sin/cos + u/v)
# Keep hour and month as raw too for reference
df = df.dropna()  # removes first 6 rows where diff() creates NaN

print(f"Rows after feature engineering: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")
print(df.head(3))

df.to_csv('vabb_metar_features_final.csv', index=False)
print("Saved: vabb_metar_features_final.csv")

# ── EXPECTED OUTPUT ────────────────────────────────
# Rows:    ~184,093  (6 lost to cooling_rate diff(1) + 6 to pressure/temp diff(6))
# Columns: 30 total
#
# Original 8:  wind_dir, wind_speed, gust, visibility, temp, dewpoint, pressure, timestamp
# Cyclic   6:  wind_dir_sin, wind_dir_cos, hour_sin, hour_cos, month_sin, month_cos
# Raw ref  2:  hour, month
# U/V      2:  u_wind, v_wind                          ← NEW
# Derived  4:  pressure_tendency, temp_trend, dewpoint_depression, gust_excess
# Cooling  1:  cooling_rate                             ← NEW
# Flags    3:  monsoon_flag, sea_breeze_phase, gap_flag ← 2 NEW
# Time     1:  time_gap
# ──────────────────────────────────────────────────
# Total:  27 columns (excl. time_gap if you drop it, 26)
