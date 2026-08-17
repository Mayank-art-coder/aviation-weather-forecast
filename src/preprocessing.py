import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_metar(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules from Stage 2."""
    initial = len(df)
    df = df[df['wind_dir'].between(0, 360, inclusive='both')]
    df = df[df['wind_speed'] <= 80]
    df = df[df['gust'] <= 100]
    df = df[df['temp'].between(10, 42)]
    df['wind_dir'] = df['wind_dir'].fillna(0)
    df['pressure'] = df['pressure'].ffill(limit=6)
    df = df.drop_duplicates(subset='timestamp', keep='first')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Cleaning: {initial} → {len(df)} rows")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 26 feature engineering steps."""
    # Cyclical encoding
    df['wind_dir_sin'] = np.sin(np.radians(df['wind_dir']))
    df['wind_dir_cos'] = np.cos(np.radians(df['wind_dir']))
    df['hour']         = df['timestamp'].dt.hour
    df['hour_sin']     = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']     = np.cos(2 * np.pi * df['hour'] / 24)
    df['month']        = df['timestamp'].dt.month
    df['month_sin']    = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']    = np.cos(2 * np.pi * df['month'] / 12)

    # U/V wind components
    df['u_wind'] = df['wind_speed'] * np.sin(np.radians(df['wind_dir']))
    df['v_wind'] = df['wind_speed'] * np.cos(np.radians(df['wind_dir']))

    # Derived features
    df['pressure_tendency']  = df['pressure'].diff(periods=6)
    df['temp_trend']         = df['temp'].diff(periods=6)
    df['dewpoint_depression']= df['temp'] - df['dewpoint']
    df['cooling_rate']       = df['temp'].diff(periods=1)

    # Mumbai-specific
    df['monsoon_flag']    = df['timestamp'].dt.month.isin([6,7,8,9]).astype(int)
    df['sea_breeze_phase']= (
        (df['timestamp'].dt.hour >= 3) &
        (df['timestamp'].dt.hour < 14)
    ).astype(int)

    # Gap flag
    df['time_gap'] = df['timestamp'].diff().dt.total_seconds() / 60
    df['gap_flag'] = (df['time_gap'] > 180).astype(int)

    # NEW: visibility velocity + fog persistence
    df['visibility_velocity'] = df['visibility'].diff(periods=1) * 2
    df['dewpoint_depression_velocity'] = df['dewpoint_depression'].diff(periods=1) * 2
    low_vis = (df['visibility'] < 1000).astype(int)
    hours_since_low_vis = (
        low_vis[::-1].groupby(low_vis[::-1].cumsum()).cumcount()[::-1] * 0.5
    )
    df['fog_persistence_memory'] = np.exp(-hours_since_low_vis / 4)

    # NEW: wind KE + shear
    df['wind_ke'] = 0.5 * (df['u_wind']**2 + df['v_wind']**2)
    df['wind_ke_volatility'] = df['wind_ke'].rolling(12).std()
    df['wind_ke_divergence'] = df['wind_ke'].rolling(4).mean() - df['wind_ke'].rolling(48).mean()
    df['u_shear_3h'] = df['u_wind'] - df['u_wind'].shift(6)
    df['v_shear_3h'] = df['v_wind'] - df['v_wind'].shift(6)

    df = df.dropna()
    logger.info(f"Features engineered: {len(df)} rows, {len(df.columns)} columns")
    return df
