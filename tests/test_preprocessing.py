import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import clean_metar, engineer_features

def make_sample_df():
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=20, freq='30min'),
        'wind_dir':   [270]*20,
        'wind_speed': [10]*20,
        'gust':       [15]*20,
        'visibility': [5000]*20,
        'temp':       [28]*20,
        'dewpoint':   [22]*20,
        'pressure':   [1008]*20
    })

def test_clean_drops_bad_wind_dir():
    df = make_sample_df()
    df.loc[0, 'wind_dir'] = 370  # impossible
    cleaned = clean_metar(df)
    assert 370 not in cleaned['wind_dir'].values

def test_clean_drops_high_wind():
    df = make_sample_df()
    df.loc[0, 'wind_speed'] = 90  # sensor error
    cleaned = clean_metar(df)
    assert 90 not in cleaned['wind_speed'].values

def test_features_adds_uv():
    df = make_sample_df()
    df = clean_metar(df)
    df = engineer_features(df)
    assert 'u_wind' in df.columns
    assert 'v_wind' in df.columns

def test_features_adds_monsoon_flag():
    df = make_sample_df()
    df = clean_metar(df)
    df = engineer_features(df)
    assert 'monsoon_flag' in df.columns
    assert df['monsoon_flag'].isin([0,1]).all()

def test_gap_flag_detects_gaps():
    df = make_sample_df()
    # Insert a 5-hour gap
    df.loc[10, 'timestamp'] = pd.Timestamp('2024-01-01 15:00')
    df = clean_metar(df)
    df = engineer_features(df)
    assert df['gap_flag'].sum() >= 1
