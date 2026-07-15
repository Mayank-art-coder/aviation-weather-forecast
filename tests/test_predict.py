import sys, pytest
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import validate_metar_input, predict_6hr, FEATURE_COLS, SEQ_LEN

def make_valid_obs():
    return {
        'timestamp': '2024-01-01 00:00',
        'wind_dir': 270, 'wind_speed': 10,
        'gust': 15, 'visibility': 5000,
        'temp': 28, 'dewpoint': 22, 'pressure': 1008
    }

def make_dummy_history():
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(SEQ_LEN, len(FEATURE_COLS)),
        columns=FEATURE_COLS
    )
    df['gap_flag'] = 0
    return df

def test_valid_obs_passes():
    assert validate_metar_input(make_valid_obs()) is not None

def test_missing_field_raises():
    obs = make_valid_obs(); del obs['pressure']
    with pytest.raises(ValueError, match="Missing required field"):
        validate_metar_input(obs)

def test_bad_wind_dir_raises():
    obs = make_valid_obs(); obs['wind_dir'] = 400
    with pytest.raises(ValueError, match="wind_dir"):
        validate_metar_input(obs)

def test_bad_temp_raises():
    obs = make_valid_obs(); obs['temp'] = 60
    with pytest.raises(ValueError, match="temp"):
        validate_metar_input(obs)

def test_predict_returns_correct_keys():
    result = predict_6hr(make_dummy_history())
    assert 'predictions' in result
    assert 'fog_alert'   in result
    assert result['station'] == 'VABB'

def test_predict_all_variables_present():
    preds = predict_6hr(make_dummy_history())['predictions']
    for key in ['temperature_c','wind_speed_kt','wind_dir_deg',
                'gust_kt','pressure_hpa','visibility_m']:
        assert key in preds

def test_predict_fog_alert_structure():
    fog = predict_6hr(make_dummy_history())['fog_alert']
    assert isinstance(fog['low_vis_flag'], bool)
    assert 0.0 <= fog['probability'] <= 1.0

def test_insufficient_history_raises():
    short = pd.DataFrame(
        np.random.randn(10, len(FEATURE_COLS)),
        columns=FEATURE_COLS
    )
    with pytest.raises(ValueError, match="Need at least"):
        predict_6hr(short)
