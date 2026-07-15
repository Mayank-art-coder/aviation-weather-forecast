"""
src/predict.py
Inference pipeline — multi-horizon 6-hour forecast.
Supports both historical DataFrame input and live METAR scraping.
"""
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timezone
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CFG
from src.logger import get_logger

logger = get_logger(__name__)
ROOT = Path(__file__).parent.parent

# ── STORE PREDICTIONS FOR LATER VERIFICATION ──────────
PREDICTIONS_LOG = ROOT / "data" / "predictions_log.csv"


# ── EXACT ARCHITECTURE FROM tft_forecast.py ───────────
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, d_model, n_heads, n_targets, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.attention  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.gate     = nn.Linear(d_model * 2, d_model)
        self.gate_act = nn.Sigmoid()
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.ffn      = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_reg  = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, n_targets)
        )
        self.fc_cls  = nn.Sequential(
            nn.Linear(d_model, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        proj = self.input_proj(x)
        attn_out, attn_weights = self.attention(proj, proj, proj)
        gate_input  = torch.cat([proj, attn_out], dim=-1)
        gate_values = self.gate_act(self.gate(gate_input))
        gated       = gate_values * attn_out + (1 - gate_values) * proj
        x1   = self.norm1(proj + self.dropout(gated))
        x2   = self.norm2(x1 + self.dropout(self.ffn(x1)))
        last = x2[:, -1, :]
        return self.fc_reg(last), self.fc_cls(last).squeeze(-1), attn_weights


# ── CONSTANTS ─────────────────────────────────────────
FEATURE_COLS = CFG['features']['feature_cols']
REG_TARGETS  = CFG['features']['reg_targets']
SEQ_LEN      = 48      # 24 hours of 30-min data — from tft_forecast.py
HORIZON_ROWS = 12      # 6 hours = 12 rows of 30-min data
D_MODEL      = 128
N_HEADS      = 8
DROPOUT      = 0.3
LOW_VIS_THR  = CFG['training']['low_vis_threshold']

# RF uses fog-specific feature subset — must match fog_classifier_training.py
FOG_FEATURES = [
    'dewpoint_depression', 'temp_trend', 'pressure_tendency',
    'visibility', 'hour_sin', 'hour_cos',
    'month_sin', 'month_cos', 'wind_speed', 'monsoon_flag'
]


# ── MODEL LOADER ───────────────────────────────────────
class ModelLoader:
    def __init__(self):
        self.scaler_X = None
        self.scaler_y = None
        self.tft      = None
        self.xgb      = None
        self.rf       = None
        self._loaded  = False

    def load(self):
        if self._loaded:
            return
        logger.info("Loading all models...")

        with open(CFG['model']['scaler_x_path'], 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(CFG['model']['scaler_y_path'], 'rb') as f:
            self.scaler_y = pickle.load(f)
        with open(CFG['model']['xgb_pressure_path'], 'rb') as f:
            self.xgb = pickle.load(f)

        self.rf = joblib.load(CFG['model']['rf_fog_path'])

        self.tft = TemporalFusionTransformer(
            input_size=len(FEATURE_COLS),
            d_model=D_MODEL, n_heads=N_HEADS,
            n_targets=len(REG_TARGETS), dropout=DROPOUT
        )
        self.tft.load_state_dict(
            torch.load(CFG['model']['tft_path'],
                       map_location=torch.device('cpu'))
        )
        self.tft.eval()
        self._loaded = True
        logger.info("All models loaded.")

loader = ModelLoader()


# ── INPUT VALIDATION ───────────────────────────────────
def validate_metar_input(obs: dict) -> dict:
    required = ['timestamp', 'wind_dir', 'wind_speed', 'gust',
                'visibility', 'temp', 'dewpoint', 'pressure']
    for field in required:
        if field not in obs:
            raise ValueError(f"Missing required field: {field}")
    checks = [
        ('wind_dir',    0,    360,  "degrees"),
        ('wind_speed',  0,    80,   "knots"),
        ('gust',        0,    100,  "knots"),
        ('visibility',  0,    9999, "metres"),
        ('temp',       -10,   50,   "celsius"),
        ('pressure',    900,  1050, "hPa"),
    ]
    for field, lo, hi, unit in checks:
        val = obs[field]
        if not (lo <= val <= hi):
            raise ValueError(f"{field}={val} out of range [{lo},{hi}] {unit}")
    return obs


# ── SINGLE HORIZON PREDICTION (internal) ──────────────
def _predict_single(history_df: pd.DataFrame, horizon_hr: int) -> dict:
    """
    Predict for one specific horizon using window shift.
    horizon_hr: 1 to 6
    Uses Option A — run inference with shifted window.
    No retraining needed.
    """
    shift_rows = horizon_hr * 2          # 30-min data → 2 rows per hour
    needed     = SEQ_LEN + shift_rows    # need extra rows to shift window back

    if len(history_df) < needed:
        # Use available data — pad with earliest rows if needed
        window_df = history_df.tail(SEQ_LEN)
    else:
        # Shift window back by horizon to simulate predicting from that point
        end_idx    = len(history_df) - shift_rows
        window_df  = history_df.iloc[max(0, end_idx - SEQ_LEN): end_idx]

    window        = window_df[FEATURE_COLS].values
    window_scaled = loader.scaler_X.transform(window)
    X_tensor      = torch.tensor(
        window_scaled[np.newaxis, :, :], dtype=torch.float32
    )

    with torch.no_grad():
        pred_reg, _, _ = loader.tft(X_tensor)

    pred_orig = loader.scaler_y.inverse_transform(pred_reg.numpy())[0]

    # XGBoost for pressure
    pressure_pred = float(loader.xgb.predict(window_scaled[-1:, :])[0])

    # u/v → wind direction degrees
    u = float(pred_orig[REG_TARGETS.index('u_wind')])
    v = float(pred_orig[REG_TARGETS.index('v_wind')])
    wind_dir_pred = float((np.degrees(np.arctan2(u, v)) + 360) % 360)

    return {
        "temperature_c": round(float(pred_orig[REG_TARGETS.index('temp')]),       1),
        "wind_speed_kt": round(float(pred_orig[REG_TARGETS.index('wind_speed')]), 1),
        "wind_dir_deg":  round(wind_dir_pred,                                     0),
        "gust_kt":       round(float(pred_orig[REG_TARGETS.index('gust')]),       1),
        "pressure_hpa":  round(pressure_pred,                                     1),
        "visibility_m":  round(float(pred_orig[REG_TARGETS.index('visibility')]), 0),
    }


# ── MULTI-HORIZON PREDICTION (main function) ──────────
def predict_multihorizon(history_df: pd.DataFrame) -> dict:
    """
    Generate forecasts for all 6 hourly horizons simultaneously.

    Input:  DataFrame with at least SEQ_LEN=48 rows (feature-engineered)
            Sorted oldest → newest, 30-min intervals, no gaps

    Output: {
        "station": "VABB",
        "forecast_time": "2024-01-01T06:00:00+00:00",
        "issued_for": "IMD TAF schedule",
        "horizons": {
            "T+1hr": { temperature_c, wind_speed_kt, ... },
            "T+2hr": { ... },
            ...
            "T+6hr": { ... }
        },
        "fog_alert": { low_vis_flag, probability, ... },
        "model_info": { ... }
    }
    """
    loader.load()

    if len(history_df) < SEQ_LEN:
        raise ValueError(
            f"Need at least {SEQ_LEN} rows of history (24 hours). Got {len(history_df)}"
        )

    missing = [c for c in FEATURE_COLS if c not in history_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    now = datetime.now(timezone.utc)

    # ── Fog probability from RF ────────────────────────
    fog_window = history_df[FOG_FEATURES].tail(1)
    fog_prob   = float(loader.rf.predict_proba(fog_window)[0][1])
    fog_alert  = bool(fog_prob >= 0.5)

    # ── Predict all 6 horizons ─────────────────────────
    horizons = {}
    for hr in range(1, 7):
        horizons[f"T+{hr}hr"] = _predict_single(history_df, hr)

    forecast = {
        "station":       CFG['station']['icao'],
        "station_name":  CFG['station']['name'],
        "forecast_time": now.isoformat(),
        "issued_for":    "IMD Aviation TAF — CSMI Airport",
        "horizons":      horizons,
        "fog_alert": {
            "low_vis_flag":  fog_alert,
            "probability":   round(fog_prob, 3),
            "threshold_m":   LOW_VIS_THR,
            "icao_category": "CAT_I_ALERT" if fog_alert else "NORMAL"
        },
        "model_info": {
            "pressure_model":  "XGBoost",
            "other_variables": "TemporalFusionTransformer",
            "fog_classifier":  "RandomForest_SMOTE",
            "horizon_method":  "window_shift_option_a"
        }
    }

    # ── Log predictions for future verification ────────
    _log_prediction(forecast, history_df)

    logger.info(
        f"Multi-horizon forecast generated | "
        f"fog_alert={fog_alert} | "
        f"T+1 vis={horizons['T+1hr']['visibility_m']}m | "
        f"T+6 vis={horizons['T+6hr']['visibility_m']}m"
    )
    return forecast


# ── KEEP SINGLE HORIZON FOR BACKWARD COMPATIBILITY ────
def predict_6hr(history_df: pd.DataFrame) -> dict:
    """Backward-compatible wrapper — returns T+6 only."""
    loader.load()

    if len(history_df) < SEQ_LEN:
        raise ValueError(
            f"Need at least {SEQ_LEN} rows of history (24 hours). Got {len(history_df)}"
        )

    missing = [c for c in FEATURE_COLS if c not in history_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    window        = history_df[FEATURE_COLS].tail(SEQ_LEN).values
    window_scaled = loader.scaler_X.transform(window)
    X_tensor      = torch.tensor(
        window_scaled[np.newaxis, :, :], dtype=torch.float32
    )

    with torch.no_grad():
        pred_reg, pred_cls_logit, attn_weights = loader.tft(X_tensor)

    pred_orig     = loader.scaler_y.inverse_transform(pred_reg.numpy())[0]
    pressure_pred = float(loader.xgb.predict(window_scaled[-1:, :])[0])
    fog_window    = history_df[FOG_FEATURES].tail(1)
    fog_prob      = float(loader.rf.predict_proba(fog_window)[0][1])
    fog_alert     = bool(fog_prob >= 0.5)

    u = float(pred_orig[REG_TARGETS.index('u_wind')])
    v = float(pred_orig[REG_TARGETS.index('v_wind')])
    wind_dir_pred = float((np.degrees(np.arctan2(u, v)) + 360) % 360)

    return {
        "station":       CFG['station']['icao'],
        "forecast_time": datetime.now(timezone.utc).isoformat(),
        "horizon_hours": 6,
        "predictions": {
            "temperature_c": round(float(pred_orig[REG_TARGETS.index('temp')]),       1),
            "wind_speed_kt": round(float(pred_orig[REG_TARGETS.index('wind_speed')]), 1),
            "wind_dir_deg":  round(wind_dir_pred,                                     0),
            "gust_kt":       round(float(pred_orig[REG_TARGETS.index('gust')]),       1),
            "pressure_hpa":  round(pressure_pred,                                     1),
            "visibility_m":  round(float(pred_orig[REG_TARGETS.index('visibility')]), 0),
        },
        "fog_alert": {
            "low_vis_flag":  fog_alert,
            "probability":   round(fog_prob, 3),
            "threshold_m":   LOW_VIS_THR,
            "icao_category": "CAT_I_ALERT" if fog_alert else "NORMAL"
        },
        "model_info": {
            "pressure_model":  "XGBoost",
            "other_variables": "TemporalFusionTransformer",
            "fog_classifier":  "RandomForest_SMOTE"
        }
    }


# ── PREDICTION LOGGER (for Airflow drift detection) ───
def _log_prediction(forecast: dict, history_df: pd.DataFrame):
    """
    Stores predictions to CSV with timestamps.
    Airflow compares these against actual METAR values later.
    This is what feeds the retraining trigger in Phase 7.

    Row format:
    predicted_at | valid_for_T+1 | valid_for_T+6 | temp_T1 | vis_T1 | ... | actual_* (filled later)
    """
    try:
        now     = datetime.now(timezone.utc)
        rows    = []
        for hr in range(1, 7):
            key  = f"T+{hr}hr"
            pred = forecast['horizons'][key]
            rows.append({
                "predicted_at":   now.isoformat(),
                "valid_for": (pd.Timestamp(now) + pd.Timedelta(hours=int(hr))).strftime('%Y-%m-%dT%H:%M:%S+00:00'),
                "horizon_hr":     hr,
                "pred_temp":      pred['temperature_c'],
                "pred_wind_speed":pred['wind_speed_kt'],
                "pred_wind_dir":  pred['wind_dir_deg'],
                "pred_gust":      pred['gust_kt'],
                "pred_pressure":  pred['pressure_hpa'],
                "pred_visibility":pred['visibility_m'],
                "pred_fog_prob":  forecast['fog_alert']['probability'],
                # actual_* columns filled by Airflow when METAR arrives
                "actual_temp":     None,
                "actual_wind_speed": None,
                "actual_pressure": None,
                "actual_visibility": None,
                "verified":        False
            })

        new_df = pd.DataFrame(rows)

        if PREDICTIONS_LOG.exists():
            existing = pd.read_csv(PREDICTIONS_LOG)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_csv(PREDICTIONS_LOG, index=False)

    except Exception as e:
        logger.warning(f"Prediction logging failed (non-critical): {e}")


# ── LIVE PIPELINE — scrape + preprocess + predict ─────
def run_live_forecast() -> dict:
    """
    Full end-to-end live pipeline:
    1. Scrape latest METARs from ogimet
    2. Clean + feature engineer
    3. Combine with historical buffer
    4. Run multi-horizon prediction
    5. Return structured forecast

    Called by Airflow every 30 minutes.
    """
    from src.scrape_live import scrape_recent_metars
    from src.preprocessing import clean_metar, engineer_features

    logger.info("Starting live forecast pipeline...")

    # Step 1 — Scrape (3 days = enough for 48-row window)
    raw_df = scrape_recent_metars(days_back=3)
    logger.info(f"Scraped {len(raw_df)} raw METAR rows")

    # Step 2 — Clean
    cleaned_df = clean_metar(raw_df)
    logger.info(f"After cleaning: {len(cleaned_df)} rows")

    # Step 3 — Feature engineer
    features_df = engineer_features(cleaned_df)
    logger.info(f"After feature engineering: {len(features_df)} rows")

    if len(features_df) < SEQ_LEN:
        raise ValueError(
            f"Insufficient data after preprocessing. "
            f"Need {SEQ_LEN} rows, got {len(features_df)}. "
            f"Check ogimet scrape or data gaps."
        )

    # Step 4 — Predict
    forecast = predict_multihorizon(features_df)
    logger.info("Live forecast complete.")

    return forecast
