"""
api/main.py
FastAPI application for aviation weather forecasting.
Models loaded once at startup — not on every request.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.predict import (
    loader,
    predict_multihorizon,
    predict_6hr,
    validate_metar_input,
    FEATURE_COLS,
    FOG_FEATURES,
    SEQ_LEN
)
from src.preprocessing import engineer_features
from src.logger import get_logger
from api.schemas import (
    METARObservation,
    ForecastResponse,
    HealthResponse,
    BatchMETARRequest
)

logger = get_logger(__name__)


# ── STARTUP: load models once ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting — loading models...")
    loader.load()
    logger.info("Models loaded. API ready.")
    yield
    logger.info("API shutting down.")


# ── APP ────────────────────────────────────────────────
app = FastAPI(
    title       = "Aviation Weather Forecast API",
    description = """
## IMD Mumbai — CSMI Airport Weather Forecasting

AI/ML system forecasting 6 aviation weather parameters
for the next 6 hours at Chhatrapati Shivaji Maharaj International Airport.

### Models
- **TFT** (Temporal Fusion Transformer) — temp, wind, gust, visibility
- **XGBoost** — pressure (selective ensemble)
- **Random Forest + SMOTE** — low visibility fog alert

### Research
Built on 10 years of METAR data (184k observations).
Physics-guided feature engineering validated by SHAP analysis.
    """,
    version     = "1.0.0",
    lifespan    = lifespan
)

# Allow Streamlit dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── HEALTH ENDPOINT ────────────────────────────────────
@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["System"],
    summary        = "API health check"
)
def health():
    """
    Returns API status and model load state.
    Used by Docker health check and monitoring.
    """
    return {
        "status":        "healthy",
        "models_loaded": loader._loaded,
        "station":       "VABB",
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "version":       "1.0.0"
    }


# ── PREDICT ENDPOINT — multi-horizon ──────────────────
@app.post(
    "/predict",
    response_model = ForecastResponse,
    tags           = ["Forecast"],
    summary        = "Generate 6-hour aviation weather forecast"
)
def predict(request: BatchMETARRequest):
    """
    Generate multi-horizon forecast (T+1hr to T+6hr) from METAR history.

    **Input:** List of recent METAR observations (minimum 48 = last 24 hours)

    **Output:** Forecast for each of the next 6 hours + fog alert

    **IMD Schedule:** Issue every 3 hours per TAF validity windows:
    - 0030 UTC → valid 0100–0600
    - 0330 UTC → valid 0400–0900
    - etc.
    """
    try:
        # Validate each observation
        for obs in request.observations:
            validate_metar_input(obs.model_dump())

        if len(request.observations) < SEQ_LEN:
            raise HTTPException(
                status_code = 422,
                detail      = f"Need at least {SEQ_LEN} observations (24 hours). "
                              f"Got {len(request.observations)}."
            )

        # Convert to DataFrame
        df = pd.DataFrame([obs.model_dump() for obs in request.observations])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Feature engineering on incoming raw observations
        df = engineer_features(df)

        if len(df) < SEQ_LEN:
            raise HTTPException(
                status_code = 422,
                detail      = f"After feature engineering, insufficient rows. "
                              f"Check for data gaps in observations."
            )

        # Run multi-horizon forecast
        forecast = predict_multihorizon(df)
        return forecast

    except HTTPException:
        raise                          # re-raise as-is, don't convert to 500
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── PREDICT SIMPLE ENDPOINT — 6hr only ────────────────
@app.post(
    "/predict/6hr",
    tags    = ["Forecast"],
    summary = "Single 6-hour forecast (simplified)"
)
def predict_single_6hr(request: BatchMETARRequest):
    """
    Simplified endpoint returning only T+6 forecast.
    Useful for quick checks and testing.
    """
    try:
        for obs in request.observations:
            validate_metar_input(obs.model_dump())

        if len(request.observations) < SEQ_LEN:
            raise HTTPException(
                status_code = 422,
                detail      = f"Need at least {SEQ_LEN} observations."
            )

        df = pd.DataFrame([obs.model_dump() for obs in request.observations])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = engineer_features(df)

        return predict_6hr(df)

    except HTTPException:
        raise                          # re-raise as-is, don't convert to 500
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── MODEL INFO ENDPOINT ────────────────────────────────
@app.get(
    "/model/info",
    tags    = ["System"],
    summary = "Model architecture and performance info"
)
def model_info():
    """Returns model details, training metrics, and feature info."""
    return {
        "ensemble": {
            "TFT": {
                "targets":     ["temp","wind_speed","gust","visibility","u_wind","v_wind"],
                "architecture":"TemporalFusionTransformer",
                "seq_len":     48,
                "d_model":     128,
                "n_heads":     8,
                "rmse_6hr": {
                    "temp":       1.121,
                    "wind_speed": 2.257,
                    "gust":       3.075,
                    "visibility": 509.5
                }
            },
            "XGBoost": {
                "targets":    ["pressure"],
                "reason":     "Beats TFT on smooth pressure variable",
                "rmse_6hr":   0.752
            },
            "RandomForest": {
                "target":     "low_visibility_alert",
                "threshold_m": 1000,
                "recall":     0.812,
                "precision":  0.040,
                "note":       "High recall prioritised for aviation safety"
            }
        },
        "features": {
            "total":        len(FEATURE_COLS),
            "fog_specific": len(FOG_FEATURES),
            "physics_based":["dewpoint_depression","cooling_rate",
                             "pressure_tendency","temp_trend"],
            "mumbai_specific":["monsoon_flag","sea_breeze_phase"]
        },
        "data": {
            "station":        "VABB (CSMI Mumbai)",
            "training_years": "2014-2024",
            "observations":   184099
        }
    }
