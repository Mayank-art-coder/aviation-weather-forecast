"""
Run this once to register all existing trained models into MLflow.
After this, all future training runs log automatically.
"""
import pickle
import joblib
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlflow_utils import log_training_run
from src.logger import get_logger

logger = get_logger(__name__)
ROOT = Path(__file__).parent.parent

def register_xgboost_pressure():
    logger.info("Registering XGBoost pressure model...")
    with open(ROOT / "models/xgb_pressure_model_6hr.pkl", "rb") as f:
        model = pickle.load(f)

    params = {
        "model_type":      "XGBoost",
        "target":          "pressure",
        "horizon_hr":      6,
        "n_estimators":    500,
        "learning_rate":   0.05,
        "max_depth":       6,
        "subsample":       0.8,
        "colsample_bytree":0.8,
        "station":         "VABB"
    }
    metrics = {
        "rmse_6hr":        0.752,
        "mae_6hr":         0.593,
        "skill_score_pct": 64.9
    }
    log_training_run(
        model_name="XGBoost_pressure",
        params=params,
        metrics=metrics,
        model=model,
        model_type="xgboost"
    )
    logger.info("XGBoost pressure registered.")

def register_rf_fog():
    logger.info("Registering RF fog classifier...")
    import joblib

    model = joblib.load(ROOT / "models/fog_classifier_rf.pkl")

    params = {
        "model_type":       "RandomForest",
        "target":           "low_vis_binary",
        "threshold_m":      1000,
        "class_imbalance":  "285:1",
        "station":          "VABB"
    }
    metrics = {
        "recall_lowvis":    0.812,
        "precision_lowvis": 0.040,
        "f1_lowvis":        0.076
    }
    log_training_run(
        model_name="RF_fog_classifier",
        params=params,
        metrics=metrics,
        model=model,
        model_type="sklearn"
    )
    logger.info("RF fog classifier registered.")

def register_tft():
    logger.info("Registering TFT ensemble model...")
    import torch

    params = {
        "model_type":    "TFT_XGBoost_ensemble",
        "tft_targets":   "temp,wind_speed,gust,visibility,u_wind,v_wind",
        "xgb_target":    "pressure",
        "horizon_hr":    6,
        "seq_len":       24,
        "hidden_size":   64,
        "num_layers":    1,
        "dropout":       0.6,
        "pos_weight":    50,
        "seed":          42,
        "station":       "VABB"
    }
    metrics = {
        "temp_rmse":       1.121,
        "wind_speed_rmse": 2.257,
        "gust_rmse":       3.075,
        "pressure_rmse":   0.752,
        "visibility_rmse": 509.526,
        "u_wind_rmse":     3.039,
        "v_wind_rmse":     2.666,
        "lowvis_recall":   0.869,
        "lowvis_precision":0.033
    }
    log_training_run(
        model_name="TFT_ensemble",
        params=params,
        metrics=metrics,
        model_type="sklearn"   # no model object — .pt logged separately
    )
    logger.info("TFT ensemble registered.")

if __name__ == "__main__":
    register_xgboost_pressure()
    register_rf_fog()
    register_tft()
    logger.info("All models registered in MLflow.")
    print("\nDone. Run: mlflow ui --port 5000")
    print("Then open: http://localhost:5000")
