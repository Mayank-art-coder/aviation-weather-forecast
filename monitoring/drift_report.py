"""
monitoring/drift_report.py
Evidently AI — Data drift and prediction drift monitoring.

Compares:
  - Live METAR features vs training data distribution
  - Live predictions vs expected prediction distribution
  - Target drift when actuals are available

Run manually or triggered by Airflow when drift detected.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.logger import get_logger

logger = get_logger(__name__)
ROOT   = Path(__file__).parent.parent

# ── EVIDENTLY IMPORTS ──────────────────────────────────
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
    ColumnSummaryMetric
)

# ── KEY FEATURES TO MONITOR ────────────────────────────
# Subset — most important for aviation forecast quality
MONITOR_FEATURES = [
    'temp', 'dewpoint', 'pressure', 'visibility',
    'wind_speed', 'wind_dir', 'gust',
    'dewpoint_depression', 'pressure_tendency',
    'cooling_rate', 'monsoon_flag'
]

PREDICTION_COLS = [
    'pred_temp', 'pred_wind_speed', 'pred_pressure',
    'pred_visibility', 'pred_fog_prob'
]


# ── LOAD DATA ──────────────────────────────────────────
def load_reference_data(n_rows: int = 5000) -> pd.DataFrame:
    """
    Training data = reference distribution.
    Use last n_rows to keep it manageable.
    """
    path = ROOT / "data/features/vabb_metar_features_updated.csv"
    df   = pd.read_csv(path)
    # Use rows from training period (first 80%)
    split = int(len(df) * 0.8)
    ref   = df.iloc[:split].tail(n_rows)
    logger.info(f"Reference data: {len(ref)} rows from training set")
    return ref


def load_current_data() -> pd.DataFrame:
    """
    Live METAR features = current distribution.
    """
    path = ROOT / "data/processed/latest_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No live features found at {path}. "
            f"Run Airflow pipeline first."
        )
    df = pd.read_csv(path)
    logger.info(f"Current data: {len(df)} rows from live pipeline")
    return df


def load_predictions_log() -> pd.DataFrame:
    """
    Prediction log with actuals — for prediction drift.
    """
    path = ROOT / "data/predictions_log.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Predictions log: {len(df)} rows, "
                f"{df['verified'].sum()} verified")
    return df


# ── REPORT 1: FEATURE DRIFT ────────────────────────────
def run_feature_drift_report() -> dict:
    """
    Compare live METAR feature distributions vs training data.
    Detects if weather patterns have shifted significantly.
    """
    logger.info("Running feature drift report...")

    ref = load_reference_data()
    cur = load_current_data()

    # Keep only columns present in both
    common_cols = [c for c in MONITOR_FEATURES
                   if c in ref.columns and c in cur.columns]

    ref_subset = ref[common_cols].dropna()
    cur_subset = cur[common_cols].dropna()

    report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DataDriftPreset(),
    ])

    report.run(
        reference_data = ref_subset,
        current_data   = cur_subset
    )

    # Save HTML report
    out_dir = ROOT / "monitoring/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')
    report_path = out_dir / f"feature_drift_{timestamp}.html"
    report.save_html(str(report_path))
    logger.info(f"Feature drift report saved: {report_path}")

    # Extract drift summary as dict
    result   = report.as_dict()
    drift_detected = False
    drifted_cols   = []

    try:
        for metric in result.get('metrics', []):
            if metric.get('metric') == 'DatasetDriftMetric':
                drift_detected = metric['result']['dataset_drift']
                share_drifted  = metric['result']['share_of_drifted_columns']
                n_drifted      = metric['result']['number_of_drifted_columns']
                logger.info(
                    f"Dataset drift: {drift_detected} | "
                    f"Drifted columns: {n_drifted} ({share_drifted:.1%})"
                )
    except Exception as e:
        logger.warning(f"Could not parse drift result: {e}")

    return {
        "report_type":     "feature_drift",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "drift_detected":  drift_detected,
        "drifted_columns": drifted_cols,
        "report_path":     str(report_path),
        "reference_rows":  len(ref_subset),
        "current_rows":    len(cur_subset)
    }


# ── REPORT 2: PREDICTION DRIFT ─────────────────────────
def run_prediction_drift_report() -> dict:
    """
    Monitor how prediction distributions change over time.
    Catches model degradation before actuals confirm it.
    """
    logger.info("Running prediction drift report...")

    log_df = load_predictions_log()
    if len(log_df) < 20:
        logger.warning("Not enough predictions yet for drift report (need 20+)")
        return {"report_type": "prediction_drift", "status": "insufficient_data"}

    # Split into reference (older) and current (recent)
    split      = len(log_df) // 2
    ref_preds  = log_df.iloc[:split][PREDICTION_COLS].dropna()
    cur_preds  = log_df.iloc[split:][PREDICTION_COLS].dropna()

    if len(ref_preds) < 5 or len(cur_preds) < 5:
        logger.warning("Insufficient split data for prediction drift")
        return {"report_type": "prediction_drift", "status": "insufficient_data"}

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(
        reference_data = ref_preds,
        current_data   = cur_preds
    )

    out_dir     = ROOT / "monitoring/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')
    report_path = out_dir / f"prediction_drift_{timestamp}.html"
    report.save_html(str(report_path))
    logger.info(f"Prediction drift report saved: {report_path}")

    return {
        "report_type":    "prediction_drift",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "report_path":    str(report_path),
        "reference_rows": len(ref_preds),
        "current_rows":   len(cur_preds)
    }


# ── REPORT 3: TARGET DRIFT (when actuals available) ────
def run_target_drift_report() -> dict:
    """
    Compare predicted vs actual values for verified predictions.
    Most meaningful drift signal — requires real actuals.
    """
    logger.info("Running target drift report...")

    log_df = load_predictions_log()
    verified = log_df[log_df['verified'] == True].copy()

    if len(verified) < 10:
        logger.warning(f"Only {len(verified)} verified predictions — need 10+")
        return {"report_type": "target_drift", "status": "insufficient_data"}

    # Build comparison dataframe
    compare = pd.DataFrame({
        'pred_visibility':  verified['pred_visibility'],
        'pred_temp':        verified['pred_temp'],
        'actual_visibility':verified['actual_visibility'],
        'actual_temp':      verified['actual_temp'],
        'horizon_hr':       verified['horizon_hr']
    }).dropna()

    if len(compare) < 10:
        return {"report_type": "target_drift", "status": "insufficient_data"}

    # Calculate per-horizon RMSE
    rmse_by_horizon = {}
    for hr in range(1, 7):
        subset = compare[compare['horizon_hr'] == hr]
        if len(subset) >= 3:
            rmse_vis  = np.sqrt(((subset['pred_visibility'] -
                                   subset['actual_visibility'])**2).mean())
            rmse_temp = np.sqrt(((subset['pred_temp'] -
                                   subset['actual_temp'])**2).mean())
            rmse_by_horizon[f"T+{hr}hr"] = {
                "visibility_rmse": round(rmse_vis, 1),
                "temp_rmse":       round(rmse_temp, 2),
                "n_samples":       len(subset)
            }

    # Overall RMSE
    overall_vis  = np.sqrt(((compare['pred_visibility'] -
                              compare['actual_visibility'])**2).mean())
    overall_temp = np.sqrt(((compare['pred_temp'] -
                              compare['actual_temp'])**2).mean())

    result = {
        "report_type":     "target_drift",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "n_verified":      len(verified),
        "overall_rmse": {
            "visibility_m": round(float(overall_vis), 1),
            "temp_c":       round(float(overall_temp), 2)
        },
        "by_horizon":      rmse_by_horizon,
        "thresholds": {
            "visibility_alert": 650,
            "temp_alert":       1.5
        },
        "alerts": {
            "visibility_drift": float(overall_vis) > 650,
            "temp_drift":       float(overall_temp) > 1.5
        }
    }

    # Save JSON summary
    out_dir   = ROOT / "monitoring/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')
    json_path = out_dir / f"target_drift_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"Target drift | vis RMSE={overall_vis:.1f}m "
        f"(alert>{650}) | temp RMSE={overall_temp:.2f}°C (alert>{1.5})"
    )
    logger.info(f"Saved: {json_path}")
    return result


# ── RUN ALL REPORTS ────────────────────────────────────
def run_all_reports() -> dict:
    """Run all three monitoring reports and return summary."""
    logger.info("="*50)
    logger.info("Running full Evidently monitoring suite")
    logger.info("="*50)

    results = {}

    try:
        results['feature_drift']    = run_feature_drift_report()
    except Exception as e:
        logger.error(f"Feature drift report failed: {e}")
        results['feature_drift']    = {"status": "error", "message": str(e)}

    try:
        results['prediction_drift'] = run_prediction_drift_report()
    except Exception as e:
        logger.error(f"Prediction drift report failed: {e}")
        results['prediction_drift'] = {"status": "error", "message": str(e)}

    try:
        results['target_drift']     = run_target_drift_report()
    except Exception as e:
        logger.error(f"Target drift report failed: {e}")
        results['target_drift']     = {"status": "error", "message": str(e)}

    # Save combined summary
    out_dir   = ROOT / "monitoring/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')
    summary_path = out_dir / f"monitoring_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Full monitoring summary saved: {summary_path}")
    return results


if __name__ == "__main__":
    results = run_all_reports()
    print("\n── Monitoring Summary ──")
    for report_name, result in results.items():
        status = result.get('drift_detected', result.get('status', 'done'))
        print(f"{report_name:25}: {status}")
    print(f"\nReports saved to: monitoring/reports/")
