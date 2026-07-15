"""
Airflow DAG — Aviation Weather Forecast Pipeline
Orchestrates scraping + preprocessing + forecast via API call.
No ML libraries needed here — API handles all model inference.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/arun/Desktop/aviation-weather-forecast")
sys.path.insert(0, str(PROJECT_ROOT))

API_URL = "http://localhost:8000"

default_args = {
    "owner":            "IMD-Mumbai",
    "depends_on_past":  False,
    "start_date":       datetime(2026, 7, 11),
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id          = "metar_forecast_pipeline",
    description     = "Aviation weather forecast — VABB CSMI Airport",
    schedule        = "*/30 * * * *",
    default_args    = default_args,
    catchup         = False,
    max_active_runs = 1,
    tags            = ["IMD", "aviation", "VABB", "forecast"]
)


# ── TASK 1: CHECK API HEALTH ───────────────────────────
def check_api(**context):
    import requests
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        if r.status_code != 200:
            raise ValueError(f"API unhealthy: {r.status_code}")
        data = r.json()
        if not data.get("models_loaded"):
            raise ValueError("Models not loaded in API")
        print(f"API healthy — models_loaded={data['models_loaded']}")
    except requests.exceptions.ConnectionError:
        raise ValueError(
            f"Cannot connect to API at {API_URL}. "
            f"Start it with: uvicorn api.main:app --port 8000"
        )


# ── TASK 2: SCRAPE METAR ───────────────────────────────
def scrape_metar(**context):
    import pandas as pd
    from src.scrape_live import scrape_recent_metars

    df = scrape_recent_metars(days_back=3)

    out_path = PROJECT_ROOT / "data/processed/latest_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    context['ti'].xcom_push(key='raw_rows', value=len(df))
    print(f"Scraped {len(df)} METAR rows → {out_path}")


# ── TASK 3: VALIDATE DATA ──────────────────────────────
def validate_data(**context):
    import pandas as pd

    path = PROJECT_ROOT / "data/processed/latest_raw.csv"
    df   = pd.read_csv(path)

    issues = []
    if len(df) < 48:
        issues.append(f"Insufficient rows: {len(df)} < 48")
    if df['visibility'].isna().sum() > 10:
        issues.append("Too many missing visibility values")
    if df['pressure'].isna().sum() > 10:
        issues.append("Too many missing pressure values")

    if issues:
        raise ValueError(f"Data validation failed: {issues}")

    print(f"Validation passed — {len(df)} rows | "
          f"{df['timestamp'].min()} to {df['timestamp'].max()}")


# ── TASK 4: PREPROCESS ────────────────────────────────
def preprocess(**context):
    import pandas as pd
    from src.preprocessing import clean_metar, engineer_features

    path = PROJECT_ROOT / "data/processed/latest_raw.csv"
    df   = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    cleaned  = clean_metar(df)
    features = engineer_features(cleaned)

    out_path = PROJECT_ROOT / "data/processed/latest_features.csv"
    features.to_csv(out_path, index=False)
    print(f"Preprocessed: {len(features)} rows, {len(features.columns)} columns")


# ── TASK 5: FORECAST VIA API ───────────────────────────
def run_forecast(**context):
    """
    Calls FastAPI /predict endpoint.
    No torch/sklearn needed here — API handles all ML inference.
    """
    import requests
    import pandas as pd
    import json

    path = PROJECT_ROOT / "data/processed/latest_raw.csv"
    df   = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Use last 54 raw rows as observations
    raw_cols = ['timestamp','wind_dir','wind_speed','gust',
                'visibility','temp','dewpoint','pressure']
    obs = df[raw_cols].tail(54).copy()
    obs['timestamp'] = obs['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Handle NaN values
    obs = obs.ffill().bfill()
    observations = obs.to_dict('records')

    print(f"Sending {len(observations)} observations to API...")

    r = requests.post(
        f"{API_URL}/predict",
        json    = {"observations": observations},
        timeout = 60
    )

    if r.status_code != 200:
        raise ValueError(f"API returned {r.status_code}: {r.text}")

    forecast = r.json()

    # Save forecast JSON for dashboard
    out_path = PROJECT_ROOT / "data/processed/latest_forecast.json"
    with open(out_path, 'w') as f:
        json.dump(forecast, f, indent=2)

    fog_flag = forecast['fog_alert']['low_vis_flag']
    fog_prob = forecast['fog_alert']['probability']
    t1_vis   = forecast['horizons']['T+1hr']['visibility_m']
    t6_vis   = forecast['horizons']['T+6hr']['visibility_m']

    print(f"Forecast saved | fog={fog_flag} prob={fog_prob:.3f} | "
          f"vis T+1={t1_vis}m T+6={t6_vis}m")

    context['ti'].xcom_push(key='fog_flag', value=fog_flag)
    context['ti'].xcom_push(key='fog_prob', value=fog_prob)


# ── TASK 6: VERIFY PAST PREDICTIONS ───────────────────
def verify_past_predictions(**context):
    import pandas as pd
    import numpy as np
    from datetime import datetime, timezone

    log_path = PROJECT_ROOT / "data/predictions_log.csv"
    raw_path = PROJECT_ROOT / "data/processed/latest_raw.csv"

    if not log_path.exists():
        print("No prediction log yet — skipping verification")
        return

    log_df = pd.read_csv(log_path)
    raw_df = pd.read_csv(raw_path)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    log_df['valid_for'] = pd.to_datetime(log_df['valid_for'], format='mixed', utc=True)

    now        = datetime.now(timezone.utc)
    unverified = log_df[
        (log_df['verified'] == False) &
        (log_df['valid_for'] <= pd.Timestamp(now))
    ]

    matched = 0
    for idx, row in unverified.iterrows():
        valid_time = row['valid_for'].replace(tzinfo=None)
        time_diff  = (raw_df['timestamp'] - valid_time).abs()
        closest    = time_diff.idxmin()
        if time_diff[closest].total_seconds() <= 1200:
            actual = raw_df.loc[closest]
            log_df.at[idx, 'actual_temp']        = actual.get('temp')
            log_df.at[idx, 'actual_wind_speed']  = actual.get('wind_speed')
            log_df.at[idx, 'actual_pressure']    = actual.get('pressure')
            log_df.at[idx, 'actual_visibility']  = actual.get('visibility')
            log_df.at[idx, 'verified']           = True
            matched += 1

    log_df.to_csv(log_path, index=False)

    verified = log_df[log_df['verified'] == True].tail(48)
    if len(verified) >= 10:
        rmse_vis  = np.sqrt(((verified['pred_visibility'] -
                               verified['actual_visibility'])**2).mean())
        rmse_temp = np.sqrt(((verified['pred_temp'] -
                               verified['actual_temp'])**2).mean())
        print(f"Rolling RMSE — Vis: {rmse_vis:.1f}m | Temp: {rmse_temp:.2f}°C")
        context['ti'].xcom_push(key='rmse_visibility', value=float(rmse_vis))
        context['ti'].xcom_push(key='rmse_temp',       value=float(rmse_temp))
    else:
        print(f"Not enough verified rows yet ({len(verified)}/10 minimum)")

    print(f"Verified {matched} past predictions")


# ── TASK 7: CHECK DRIFT ────────────────────────────────
def check_drift(**context):
    ti       = context['ti']
    rmse_vis = ti.xcom_pull(task_ids='verify_predictions', key='rmse_visibility') or 0
    rmse_temp= ti.xcom_pull(task_ids='verify_predictions', key='rmse_temp') or 0

    # Thresholds = 25% worse than test RMSE
    VIS_THRESHOLD  = 650    # test RMSE was 509m
    TEMP_THRESHOLD = 1.5    # test RMSE was 1.12°C

    if rmse_vis > VIS_THRESHOLD or rmse_temp > TEMP_THRESHOLD:
        print(f"DRIFT DETECTED — vis={rmse_vis:.1f}m temp={rmse_temp:.2f}°C")
        return "trigger_alert"

    print(f"No drift — vis={rmse_vis:.1f}m temp={rmse_temp:.2f}°C")
    return "no_action"


# ── TASK 8: ALERT ──────────────────────────────────────
def send_alert(**context):
    ti       = context['ti']
    rmse_vis = ti.xcom_pull(task_ids='verify_predictions', key='rmse_visibility') or 0
    rmse_temp= ti.xcom_pull(task_ids='verify_predictions', key='rmse_temp') or 0

    msg = (
        f"[IMD FORECAST ALERT] Model drift detected at VABB\n"
        f"Visibility RMSE: {rmse_vis:.1f}m (threshold: 650m)\n"
        f"Temperature RMSE: {rmse_temp:.2f}°C (threshold: 1.5°C)\n"
        f"Action: Retrain model on recent 30-day data\n"
        f"MLflow: http://localhost:5000"
    )
    print(msg)

    alert_path = PROJECT_ROOT / "logs/drift_alerts.log"
    alert_path.parent.mkdir(exist_ok=True)
    with open(alert_path, 'a') as f:
        f.write(f"\n{datetime.now()} | {msg}\n")


def run_evidently_monitoring(**context):
    """Runs Evidently drift reports — triggered daily or on drift alert."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable,
         "/home/arun/Desktop/aviation-weather-forecast/monitoring/drift_report.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise ValueError(f"Monitoring failed: {result.stderr}")

t10 = PythonOperator(
    task_id         = "evidently_monitoring",
    python_callable = run_evidently_monitoring,
    dag             = dag
)

# Add to flow — runs after drift check regardless of outcome
[t8, t9] >> t10

# ── WIRE TASKS ─────────────────────────────────────────
t1 = PythonOperator(task_id="check_api",           python_callable=check_api,               dag=dag)
t2 = PythonOperator(task_id="scrape_metar",        python_callable=scrape_metar,            dag=dag)
t3 = PythonOperator(task_id="validate_data",       python_callable=validate_data,           dag=dag)
t4 = PythonOperator(task_id="preprocess",          python_callable=preprocess,              dag=dag)
t5 = PythonOperator(task_id="run_forecast",        python_callable=run_forecast,            dag=dag)
t6 = PythonOperator(task_id="verify_predictions",  python_callable=verify_past_predictions, dag=dag)
t7 = BranchPythonOperator(task_id="check_drift",   python_callable=check_drift,             dag=dag)
t8 = PythonOperator(task_id="trigger_alert",       python_callable=send_alert,              dag=dag)
t9 = EmptyOperator(task_id="no_action",            dag=dag)

t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> [t8, t9]
