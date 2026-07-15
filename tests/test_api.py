import sys, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from api.main import app

client = TestClient(app)

def make_observations(n=54):
    base = datetime(2024, 6, 15, 0, 0, 0)
    return [
        {
            "timestamp":  (base + timedelta(minutes=30*i)).strftime('%Y-%m-%d %H:%M:%S'),
            "wind_dir":   270,
            "wind_speed": 10,
            "gust":       15,
            "visibility": 5000,
            "temp":       28,
            "dewpoint":   22,
            "pressure":   1008
        }
        for i in range(n)
    ]

# In tests/test_api.py replace test_health with:
def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] == True
        assert data["station"] == "VABB"

def test_model_info():
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "ensemble" in data
    assert "TFT" in data["ensemble"]
    assert "XGBoost" in data["ensemble"]

def test_predict_returns_6_horizons():
    r = client.post("/predict", json={"observations": make_observations(54)})
    assert r.status_code == 200
    data = r.json()
    assert "horizons" in data
    for hr in range(1, 7):
        assert f"T+{hr}hr" in data["horizons"]

def test_predict_all_variables_in_horizon():
    r = client.post("/predict", json={"observations": make_observations(54)})
    assert r.status_code == 200
    h1 = r.json()["horizons"]["T+1hr"]
    for key in ["temperature_c","wind_speed_kt","wind_dir_deg",
                "gust_kt","pressure_hpa","visibility_m"]:
        assert key in h1

def test_predict_fog_alert_present():
    r = client.post("/predict", json={"observations": make_observations(54)})
    assert r.status_code == 200
    fog = r.json()["fog_alert"]
    assert "low_vis_flag" in fog
    assert "probability" in fog
    assert isinstance(fog["low_vis_flag"], bool)

def test_predict_too_few_observations():
    r = client.post("/predict", json={"observations": make_observations(10)})
    assert r.status_code == 422

def test_predict_invalid_wind_dir():
    obs = make_observations(54)
    obs[0]["wind_dir"] = 400
    r = client.post("/predict", json={"observations": obs})
    assert r.status_code == 422

def test_predict_6hr_endpoint():
    r = client.post("/predict/6hr", json={"observations": make_observations(54)})
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert "horizon_hours" in data
    assert data["horizon_hours"] == 6
