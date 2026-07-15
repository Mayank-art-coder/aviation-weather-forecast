import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent

def load_config(path: str = None) -> dict:
    config_path = path or ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Resolve all paths relative to project root
    cfg['data']['features_path'] = ROOT / cfg['data']['features_path']
    cfg['model']['tft_path']          = ROOT / cfg['model']['tft_path']
    cfg['model']['xgb_pressure_path'] = ROOT / cfg['model']['xgb_pressure_path']
    cfg['model']['rf_fog_path']       = ROOT / cfg['model']['rf_fog_path']
    cfg['model']['scaler_x_path']     = ROOT / cfg['model']['scaler_x_path']
    cfg['model']['scaler_y_path']     = ROOT / cfg['model']['scaler_y_path']
    return cfg

CFG = load_config()
