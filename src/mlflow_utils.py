import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from src.logger import get_logger

logger = get_logger(__name__)

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "aviation-weather-forecast-VABB"

def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking: {TRACKING_URI}")
    logger.info(f"Experiment: {EXPERIMENT_NAME}")

def log_training_run(
    model_name: str,
    params: dict,
    metrics: dict,
    model=None,
    history_df: pd.DataFrame = None,
    shap_df: pd.DataFrame = None,
    model_type: str = "sklearn"
):
    """
    Log one complete training run to MLflow.

    Args:
        model_name:  e.g. "XGBoost_pressure", "LSTM", "TFT_ensemble"
        params:      dict of hyperparameters
        metrics:     dict of evaluation metrics
        model:       trained model object (optional)
        history_df:  training history dataframe (optional)
        shap_df:     SHAP importance dataframe (optional)
        model_type:  "sklearn" or "pytorch"
    """
    setup_mlflow()

    with mlflow.start_run(run_name=model_name):

        # ── LOG PARAMETERS ────────────────────────────
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")

        # ── LOG METRICS ───────────────────────────────
        mlflow.log_metrics(metrics)
        logger.info(f"Logged {len(metrics)} metrics")

        # ── LOG LOSS CURVE PLOT ───────────────────────
        if history_df is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(history_df['train_loss'], label='Train loss')
            axes[0].plot(history_df['val_loss'],   label='Val loss')
            axes[0].set_title(f'{model_name} — Loss curve')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            if 'val_recall' in history_df.columns:
                axes[1].plot(history_df['val_recall'],
                             label='Val recall (low-vis)', color='orange')
                axes[1].axhline(y=0.70, color='red',
                                linestyle='--', label='Target 0.70')
                axes[1].set_title(f'{model_name} — Low-vis recall')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Recall')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = f"/tmp/{model_name}_loss_curve.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            mlflow.log_artifact(plot_path, "plots")
            logger.info("Logged loss curve plot")

        # ── LOG SHAP PLOT ─────────────────────────────
        if shap_df is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            top15 = shap_df.head(15)
            ax.barh(top15['feature'][::-1],
                    top15['mean_shap'][::-1],
                    color='steelblue')
            ax.set_title(f'{model_name} — SHAP Feature Importance')
            ax.set_xlabel('Mean |SHAP| value')
            plt.tight_layout()
            shap_path = f"/tmp/{model_name}_shap.png"
            plt.savefig(shap_path, dpi=150)
            plt.close()
            mlflow.log_artifact(shap_path, "plots")
            logger.info("Logged SHAP plot")

        # ── LOG MODEL ─────────────────────────────────
        if model is not None:

            if model_type == "xgboost":
                  

                mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model"
                )

        elif model_type == "sklearn":
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model"
            )

        elif model_type == "pytorch":
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model"
            )

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        logger.info(f"Logged {model_type} model: {model_name}")
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run complete. ID: {run_id}")

        return run_id
