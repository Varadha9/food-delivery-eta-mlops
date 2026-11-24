"""
Training Pipeline for Food Delivery ETA Multi-Component Models.
Trains KPT (Kitchen Prep Time) and VTT (Vehicle Transit Time) models,
logs experiments with MLflow, evaluates performance, and saves model artifacts.
"""
import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn

from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    KPT_MODEL_PATH,
    VTT_MODEL_PATH,
    PIPELINE_METADATA_PATH,
    KPT_FEATURES,
    VTT_FEATURES,
    TARGET_KPT,
    TARGET_VTT,
    TARGET_TOTAL_ETA,
    TARGET_MAX_MAE_MIN,
    MODELS_DIR
)
from src.evaluate import calculate_eta_metrics

def train_pipeline(
    experiment_name: str = "Food_Delivery_ETA_Prediction",
    n_estimators: int = 120,
    max_depth: int = 5,
    learning_rate: float = 0.08
):
    print("=" * 70)
    print("STARTING FOOD DELIVERY ETA TRAINING PIPELINE (MLFLOW TRACKED)")
    print("=" * 70)
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    X_train_kpt = train_df[KPT_FEATURES]
    y_train_kpt = train_df[TARGET_KPT]
    X_test_kpt = test_df[KPT_FEATURES]
    y_test_kpt = test_df[TARGET_KPT]

    X_train_vtt = train_df[VTT_FEATURES]
    y_train_vtt = train_df[TARGET_VTT]
    X_test_vtt = test_df[VTT_FEATURES]
    y_test_vtt = test_df[TARGET_VTT]
    
    y_test_total = test_df[TARGET_TOTAL_ETA]

    # 2. Configure MLflow Experiment
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"eta_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Log Hyperparameters
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "kpt_features_count": len(KPT_FEATURES),
            "vtt_features_count": len(VTT_FEATURES),
            "train_samples": len(train_df),
            "loss_function": "huber"
        }
        mlflow.log_params(params)

        # 3. Build & Train KPT Model Pipeline
        print("\n[Stage 1/2] Training Kitchen Prep Time (KPT) Pipeline...")
        start_kpt = time.time()
        kpt_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                loss="huber",
                random_state=42
            ))
        ])
        kpt_pipeline.fit(X_train_kpt, y_train_kpt)
        kpt_train_time = time.time() - start_kpt
        print(f"KPT Training finished in {kpt_train_time:.2f}s")

        # 4. Build & Train VTT Model Pipeline
        print("\n[Stage 2/2] Training Vehicle Transit Time (VTT) Pipeline...")
        start_vtt = time.time()
        vtt_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                loss="huber",
                random_state=42
            ))
        ])
        vtt_pipeline.fit(X_train_vtt, y_train_vtt)
        vtt_train_time = time.time() - start_vtt
        print(f"VTT Training finished in {vtt_train_time:.2f}s")

        # 5. Evaluate on Holdout Test Set
        print("\nEvaluating Sub-models and Combined Total ETA on Holdout Test Set...")
        pred_kpt = kpt_pipeline.predict(X_test_kpt)
        pred_vtt = vtt_pipeline.predict(X_test_vtt)
        
        # Combined prediction with 0.5 min safety buffer to minimize asymmetric late delivery penalty
        pred_total_eta = np.round(pred_kpt + pred_vtt + 0.5, 2)

        kpt_metrics = calculate_eta_metrics(y_test_kpt, pred_kpt)
        vtt_metrics = calculate_eta_metrics(y_test_vtt, pred_vtt)
        total_metrics = calculate_eta_metrics(y_test_total, pred_total_eta)

        print("\n--- Model Evaluation Results ---")
        print(f"KPT Model   -> MAE: {kpt_metrics['mae_min']:.2f} min | RMSE: {kpt_metrics['rmse_min']:.2f} min | R2: {kpt_metrics['r2_score']:.4f}")
        print(f"VTT Model   -> MAE: {vtt_metrics['mae_min']:.2f} min | RMSE: {vtt_metrics['rmse_min']:.2f} min | R2: {vtt_metrics['r2_score']:.4f}")
        print(f"Total ETA   -> MAE: {total_metrics['mae_min']:.2f} min | RMSE: {total_metrics['rmse_min']:.2f} min | R2: {total_metrics['r2_score']:.4f}")
        print(f"On-Time Delivery Rate: {total_metrics['on_time_rate_percent']:.1f}% | Severe Late Rate (>5m): {total_metrics['severe_late_rate_percent']:.1f}%")

        # Log Metrics to MLflow
        mlflow.log_metrics({
            "kpt_mae": kpt_metrics["mae_min"],
            "kpt_rmse": kpt_metrics["rmse_min"],
            "vtt_mae": vtt_metrics["mae_min"],
            "vtt_rmse": vtt_metrics["rmse_min"],
            "total_eta_mae": total_metrics["mae_min"],
            "total_eta_rmse": total_metrics["rmse_min"],
            "total_eta_r2": total_metrics["r2_score"],
            "total_eta_pinball_loss": total_metrics["asymmetric_pinball_loss"],
            "on_time_rate_percent": total_metrics["on_time_rate_percent"],
            "severe_late_rate_percent": total_metrics["severe_late_rate_percent"]
        })

        # 6. Save Model Artifacts
        print(f"\nSaving model pipelines to {MODELS_DIR}...")
        joblib.dump(kpt_pipeline, KPT_MODEL_PATH)
        joblib.dump(vtt_pipeline, VTT_MODEL_PATH)
        
        metadata = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "kpt_features": KPT_FEATURES,
            "vtt_features": VTT_FEATURES,
            "metrics": {
                "kpt": kpt_metrics,
                "vtt": vtt_metrics,
                "total_eta": total_metrics
            },
            "sla_check": {
                "target_mae_min": TARGET_MAX_MAE_MIN,
                "achieved_mae_min": total_metrics["mae_min"],
                "passed": total_metrics["mae_min"] <= TARGET_MAX_MAE_MIN
            }
        }
        with open(PIPELINE_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        # Log Artifacts to MLflow
        mlflow.log_artifact(str(KPT_MODEL_PATH), artifact_path="models")
        mlflow.log_artifact(str(VTT_MODEL_PATH), artifact_path="models")
        mlflow.log_artifact(str(PIPELINE_METADATA_PATH), artifact_path="metadata")

        print("Model artifacts and metadata saved successfully!")
        
        # 7. Quality Gate Verification
        if total_metrics["mae_min"] <= TARGET_MAX_MAE_MIN:
            print(f"\n[PASS] Model meets production SLA target (MAE: {total_metrics['mae_min']:.2f} <= {TARGET_MAX_MAE_MIN} min). Approved for deployment!")
        else:
            print(f"\n[FAIL] Model did not meet SLA target (MAE: {total_metrics['mae_min']:.2f} > {TARGET_MAX_MAE_MIN} min).")
            
        return metadata

if __name__ == "__main__":
    train_pipeline()
