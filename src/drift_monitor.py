"""
Production Monitoring Engine for Food Delivery ETA ML System.
Detects:
1. Data Drift: Two-sample Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI).
2. Concept Drift: Ground truth tracking and percentage degradation in MAE/RMSE.
Generates automated summary metrics and alert triggers.
"""
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

from src.config import (
    TRAIN_DATA_PATH,
    PRODUCTION_STREAM_PATH,
    KPT_MODEL_PATH,
    VTT_MODEL_PATH,
    REPORTS_DIR,
    KPT_FEATURES,
    VTT_FEATURES,
    ALL_FEATURES,
    TARGET_TOTAL_ETA,
    KS_ALPHA,
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    CONCEPT_DRIFT_MAE_DEGRADATION
)
from src.evaluate import calculate_eta_metrics

def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between reference and current distributions.
    PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Determine quantile bins based on reference
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_edges = np.unique(bin_edges)
    
    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    # Convert to proportions with smoothing to avoid log(0)
    ref_props = (ref_counts + 1e-4) / (len(ref_clean) + 1e-4 * len(ref_counts))
    curr_props = (curr_counts + 1e-4) / (len(curr_clean) + 1e-4 * len(curr_counts))

    psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    return float(psi)

def detect_data_drift(train_df: pd.DataFrame, prod_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates KS-Test and PSI for all input features.
    """
    drift_results = {}
    drift_detected_count = 0

    for col in ALL_FEATURES:
        ref_vals = train_df[col].values
        curr_vals = prod_df[col].values

        # 1. Two-sample Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(ref_vals, curr_vals)
        ks_drift = bool(ks_pval < KS_ALPHA)

        # 2. Population Stability Index
        psi_val = calculate_psi(ref_vals, curr_vals)
        if psi_val >= PSI_SIGNIFICANT:
            psi_status = "SIGNIFICANT_DRIFT"
        elif psi_val >= PSI_MODERATE:
            psi_status = "MODERATE_DRIFT"
        else:
            psi_status = "STABLE"

        is_drift = ks_drift or (psi_val >= PSI_MODERATE)
        if is_drift:
            drift_detected_count += 1

        drift_results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": float(ks_pval),
            "ks_drift_detected": ks_drift,
            "psi_value": round(float(psi_val), 4),
            "psi_status": psi_status,
            "drift_detected": is_drift
        }

    return {
        "features": drift_results,
        "total_features": len(ALL_FEATURES),
        "drifted_features_count": drift_detected_count,
        "overall_data_drift": drift_detected_count >= 2
    }

def detect_concept_drift(
    train_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    kpt_model,
    vtt_model
) -> Dict[str, Any]:
    """
    Evaluates concept drift by assessing performance degradation on the production stream
    compared to the baseline.
    """
    # Baseline test predictions
    y_baseline = train_df[TARGET_TOTAL_ETA].values
    kpt_pred_base = kpt_model.predict(train_df[KPT_FEATURES])
    vtt_pred_base = vtt_model.predict(train_df[VTT_FEATURES])
    pred_baseline = kpt_pred_base + vtt_pred_base + 0.5
    base_metrics = calculate_eta_metrics(y_baseline, pred_baseline)

    # Production stream predictions
    y_prod = prod_df[TARGET_TOTAL_ETA].values
    kpt_pred_prod = kpt_model.predict(prod_df[KPT_FEATURES])
    vtt_pred_prod = vtt_model.predict(prod_df[VTT_FEATURES])
    pred_prod = kpt_pred_prod + vtt_pred_prod + 0.5
    prod_metrics = calculate_eta_metrics(y_prod, pred_prod)

    mae_degradation = (prod_metrics["mae_min"] - base_metrics["mae_min"]) / base_metrics["mae_min"]
    rmse_degradation = (prod_metrics["rmse_min"] - base_metrics["rmse_min"]) / base_metrics["rmse_min"]
    concept_drift_detected = bool(mae_degradation >= CONCEPT_DRIFT_MAE_DEGRADATION)

    return {
        "baseline_mae": base_metrics["mae_min"],
        "production_mae": prod_metrics["mae_min"],
        "mae_degradation_ratio": round(float(mae_degradation), 4),
        "baseline_rmse": base_metrics["rmse_min"],
        "production_rmse": prod_metrics["rmse_min"],
        "rmse_degradation_ratio": round(float(rmse_degradation), 4),
        "baseline_on_time_rate": base_metrics["on_time_rate_percent"],
        "production_on_time_rate": prod_metrics["on_time_rate_percent"],
        "concept_drift_detected": concept_drift_detected,
        "action_required": "TRIGGER_AUTOMATED_RETRAINING" if concept_drift_detected else "NO_ACTION"
    }

def generate_drift_dashboard(train_df: pd.DataFrame, prod_df: pd.DataFrame, save_path: str):
    """
    Generates a visual comparison dashboard of key drifted features.
    """
    key_features = ["rain_intensity_mm", "traffic_density", "kitchen_active_orders", "distance_km"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(key_features):
        ax = axes[idx]
        ax.hist(train_df[feature], bins=30, alpha=0.6, label="Baseline (Reference)", color="#2563eb", density=True)
        ax.hist(prod_df[feature], bins=30, alpha=0.6, label="Production Stream", color="#dc2626", density=True)
        ax.set_title(f"Feature Distribution: {feature}", fontsize=12, fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("MLOps Production Drift Monitoring: Baseline vs Production Stream", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Drift dashboard saved to: {save_path}")

def run_monitoring_pipeline():
    print("=" * 70)
    print("RUNNING PRODUCTION DRIFT MONITORING ENGINE")
    print("=" * 70)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading reference baseline: {TRAIN_DATA_PATH}")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"Loading production stream: {PRODUCTION_STREAM_PATH}")
    prod_df = pd.read_csv(PRODUCTION_STREAM_PATH)

    print("\nLoading production model pipelines...")
    kpt_model = joblib.load(KPT_MODEL_PATH)
    vtt_model = joblib.load(VTT_MODEL_PATH)

    # 1. Data Drift Analysis
    print("\nAnalyzing Data Drift (KS-Test and PSI)...")
    data_drift_summary = detect_data_drift(train_df, prod_df)
    for feat, res in data_drift_summary["features"].items():
        status = "⚠️ DRIFT" if res["drift_detected"] else "✅ STABLE"
        print(f"  [{status}] {feat:<24} | KS p-val: {res['ks_pvalue']:.4e} | PSI: {res['psi_value']:.4f} ({res['psi_status']})")

    # 2. Concept Drift Analysis
    print("\nAnalyzing Concept Drift (Ground Truth Accuracy Degradation)...")
    concept_drift_summary = detect_concept_drift(train_df, prod_df, kpt_model, vtt_model)
    print(f"  Baseline MAE:   {concept_drift_summary['baseline_mae']:.2f} min")
    print(f"  Production MAE: {concept_drift_summary['production_mae']:.2f} min")
    print(f"  MAE Degradation: {concept_drift_summary['mae_degradation_ratio'] * 100:.1f}%")
    print(f"  Baseline On-Time Rate:   {concept_drift_summary['baseline_on_time_rate']:.1f}%")
    print(f"  Production On-Time Rate: {concept_drift_summary['production_on_time_rate']:.1f}%")
    print(f"  Concept Drift Detected:  {concept_drift_summary['concept_drift_detected']}")
    print(f"  Action Triggered:        {concept_drift_summary['action_required']}")

    # 3. Export Summary & Visualizations
    dashboard_path = REPORTS_DIR / "drift_dashboard.png"
    generate_drift_dashboard(train_df, prod_df, str(dashboard_path))

    full_report = {
        "data_drift": data_drift_summary,
        "concept_drift": concept_drift_summary
    }
    report_json_path = REPORTS_DIR / "drift_summary.json"
    with open(report_json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Report summary saved to: {report_json_path}")

    return full_report

if __name__ == "__main__":
    run_monitoring_pipeline()
