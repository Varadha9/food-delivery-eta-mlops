"""
Configuration settings for Food-Delivery ETA ML System.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
REPORTS_DIR = BASE_DIR / "reports"

DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)
DOCS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

# Datasets
TRAIN_DATA_PATH = DATA_DIR / "orders_train.csv"
TEST_DATA_PATH = DATA_DIR / "orders_test.csv"
PRODUCTION_STREAM_PATH = DATA_DIR / "orders_production_stream.csv"

# Model Artifacts
KPT_MODEL_PATH = MODELS_DIR / "kpt_model.joblib"
VTT_MODEL_PATH = MODELS_DIR / "vtt_model.joblib"
PIPELINE_METADATA_PATH = MODELS_DIR / "metadata.json"

# Feature Names
KPT_FEATURES = [
    "item_count",
    "cuisine_complexity",
    "kitchen_active_orders",
    "is_peak_hour",
    "restaurant_avg_prep_min",
    "order_value_usd"
]

VTT_FEATURES = [
    "distance_km",
    "traffic_density",
    "rain_intensity_mm",
    "rider_speed_kmh",
    "rider_wait_time_min"
]

ALL_FEATURES = KPT_FEATURES + VTT_FEATURES

# Target Variables
TARGET_KPT = "kitchen_prep_time_min"
TARGET_VTT = "transit_time_min"
TARGET_TOTAL_ETA = "actual_total_delivery_min"

# SLAs and Evaluation Thresholds
LATENCY_P95_MS = 30.0
LATENCY_P99_MS = 50.0
TARGET_MAX_MAE_MIN = 3.0
ASYMMETRIC_TAU = 0.75  # Underestimating delivery time is penalized 3x more

# Drift Thresholds
KS_ALPHA = 0.05
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
CONCEPT_DRIFT_MAE_DEGRADATION = 0.20  # 20% error increase triggers retraining
