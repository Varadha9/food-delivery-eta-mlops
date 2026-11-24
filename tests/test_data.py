import pytest
import pandas as pd
from src.data_generator import generate_orders_dataset
from src.config import ALL_FEATURES, TARGET_TOTAL_ETA

def test_dataset_generation_shape_and_columns():
    df = generate_orders_dataset(n_samples=100, random_seed=42)
    assert len(df) == 100
    for col in ALL_FEATURES + [TARGET_TOTAL_ETA]:
        assert col in df.columns, f"Missing expected column: {col}"

def test_dataset_values_within_physical_bounds():
    df = generate_orders_dataset(n_samples=200, random_seed=42)
    assert (df["distance_km"] > 0).all()
    assert (df["item_count"] >= 1).all()
    assert (df["actual_total_delivery_min"] > 5.0).all()
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected NaNs"

def test_drift_generation_shifts_distributions():
    df_normal = generate_orders_dataset(n_samples=500, random_seed=1, simulate_drift=False)
    df_drifted = generate_orders_dataset(n_samples=500, random_seed=1, simulate_drift=True)
    
    # Drifted dataset should have significantly higher rain and kitchen queues
    assert df_drifted["rain_intensity_mm"].mean() > df_normal["rain_intensity_mm"].mean()
    assert df_drifted["kitchen_active_orders"].mean() > df_normal["kitchen_active_orders"].mean()
