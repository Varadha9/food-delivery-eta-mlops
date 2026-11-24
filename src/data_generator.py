"""
Data Generator for Food Delivery ETA ML System.
Generates realistic baseline datasets (training, testing) and production stream datasets
with intentional seasonal/traffic drift to test monitoring pipelines.
"""
import numpy as np
import pandas as pd
from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PRODUCTION_STREAM_PATH,
    DATA_DIR
)

def generate_orders_dataset(
    n_samples: int = 10000,
    random_seed: int = 42,
    simulate_drift: bool = False
) -> pd.DataFrame:
    """
    Generates a realistic food-delivery dataset.
    
    If simulate_drift is True, simulates adverse weather (heavy rain),
    heightened traffic congestion, and kitchen overload (e.g. monsoon festival rush).
    """
    np.random.seed(random_seed)
    
    # 1. Base order identifiers
    order_ids = [f"ORD-{i:06d}" for i in range(1, n_samples + 1)]
    restaurant_ids = [f"REST-{np.random.randint(1, 150):03d}" for _ in range(n_samples)]
    customer_ids = [f"CUST-{np.random.randint(1, 5000):04d}" for _ in range(n_samples)]
    rider_ids = [f"RIDER-{np.random.randint(1, 800):03d}" for _ in range(n_samples)]
    
    # 2. Kitchen features
    item_count = np.random.poisson(lam=3.2, size=n_samples) + 1
    cuisine_complexity = np.random.choice([0.8, 1.0, 1.2, 1.5, 1.8], size=n_samples, p=[0.2, 0.4, 0.2, 0.15, 0.05])
    
    if simulate_drift:
        # Severe kitchen queue surge
        kitchen_active_orders = np.random.negative_binomial(n=5, p=0.25, size=n_samples) + 5
        rain_intensity_mm = np.random.exponential(scale=6.5, size=n_samples) # Heavy rain
        traffic_density = np.clip(np.random.beta(a=5, b=2, size=n_samples), 0.1, 0.99) # Severe traffic
    else:
        # Normal operational distribution
        kitchen_active_orders = np.random.poisson(lam=4.5, size=n_samples)
        rain_intensity_mm = np.random.choice([0.0, 0.5, 1.5, 3.0], size=n_samples, p=[0.75, 0.15, 0.08, 0.02])
        traffic_density = np.clip(np.random.beta(a=2, b=4, size=n_samples), 0.05, 0.95)
        
    is_peak_hour = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    restaurant_avg_prep_min = np.random.normal(loc=14.0, scale=3.0, size=n_samples).clip(min=6.0, max=28.0)
    order_value_usd = np.round((item_count * np.random.uniform(7.0, 16.0, size=n_samples)) + 5.0, 2)
    
    # 3. Courier & Transit features
    distance_km = np.random.gamma(shape=2.5, scale=1.6, size=n_samples).clip(min=0.5, max=18.0)
    rider_speed_kmh = np.random.normal(loc=26.0, scale=4.0, size=n_samples).clip(min=12.0, max=45.0)
    rider_wait_time_min = np.random.exponential(scale=2.5, size=n_samples).clip(min=0.5, max=12.0)
    
    # 4. Realistic ground truth target calculation (with physics and stochastic variance)
    # Kitchen Prep Time (KPT) in minutes
    kpt_noise = np.random.normal(0, 1.8, size=n_samples)
    kpt_raw = (
        restaurant_avg_prep_min
        + (item_count * 1.4 * cuisine_complexity)
        + (kitchen_active_orders * 0.75)
        + (is_peak_hour * 2.8)
        + kpt_noise
    ).clip(min=5.0)
    
    # Vehicle Transit Time (VTT) in minutes
    effective_speed = (rider_speed_kmh * (1.0 - 0.45 * traffic_density)).clip(min=8.0)
    transit_noise = np.random.normal(0, 1.5, size=n_samples)
    vtt_raw = (
        (distance_km / effective_speed) * 60.0
        + (rain_intensity_mm * 0.55)
        + rider_wait_time_min
        + transit_noise
    ).clip(min=4.0)
    
    actual_total_delivery_min = np.round(kpt_raw + vtt_raw, 2)
    
    df = pd.DataFrame({
        "order_id": order_ids,
        "restaurant_id": restaurant_ids,
        "customer_id": customer_ids,
        "rider_id": rider_ids,
        "item_count": item_count,
        "cuisine_complexity": np.round(cuisine_complexity, 2),
        "kitchen_active_orders": kitchen_active_orders,
        "is_peak_hour": is_peak_hour,
        "restaurant_avg_prep_min": np.round(restaurant_avg_prep_min, 2),
        "order_value_usd": order_value_usd,
        "distance_km": np.round(distance_km, 2),
        "traffic_density": np.round(traffic_density, 3),
        "rain_intensity_mm": np.round(rain_intensity_mm, 2),
        "rider_speed_kmh": np.round(rider_speed_kmh, 1),
        "rider_wait_time_min": np.round(rider_wait_time_min, 2),
        "kitchen_prep_time_min": np.round(kpt_raw, 2),
        "transit_time_min": np.round(vtt_raw, 2),
        "actual_total_delivery_min": actual_total_delivery_min
    })
    return df

def generate_all_datasets():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    print("Generating baseline training dataset (20,000 orders)...")
    train_df = generate_orders_dataset(n_samples=20000, random_seed=42, simulate_drift=False)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    print(f"Saved: {TRAIN_DATA_PATH} ({len(train_df)} rows)")

    print("Generating baseline holdout test dataset (5,000 orders)...")
    test_df = generate_orders_dataset(n_samples=5000, random_seed=101, simulate_drift=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved: {TEST_DATA_PATH} ({len(test_df)} rows)")

    print("Generating production stream dataset with simulated drift (5,000 orders)...")
    prod_df = generate_orders_dataset(n_samples=5000, random_seed=999, simulate_drift=True)
    prod_df.to_csv(PRODUCTION_STREAM_PATH, index=False)
    print(f"Saved: {PRODUCTION_STREAM_PATH} ({len(prod_df)} rows)")

if __name__ == "__main__":
    generate_all_datasets()
