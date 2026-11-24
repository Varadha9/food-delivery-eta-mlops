import numpy as np
import pytest
from src.features import FeatureStoreMock
from src.evaluate import calculate_eta_metrics, asymmetric_pinball_loss
from src.drift_monitor import calculate_psi

def test_feature_store_online_retrieval():
    fs = FeatureStoreMock()
    feats = fs.get_online_features({"restaurant_id": "REST-001", "rider_id": "RIDER-001"})
    assert "restaurant_avg_prep_min" in feats
    assert "rider_speed_kmh" in feats
    assert feats["restaurant_avg_prep_min"] > 0

def test_asymmetric_pinball_loss():
    # If delivery is late (actual = 40, predicted = 30), penalty should be higher
    late_loss = asymmetric_pinball_loss(np.array([40.0]), np.array([30.0]), tau=0.75)
    # If delivery is early (actual = 30, predicted = 40), penalty should be lower
    early_loss = asymmetric_pinball_loss(np.array([30.0]), np.array([40.0]), tau=0.75)
    
    assert late_loss == pytest.approx(0.75 * 10.0)
    assert early_loss == pytest.approx(0.25 * 10.0)
    assert late_loss > early_loss

def test_psi_identical_distributions():
    data = np.random.normal(loc=10, scale=2, size=1000)
    psi = calculate_psi(data, data)
    assert psi < 0.05, f"PSI should be near zero for identical distributions, got {psi}"

def test_psi_shifted_distributions():
    ref = np.random.normal(loc=10, scale=2, size=1000)
    shifted = np.random.normal(loc=20, scale=4, size=1000)
    psi = calculate_psi(ref, shifted)
    assert psi > 0.25, f"PSI should flag significant drift, got {psi}"
