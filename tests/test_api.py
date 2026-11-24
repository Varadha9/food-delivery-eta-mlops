import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["HEALTHY", "DEGRADED"]
    assert "uptime_seconds" in data

def test_single_eta_prediction_success():
    payload = {
        "order_id": "ORD-TEST-99",
        "restaurant_id": "REST-010",
        "customer_id": "CUST-500",
        "distance_km": 3.8,
        "item_count": 2,
        "cuisine_complexity": 1.2,
        "traffic_density": 0.5,
        "rain_intensity_mm": 1.0
    }
    response = client.post("/predict/eta", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORD-TEST-99"
    assert data["total_eta_min"] > 0
    assert data["kitchen_prep_time_min"] > 0
    assert data["transit_time_min"] > 0
    assert "-" in data["estimated_arrival_window"]
    assert data["serving_mode"] in ["realtime_model", "degraded_fallback"]

def test_single_eta_validation_error():
    # Negative distance is invalid
    payload = {
        "order_id": "ORD-INVALID",
        "restaurant_id": "REST-001",
        "customer_id": "CUST-001",
        "distance_km": -5.0
    }
    response = client.post("/predict/eta", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_batch_eta_prediction():
    payload = {
        "orders": [
            {
                "order_id": f"ORD-B-{i}",
                "restaurant_id": "REST-005",
                "customer_id": f"CUST-{i}",
                "distance_km": 2.5 + i
            }
            for i in range(3)
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 3
    assert len(data["predictions"]) == 3

def test_prometheus_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "eta_requests_total" in response.text
