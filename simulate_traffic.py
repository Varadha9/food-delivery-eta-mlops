"""
Traffic Simulator for Food Delivery ETA Prediction Service.
Generates realistic client traffic (normal orders, peak hour surges, adverse weather,
batch listing requests, and validation edge cases) to test and visualize Prometheus/Grafana metrics.
"""
import time
import random
import requests
from typing import Optional

def send_traffic(base_url: str = "http://localhost:8080", num_requests: int = 60, delay: float = 0.2):
    print(f"Starting traffic simulation against {base_url} ({num_requests} requests)...")
    
    success_count = 0
    validation_err_count = 0

    for i in range(1, num_requests + 1):
        order_id = f"SIM-{i:05d}"
        restaurant_id = f"REST-{random.randint(1, 80):03d}"
        customer_id = f"CUST-{random.randint(1, 2000):04d}"
        
        # Simulate different scenarios:
        # Scenario 1: Normal clear day order (60%)
        # Scenario 2: Peak rush with high kitchen backlog (25%)
        # Scenario 3: Heavy rainstorm monsoon order (15%)
        rand_scenario = random.random()
        
        if rand_scenario < 0.60:
            scenario_name = "Normal Order"
            item_count = random.randint(1, 4)
            complexity = random.choice([0.8, 1.0, 1.2])
            distance = round(random.uniform(1.2, 6.0), 2)
            traffic = round(random.uniform(0.1, 0.45), 2)
            rain = 0.0
            kitchen_orders = random.randint(2, 6)
            is_peak = 0
        elif rand_scenario < 0.85:
            scenario_name = "Peak Hour Rush"
            item_count = random.randint(3, 8)
            complexity = random.choice([1.2, 1.5, 1.8])
            distance = round(random.uniform(2.5, 10.0), 2)
            traffic = round(random.uniform(0.6, 0.95), 2)
            rain = round(random.choice([0.0, 0.5, 1.5]), 1)
            kitchen_orders = random.randint(8, 22)
            is_peak = 1
        else:
            scenario_name = "Monsoon Downpour"
            item_count = random.randint(2, 6)
            complexity = 1.2
            distance = round(random.uniform(3.0, 12.0), 2)
            traffic = round(random.uniform(0.7, 0.98), 2)
            rain = round(random.uniform(4.0, 15.0), 1)
            kitchen_orders = random.randint(5, 14)
            is_peak = 1

        payload = {
            "order_id": order_id,
            "restaurant_id": restaurant_id,
            "customer_id": customer_id,
            "item_count": item_count,
            "cuisine_complexity": complexity,
            "distance_km": distance,
            "traffic_density": traffic,
            "rain_intensity_mm": rain,
            "kitchen_active_orders": kitchen_orders,
            "is_peak_hour": is_peak
        }

        try:
            resp = requests.post(f"{base_url}/predict/eta", json=payload, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                success_count += 1
                if i % 5 == 0 or i == num_requests:
                    print(f"[{i:02d}/{num_requests}] {scenario_name:<16} -> ETA: {data['total_eta_min']:>5.1f} min ({data['estimated_arrival_window']}) | Mode: {data['serving_mode']} | Latency: {data['latency_ms']}ms")
            else:
                print(f"[{i:02d}] HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[{i:02d}] Request failed: {e}")

        # Inject 1 intentional validation error every 20 requests to test 422 monitoring
        if i % 20 == 0:
            bad_payload = payload.copy()
            bad_payload["distance_km"] = -3.0
            requests.post(f"{base_url}/predict/eta", json=bad_payload)
            validation_err_count += 1

        if delay > 0:
            time.sleep(delay)

    # Also test batch endpoint
    print("\nSending batch catalog ETA request (10 restaurant cards)...")
    batch_payload = {
        "orders": [
            {
                "order_id": f"BATCH-{k}",
                "restaurant_id": f"REST-{k:03d}",
                "customer_id": "CUST-BROWSE",
                "distance_km": round(2.0 + k * 0.8, 1)
            }
            for k in range(1, 11)
        ]
    }
    batch_resp = requests.post(f"{base_url}/predict/batch", json=batch_payload)
    if batch_resp.status_code == 200:
        b_data = batch_resp.json()
        print(f"Batch response received: {b_data['batch_size']} predictions in {b_data['total_latency_ms']}ms")

    print("\n" + "=" * 60)
    print(f"Traffic Simulation Complete! Total: {num_requests} requests, Successful: {success_count}, Validation tests: {validation_err_count}")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    send_traffic(base_url=url, num_requests=count, delay=0.15)
