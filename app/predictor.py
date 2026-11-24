"""
Inference Engine and Fallback Predictor for Food Delivery ETA Service.
"""
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.config import (
    KPT_MODEL_PATH,
    VTT_MODEL_PATH,
    KPT_FEATURES,
    VTT_FEATURES
)
from src.features import FeatureStoreMock
from app.schemas import ETAPredictionRequest, ETAPredictionResponse

class ETAPredictor:
    def __init__(self):
        self.kpt_model = None
        self.vtt_model = None
        self.feature_store = FeatureStoreMock()
        self.load_models()

    def load_models(self):
        try:
            if KPT_MODEL_PATH.exists() and VTT_MODEL_PATH.exists():
                self.kpt_model = joblib.load(KPT_MODEL_PATH)
                self.vtt_model = joblib.load(VTT_MODEL_PATH)
                print("ETAPredictor: Models loaded successfully from disk.")
            else:
                print("ETAPredictor: Model artifacts not found. Operating in fallback mode.")
        except Exception as e:
            print(f"ETAPredictor Error loading models: {e}. Falling back to heuristics.")
            self.kpt_model = None
            self.vtt_model = None

    def _rule_based_fallback(self, request: ETAPredictionRequest) -> Dict[str, float]:
        """
        Degraded fallback heuristics when ML model is unavailable or throws exception.
        Guarantees 99.99% availability within SLA (<5ms).
        """
        # Baseline prep time heuristic
        kpt = 15.0 + (request.item_count * 1.5 * request.cuisine_complexity) + (request.is_peak_hour * 3.0)
        
        # Baseline transit heuristic
        effective_speed = 22.0 * (1.0 - 0.4 * request.traffic_density)
        vtt = (request.distance_km / max(effective_speed, 8.0)) * 60.0 + (request.rain_intensity_mm * 0.5) + 3.0
        
        return {
            "kpt": round(float(kpt), 2),
            "vtt": round(float(vtt), 2),
            "total": round(float(kpt + vtt + 0.5), 2)
        }

    def predict(self, request: ETAPredictionRequest) -> ETAPredictionResponse:
        start_time = time.perf_counter()
        
        # 1. Feature Store Hydration (if fields need real-time lookup)
        entity_keys = {
            "restaurant_id": request.restaurant_id,
            "rider_id": request.rider_id or "RIDER-001"
        }
        online_feats = self.feature_store.get_online_features(entity_keys)
        
        rest_avg_prep = request.restaurant_avg_prep_min or online_feats.get("restaurant_avg_prep_min", 15.0)
        kitchen_orders = request.kitchen_active_orders if request.kitchen_active_orders is not None else online_feats.get("kitchen_active_orders", 5)
        rider_speed = request.rider_speed_kmh or online_feats.get("rider_speed_kmh", 25.0)
        rider_wait = request.rider_wait_time_min or online_feats.get("rider_wait_time_min", 2.5)

        # 2. Attempt ML Inference
        serving_mode = "realtime_model"
        if self.kpt_model is not None and self.vtt_model is not None:
            try:
                # Prepare KPT input
                kpt_input = pd.DataFrame([{
                    "item_count": request.item_count,
                    "cuisine_complexity": request.cuisine_complexity,
                    "kitchen_active_orders": kitchen_orders,
                    "is_peak_hour": request.is_peak_hour,
                    "restaurant_avg_prep_min": rest_avg_prep,
                    "order_value_usd": request.order_value_usd or (request.item_count * 12.0)
                }])[KPT_FEATURES]
                
                # Prepare VTT input
                vtt_input = pd.DataFrame([{
                    "distance_km": request.distance_km,
                    "traffic_density": request.traffic_density,
                    "rain_intensity_mm": request.rain_intensity_mm,
                    "rider_speed_kmh": rider_speed,
                    "rider_wait_time_min": rider_wait
                }])[VTT_FEATURES]

                pred_kpt = float(self.kpt_model.predict(kpt_input)[0])
                pred_vtt = float(self.vtt_model.predict(vtt_input)[0])
                pred_total = round(pred_kpt + pred_vtt + 0.5, 2)
            except Exception as e:
                # Fallback on any prediction error
                fallback = self._rule_based_fallback(request)
                pred_kpt, pred_vtt, pred_total = fallback["kpt"], fallback["vtt"], fallback["total"]
                serving_mode = "degraded_fallback"
        else:
            fallback = self._rule_based_fallback(request)
            pred_kpt, pred_vtt, pred_total = fallback["kpt"], fallback["vtt"], fallback["total"]
            serving_mode = "degraded_fallback"

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        lower_bound = max(10, int(pred_total - 3))
        upper_bound = int(pred_total + 4)
        arrival_window = f"{lower_bound}-{upper_bound} mins"

        return ETAPredictionResponse(
            order_id=request.order_id,
            total_eta_min=round(pred_total, 2),
            kitchen_prep_time_min=round(pred_kpt, 2),
            transit_time_min=round(pred_vtt, 2),
            estimated_arrival_window=arrival_window,
            serving_mode=serving_mode,
            latency_ms=latency_ms
        )

    def predict_batch(self, requests: List[ETAPredictionRequest]) -> List[ETAPredictionResponse]:
        return [self.predict(req) for req in requests]

# Singleton instance
predictor = ETAPredictor()
