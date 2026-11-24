"""
Feature Engineering and Feature Store Abstraction.
Implements feature extraction pipelines and simulates Feast-style
dual storage (Offline point-in-time parquet joins + Online low-latency Redis/In-Memory lookup).
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.config import KPT_FEATURES, VTT_FEATURES

def create_feature_pipeline() -> Tuple[ColumnTransformer, ColumnTransformer]:
    """
    Builds Scikit-learn transformer pipelines for KPT and VTT sub-models.
    """
    kpt_transformer = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), KPT_FEATURES)
        ],
        remainder="passthrough"
    )
    
    vtt_transformer = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), VTT_FEATURES)
        ],
        remainder="passthrough"
    )
    return kpt_transformer, vtt_transformer

class FeatureStoreMock:
    """
    Simulates a Feast / Hopsworks production feature store.
    Features:
      - Online Store: low-latency in-memory dictionary (simulating Redis cluster).
      - Offline Store: batch point-in-time dataset generator.
    """
    def __init__(self):
        self._online_store: Dict[str, Dict[str, Any]] = {}
        self._populate_mock_online_entities()

    def _populate_mock_online_entities(self):
        # Seed standard restaurant & rider baseline feature vectors
        for i in range(1, 151):
            rest_id = f"REST-{i:03d}"
            self._online_store[f"restaurant:{rest_id}"] = {
                "restaurant_avg_prep_min": float(np.random.uniform(10.0, 20.0)),
                "kitchen_active_orders": int(np.random.randint(1, 10))
            }
        for j in range(1, 801):
            rider_id = f"RIDER-{j:03d}"
            self._online_store[f"rider:{rider_id}"] = {
                "rider_speed_kmh": float(np.random.uniform(22.0, 32.0)),
                "rider_wait_time_min": float(np.random.uniform(1.0, 4.0))
            }

    def get_online_features(self, entity_keys: Dict[str, str]) -> Dict[str, Any]:
        """
        Retrieves real-time features with sub-millisecond latency.
        """
        features = {}
        rest_id = entity_keys.get("restaurant_id")
        rider_id = entity_keys.get("rider_id")
        
        if rest_id:
            rest_data = self._online_store.get(f"restaurant:{rest_id}", {
                "restaurant_avg_prep_min": 15.0,
                "kitchen_active_orders": 5
            })
            features.update(rest_data)
            
        if rider_id:
            rider_data = self._online_store.get(f"rider:{rider_id}", {
                "rider_speed_kmh": 25.0,
                "rider_wait_time_min": 2.5
            })
            features.update(rider_data)
            
        return features

    def update_online_features(self, entity_key: str, features: Dict[str, Any]):
        """
        Ingests real-time streaming feature updates from Kafka / Flink.
        """
        if entity_key not in self._online_store:
            self._online_store[entity_key] = {}
        self._online_store[entity_key].update(features)
