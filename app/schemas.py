"""
Pydantic Schemas for Food Delivery ETA Prediction Microservice.
"""
from typing import Optional, List
from pydantic import BaseModel, Field

class ETAPredictionRequest(BaseModel):
    order_id: str = Field(..., description="Unique order identifier", json_schema_extra={"example": "ORD-000123"})
    restaurant_id: str = Field(..., description="Restaurant identifier", json_schema_extra={"example": "REST-042"})
    customer_id: str = Field(..., description="Customer identifier", json_schema_extra={"example": "CUST-1049"})
    rider_id: Optional[str] = Field(None, description="Assigned rider identifier (if already dispatched)", json_schema_extra={"example": "RIDER-088"})
    
    # Kitchen features
    item_count: int = Field(default=3, ge=1, le=50, description="Total number of items in order")
    cuisine_complexity: float = Field(default=1.0, ge=0.5, le=3.0, description="Cuisine complexity factor")
    kitchen_active_orders: Optional[int] = Field(default=5, ge=0, description="Active orders in kitchen queue")
    is_peak_hour: int = Field(default=0, ge=0, le=1, description="1 if during lunch/dinner rush, else 0")
    restaurant_avg_prep_min: Optional[float] = Field(default=15.0, ge=3.0, description="Historical restaurant average prep time")
    order_value_usd: Optional[float] = Field(default=28.50, ge=1.0, description="Total order monetary value in USD")
    
    # Transit features
    distance_km: float = Field(..., ge=0.1, le=50.0, description="Driving distance in kilometers", json_schema_extra={"example": 4.2})
    traffic_density: float = Field(default=0.4, ge=0.0, le=1.0, description="Congestion index between 0.0 (free) and 1.0 (gridlock)")
    rain_intensity_mm: float = Field(default=0.0, ge=0.0, le=100.0, description="Precipitation rate in mm/hour")
    rider_speed_kmh: Optional[float] = Field(default=25.0, ge=5.0, le=80.0, description="Estimated courier average speed")
    rider_wait_time_min: Optional[float] = Field(default=2.5, ge=0.0, le=30.0, description="Expected rider wait time at kitchen")

class ETAPredictionResponse(BaseModel):
    order_id: str
    total_eta_min: float = Field(..., description="Predicted total customer delivery ETA in minutes")
    kitchen_prep_time_min: float = Field(..., description="Estimated kitchen cooking time")
    transit_time_min: float = Field(..., description="Estimated courier transit and handover time")
    estimated_arrival_window: str = Field(..., description="Estimated delivery window string")
    serving_mode: str = Field(..., description="'realtime_model' or 'degraded_fallback'")
    latency_ms: float = Field(..., description="Server inference execution time in milliseconds")

class ETABatchPredictionRequest(BaseModel):
    orders: List[ETAPredictionRequest]

class ETABatchPredictionResponse(BaseModel):
    predictions: List[ETAPredictionResponse]
    batch_size: int
    total_latency_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float
