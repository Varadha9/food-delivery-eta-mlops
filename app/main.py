"""
FastAPI Microservice for Food Delivery ETA Prediction.
Includes Prometheus metrics instrumentation, health checks, and OpenAPI docs.
"""
import time
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.schemas import (
    ETAPredictionRequest,
    ETAPredictionResponse,
    ETABatchPredictionRequest,
    ETABatchPredictionResponse,
    HealthCheckResponse
)
from app.predictor import predictor

START_TIME = time.time()

app = FastAPI(
    title="Food Delivery Real-Time ETA Predictor API",
    description="Production-grade MLOps microservice providing low-latency multi-component delivery ETA predictions.",
    version="1.0.0"
)

# Prometheus Metrics Instrumentation
REQUEST_COUNT = Counter(
    "eta_requests_total",
    "Total number of ETA prediction requests received",
    ["endpoint", "serving_mode", "status"]
)
REQUEST_LATENCY = Histogram(
    "eta_request_duration_seconds",
    "Latency of ETA prediction requests in seconds",
    ["endpoint"],
    buckets=[0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.250, 0.500]
)
PREDICTED_ETA_HISTOGRAM = Histogram(
    "eta_predicted_delivery_minutes",
    "Distribution of predicted delivery times in minutes",
    buckets=[10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 90]
)
FALLBACK_TRIGGERED = Counter(
    "eta_fallback_triggered_total",
    "Total occurrences of degraded rule-based fallback triggered"
)

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check():
    """
    Kubernetes Liveness & Readiness probe endpoint.
    """
    model_loaded = (predictor.kpt_model is not None and predictor.vtt_model is not None)
    return HealthCheckResponse(
        status="HEALTHY" if model_loaded else "DEGRADED",
        model_loaded=model_loaded,
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 2)
    )

@app.post("/predict/eta", response_model=ETAPredictionResponse, tags=["Inference"])
def predict_single_eta(request: ETAPredictionRequest):
    """
    Real-time single order delivery ETA prediction for checkout and rider dispatch.
    """
    start_time = time.perf_counter()
    try:
        response = predictor.predict(request)
        duration = time.perf_counter() - start_time
        
        # Track metrics
        REQUEST_COUNT.labels(
            endpoint="/predict/eta",
            serving_mode=response.serving_mode,
            status="200"
        ).inc()
        REQUEST_LATENCY.labels(endpoint="/predict/eta").observe(duration)
        PREDICTED_ETA_HISTOGRAM.observe(response.total_eta_min)
        
        if response.serving_mode == "degraded_fallback":
            FALLBACK_TRIGGERED.inc()
            
        return response
    except Exception as e:
        REQUEST_COUNT.labels(
            endpoint="/predict/eta",
            serving_mode="error",
            status="500"
        ).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}"
        )

@app.post("/predict/batch", response_model=ETABatchPredictionResponse, tags=["Inference"])
def predict_batch_eta(batch_request: ETABatchPredictionRequest):
    """
    High-throughput batch ETA predictions for catalog feed badges and browse-mode listings.
    """
    start_time = time.perf_counter()
    responses = predictor.predict_batch(batch_request.orders)
    total_duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

    REQUEST_COUNT.labels(
        endpoint="/predict/batch",
        serving_mode="batch",
        status="200"
    ).inc()
    REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(total_duration_ms / 1000.0)

    return ETABatchPredictionResponse(
        predictions=responses,
        batch_size=len(responses),
        total_latency_ms=total_duration_ms
    )

@app.get("/metrics", tags=["Observability"])
def prometheus_metrics():
    """
    Prometheus scraping endpoint exposing latency histograms, request counters, and fallback rates.
    """
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
