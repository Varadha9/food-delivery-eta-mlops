#!/bin/bash
set -e

PORT=${1:-8080}
echo "=========================================================="
echo " Starting Food Delivery ETA Prediction Service on Port $PORT"
echo "=========================================================="

echo "1. Building Docker image..."
docker build -t food-delivery-eta:latest .

echo "2. Removing any existing container..."
docker rm -f eta_prediction_service 2>/dev/null || true

echo "3. Starting container in background..."
docker run -d --name eta_prediction_service -p ${PORT}:8000 food-delivery-eta:latest

echo "4. Waiting for service readiness..."
for i in {1..20}; do
  if curl -s http://localhost:${PORT}/health | grep -q "HEALTHY"; then
    echo "Service is healthy and ready!"
    break
  fi
  sleep 1
done

echo ""
echo "Health status:"
curl -s http://localhost:${PORT}/health
echo ""

echo ""
echo "5. Executing sample ETA prediction:"
curl -s -X POST "http://localhost:${PORT}/predict/eta" \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "ORD-LIVE-TEST",
    "restaurant_id": "REST-042",
    "customer_id": "CUST-1049",
    "item_count": 3,
    "cuisine_complexity": 1.2,
    "distance_km": 4.5,
    "traffic_density": 0.5,
    "rain_intensity_mm": 1.5,
    "kitchen_active_orders": 7
  }'
echo ""

echo ""
echo "=========================================================="
echo " Service is live and running in Docker!"
echo " Swagger UI Docs:  http://localhost:${PORT}/docs"
echo " Health Endpoint:  http://localhost:${PORT}/health"
echo " Metrics:          http://localhost:${PORT}/metrics"
echo ""
echo " To stop container:  docker stop eta_prediction_service"
echo " To view logs:       docker logs -f eta_prediction_service"
echo "=========================================================="
