"""
Generates clean, professional high-resolution architecture diagram for Food Delivery ETA ML System.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_system_architecture_diagram(output_path: str = "docs/architecture_diagram.png"):
    Path("docs").mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    # Header
    ax.text(50, 97.2, "FOOD DELIVERY REAL-TIME ETA PREDICTOR: ENTERPRISE MLOPS ARCHITECTURE",
            fontsize=15, fontweight="bold", ha="center", va="center", color="#0f172a")
    ax.text(50, 94.6, "Multi-Component Architecture (KPT + VTT) • Dual-Store Feature Store • Kubernetes Serving • Drift Observability",
            fontsize=9.5, ha="center", va="center", color="#475569")

    def draw_box(x, y, w, h, title, subtitle="", bg="#ffffff", border="#94a3b8", title_color="#0f172a"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.0",
                                     facecolor=bg, edgecolor=border, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.64, title, fontsize=9.5, fontweight="bold", ha="center", va="center", color=title_color, zorder=4)
            ax.text(x + w/2, y + h*0.30, subtitle, fontsize=7.5, ha="center", va="center", color="#475569", zorder=4)
        else:
            ax.text(x + w/2, y + h*0.5, title, fontsize=9.5, fontweight="bold", ha="center", va="center", color=title_color, zorder=4)

    def draw_container(x, y, w, h, label, bg="#f1f5f9", border="#cbd5e1"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.0,rounding_size=1.2",
                                     facecolor=bg, edgecolor=border, linewidth=1.2, linestyle="--", zorder=1)
        ax.add_patch(rect)
        ax.text(x + 2, y + h - 2.5, label, fontsize=8.5, fontweight="bold", color="#334155", zorder=2)

    def draw_arrow(x1, y1, x2, y2, label="", color="#475569", rad=0.0, label_pos=0.5, label_offset=1.3):
        cstyle = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                    mutation_scale=12, connectionstyle=cstyle), zorder=5)
        if label:
            mx = x1 + (x2 - x1) * label_pos
            my = y1 + (y2 - y1) * label_pos
            ax.text(mx, my + label_offset, label, fontsize=7.5, fontweight="semibold", color="#1e293b", ha="center", va="center",
                    bbox=dict(boxstyle="square,pad=0.25", fc="#ffffff", ec="#cbd5e1", lw=0.6, alpha=0.95), zorder=6)

    # 1. CLIENT & INGRESS LAYER
    draw_container(2, 69, 36, 23, "1. INGESTION & CLIENT INTERFACES", bg="#eff6ff", border="#93c5fd")
    draw_box(4, 82, 16, 7, "Customer App", "Checkout & Live Tracking", bg="#ffffff", border="#3b82f6")
    draw_box(4, 72, 16, 7, "Driver App", "Dispatch & Route Navigation", bg="#ffffff", border="#3b82f6")
    draw_box(23, 74, 13, 13, "API Gateway", "Envoy / Kong\nTLS, Auth, Rate Limit\nRoutes to K8s", bg="#fef3c7", border="#f59e0b", title_color="#b45309")

    # 2. KUBERNETES SERVING PODS
    draw_container(41, 69, 57, 23, "2. REAL-TIME SERVING CLUSTER (KUBERNETES / EKS)", bg="#ecfdf5", border="#6ee7b7")
    draw_box(43, 79, 26, 9.5, "FastAPI Inference Engine", "Auto-scaled Pods (3-25)\nConcurrent Async Workers", bg="#ffffff", border="#10b981")
    draw_box(72, 79, 24, 9.5, "Degraded Fallback Shield", "Deterministic Speed-Distance\nRule Heuristics (<5ms)", bg="#fef2f2", border="#ef4444", title_color="#b91c1c")
    draw_box(43, 71, 53, 6.5, "Ensemble ML Model: Kitchen Prep Time (KPT) + Transit Travel Time (VTT)", "Gradient Boosting Regressors • Asymmetric Quantile Loss Buffer (+0.5 min)", bg="#f0fdf4", border="#34d399")

    # 3. FEATURE STORE LAYER
    draw_container(2, 36, 46, 29, "3. DUAL-STORAGE FEATURE STORE (FEAST)", bg="#f5f3ff", border="#c4b5fd")
    draw_box(4, 52, 20, 9.5, "Online Store (Redis)", "P95 Latency < 4ms\nKitchen Queue, GPS, Weather", bg="#ffffff", border="#8b5cf6")
    draw_box(26, 52, 20, 9.5, "Offline Store (Parquet/S3)", "Point-in-Time Accurate Joins\nHistorical Order Warehouse", bg="#ffffff", border="#8b5cf6")
    draw_box(4, 38, 42, 10, "Event Streaming Bus (Kafka + Flink)", "Kafka Topics: orders.placed, courier.gps, order.delivered, kitchen.load\nFlink Stateful Streaming -> Feature Hydration", bg="#faf5ff", border="#a855f7")

    # 4. CONTINUOUS TRAINING & REGISTRY (CI/CD)
    draw_container(51, 36, 47, 29, "4. CONTINUOUS TRAINING & REGISTRY PIPELINE", bg="#fffbeb", border="#fde68a")
    draw_box(53, 52, 20, 9.5, "Training Pipeline", "Kubeflow / Airflow DAG\nFeature Join, Tuning, Validate", bg="#ffffff", border="#d97706")
    draw_box(76, 52, 20, 9.5, "MLflow Registry", "Versioned Model Artifacts\nStaging -> Prod Approval Gate", bg="#ffffff", border="#d97706")
    draw_box(53, 38, 43, 10, "Automated CI/CD (GitHub Actions)", "Code Linting (Flake8) • Pytest Suite (12 Tests)\nDocker Container Build • Vulnerability Scan", bg="#fff7ed", border="#f97316")

    # 5. OBSERVABILITY, DRIFT MONITORING & FEEDBACK LOOP
    draw_container(2, 4, 96, 28, "5. OBSERVABILITY, DRIFT MONITORING & CLOSED-LOOP RETRAINING", bg="#f8fafc", border="#cbd5e1")
    draw_box(4, 7, 29, 21, "Prometheus & Grafana", "• P50, P95, P99 Latency (<30ms SLA)\n• Throughput (10k QPS Peak Rush)\n• Error Rates & HTTP 4xx/5xx\n• Fallback Invocation Rate Alerting\n• Pod CPU / Memory Utilization", bg="#ffffff", border="#0284c7")
    draw_box(36, 7, 30, 21, "Drift Detection Engine", "• Data Drift: KS-Test (p<0.05) & PSI\n  (monitors rain, traffic, kitchen load)\n• Concept Drift: MAE & RMSE tracking\n  against delivered order ground truth\n• Retraining Webhook Trigger when\n  MAE degradation > 20%", bg="#ffffff", border="#e11d48", title_color="#9f1239")
    draw_box(69, 7, 27, 21, "Closed-Loop Feedback System", "• Actual 'order.delivered' Event Logged\n• Computes Error: e = y_actual - y_pred\n• Automated Parquet Partition Sink\n• Triggers Airflow Continuous Training\n• Eliminates Training-Serving Skew", bg="#ffffff", border="#059669")

    # ARROWS & FLOWS
    # Ingestion flows
    draw_arrow(20, 85, 23, 83)
    draw_arrow(20, 75, 23, 79)
    draw_arrow(36, 82, 43, 82, "REST /predict/eta")
    draw_arrow(69, 83, 72, 83, "Fallback on Timeout", color="#dc2626")

    # Serving to Redis
    draw_arrow(43, 79, 24, 61.5, "Feature Lookup (<4ms)", color="#8b5cf6")

    # Kafka to Redis and Offline Store
    draw_arrow(14, 48, 14, 52, "Stateful Updates", color="#8b5cf6")
    draw_arrow(36, 48, 36, 52, "Lakehouse Sinks", color="#8b5cf6")

    # Offline Store to Training
    draw_arrow(46, 56, 53, 56, "Point-in-Time Join", color="#d97706")
    draw_arrow(73, 56, 76, 56, "Register Models", color="#d97706")
    draw_arrow(86, 61.5, 78, 71, "GitOps Deploy to K8s", color="#059669", rad=0.15)

    # Serving Metrics to Observability
    draw_arrow(43, 71, 28, 28, "Prometheus /metrics", color="#0284c7", rad=-0.1)
    draw_arrow(60, 71, 51, 28, "Inference Telemetry", color="#e11d48")

    # Ground truth feedback to Kafka bus
    draw_arrow(69, 22, 46, 42, "Stream order.delivered", color="#059669", rad=-0.15, label_pos=0.4)
    # Drift alert to training orchestrator
    draw_arrow(48, 28, 55, 38, "Trigger Retrain Alert", color="#dc2626", label_pos=0.6, label_offset=1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Clean architecture diagram generated at: {output_path}")

if __name__ == "__main__":
    generate_system_architecture_diagram()
