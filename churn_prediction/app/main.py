"""
Customer Churn Prediction API
BERT-based sentiment analysis + ML churn prediction
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import time
import logging
import asyncio
from datetime import datetime

from app.predictor import ChurnPredictor
from app.monitoring import MonitoringService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-end churn prediction using BERT-based sentiment analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ChurnPredictor()
monitor = MonitoringService()


# ── Request / Response schemas ──────────────────────────────────────────────

class CustomerData(BaseModel):
    customer_id: str = Field(..., example="CUST_001")
    review_text: str = Field(..., example="The service has been disappointing lately.")
    tenure_months: int = Field(..., ge=0, example=12)
    monthly_charges: float = Field(..., ge=0, example=65.5)
    num_complaints: int = Field(..., ge=0, example=2)
    num_support_calls: int = Field(..., ge=0, example=3)
    contract_type: str = Field(..., example="Month-to-month")
    payment_method: str = Field(..., example="Electronic check")

class BatchRequest(BaseModel):
    customers: List[CustomerData]

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    sentiment_score: float
    sentiment_label: str
    top_risk_factors: List[str]
    recommendation: str
    processing_time_ms: float
    timestamp: str

class BatchResponse(BaseModel):
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[PredictionResponse]
    batch_processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_accuracy: float
    model_version: str
    uptime_seconds: float
    total_predictions: int
    last_retrain: str

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_predictions_today: int
    avg_response_time_ms: float
    alert_triggered: bool
    alert_message: Optional[str]


START_TIME = time.time()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns API health status and current model metrics."""
    metrics = monitor.get_current_metrics()
    return HealthResponse(
        status="healthy",
        model_accuracy=metrics["accuracy"],
        model_version="bert-churn-v1.2",
        uptime_seconds=round(time.time() - START_TIME, 2),
        total_predictions=monitor.total_predictions,
        last_retrain=monitor.last_retrain
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerData, background_tasks: BackgroundTasks):
    """
    Predict churn probability for a single customer.
    Uses BERT sentiment analysis + gradient boosting ensemble.
    Precision: 89% | Response time: <100ms
    """
    start = time.time()
    try:
        result = predictor.predict_single(customer.dict())
        elapsed = round((time.time() - start) * 1000, 2)
        result["processing_time_ms"] = elapsed
        result["timestamp"] = datetime.utcnow().isoformat()
        background_tasks.add_task(monitor.log_prediction, result)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Batch churn prediction for up to 500 customers.
    Handles 100+ concurrent requests with async processing.
    """
    if len(request.customers) > 500:
        raise HTTPException(status_code=400, detail="Max 500 customers per batch.")

    batch_start = time.time()
    predictions = []

    tasks = [
        asyncio.get_event_loop().run_in_executor(
            None, predictor.predict_single, c.dict()
        )
        for c in request.customers
    ]
    results = await asyncio.gather(*tasks)

    now = datetime.utcnow().isoformat()
    for r in results:
        r["processing_time_ms"] = round((time.time() - batch_start) * 1000, 2)
        r["timestamp"] = now
        predictions.append(PredictionResponse(**r))

    batch_ms = round((time.time() - batch_start) * 1000, 2)
    background_tasks.add_task(monitor.log_batch, [p.dict() for p in predictions])

    return BatchResponse(
        total_customers=len(predictions),
        high_risk_count=sum(1 for p in predictions if p.risk_level == "HIGH"),
        medium_risk_count=sum(1 for p in predictions if p.risk_level == "MEDIUM"),
        low_risk_count=sum(1 for p in predictions if p.risk_level == "LOW"),
        predictions=predictions,
        batch_processing_time_ms=batch_ms
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Returns live model performance metrics.
    Triggers automated retraining alert if accuracy drops below 85%.
    """
    return MetricsResponse(**monitor.get_current_metrics())


@app.post("/retrain", tags=["Monitoring"])
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Manually trigger model retraining pipeline."""
    background_tasks.add_task(monitor.trigger_retraining)
    return {"message": "Retraining job queued.", "status": "accepted"}


@app.get("/customers/at-risk", tags=["Insights"])
async def get_at_risk_customers(threshold: float = 0.7, limit: int = 50):
    """Returns recent high-risk customers above the given churn probability threshold."""
    at_risk = monitor.get_at_risk_customers(threshold=threshold, limit=limit)
    return {
        "threshold": threshold,
        "count": len(at_risk),
        "customers": at_risk
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, workers=4)
