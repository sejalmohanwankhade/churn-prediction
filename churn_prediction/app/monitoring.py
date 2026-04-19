"""
Monitoring Service
- Tracks model accuracy in real time
- Triggers automated retraining alerts when accuracy < 85%
- Stores prediction logs in-memory (swap for Redis/PostgreSQL in production)
"""

import time
import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)

ACCURACY_THRESHOLD = 0.85


class MonitoringService:

    def __init__(self):
        self.prediction_log: deque = deque(maxlen=10_000)
        self.total_predictions: int = 0
        self.last_retrain: str = (datetime.utcnow() - timedelta(days=3)).isoformat()
        self._response_times: deque = deque(maxlen=1000)
        self._alert_active: bool = False
        self._alert_message: Optional[str] = None
        self._simulated_accuracy = 0.892   # starts above threshold
        self._start_time = time.time()
        self._predictions_today = 0

    # ── Logging ──────────────────────────────────────────────────────────────

    def log_prediction(self, result: Dict[str, Any]):
        self.prediction_log.append(result)
        self.total_predictions += 1
        self._predictions_today += 1
        if "processing_time_ms" in result:
            self._response_times.append(result["processing_time_ms"])
        self._check_accuracy_threshold()

    def log_batch(self, results: List[Dict[str, Any]]):
        for r in results:
            self.log_prediction(r)

    # ── Metrics ──────────────────────────────────────────────────────────────

    def get_current_metrics(self) -> Dict[str, Any]:
        # Simulate slight accuracy drift over time for realism
        drift = random.uniform(-0.005, 0.003)
        self._simulated_accuracy = max(0.78, min(0.95, self._simulated_accuracy + drift))

        alert_triggered = self._simulated_accuracy < ACCURACY_THRESHOLD
        alert_msg = (
            f"⚠️ Model accuracy dropped to {self._simulated_accuracy:.1%} "
            f"(threshold: {ACCURACY_THRESHOLD:.0%}). Retraining pipeline triggered."
            if alert_triggered else None
        )

        avg_rt = (
            round(sum(self._response_times) / len(self._response_times), 2)
            if self._response_times else 45.0
        )

        return {
            "accuracy":  round(self._simulated_accuracy, 4),
            "precision": round(self._simulated_accuracy + random.uniform(0.002, 0.015), 4),
            "recall":    round(self._simulated_accuracy - random.uniform(0.005, 0.020), 4),
            "f1_score":  round(self._simulated_accuracy + random.uniform(-0.01, 0.01), 4),
            "total_predictions_today": self._predictions_today,
            "avg_response_time_ms":    avg_rt,
            "alert_triggered":         alert_triggered,
            "alert_message":           alert_msg,
        }

    # ── Threshold Check ───────────────────────────────────────────────────────

    def _check_accuracy_threshold(self):
        metrics = self.get_current_metrics()
        if metrics["alert_triggered"] and not self._alert_active:
            self._alert_active = True
            logger.warning(
                f"ACCURACY ALERT: {metrics['accuracy']:.2%} < {ACCURACY_THRESHOLD:.0%}. "
                "Queuing retraining job..."
            )
            # In production: trigger Airflow DAG / MLflow run / SageMaker pipeline
        elif not metrics["alert_triggered"]:
            self._alert_active = False

    # ── Retraining ────────────────────────────────────────────────────────────

    def trigger_retraining(self):
        logger.info("Retraining pipeline started...")
        time.sleep(2)  # simulate pipeline kick-off
        self._simulated_accuracy = 0.891  # post-retrain accuracy
        self.last_retrain = datetime.utcnow().isoformat()
        self._alert_active = False
        logger.info(f"Retraining complete. New accuracy: {self._simulated_accuracy:.2%}")

    # ── At-risk Query ─────────────────────────────────────────────────────────

    def get_at_risk_customers(self, threshold: float = 0.7, limit: int = 50) -> List[Dict]:
        at_risk = [
            {
                "customer_id":       p.get("customer_id"),
                "churn_probability": p.get("churn_probability"),
                "risk_level":        p.get("risk_level"),
                "sentiment_label":   p.get("sentiment_label"),
                "timestamp":         p.get("timestamp"),
            }
            for p in list(self.prediction_log)
            if p.get("churn_probability", 0) >= threshold
        ]
        at_risk.sort(key=lambda x: x["churn_probability"], reverse=True)
        return at_risk[:limit]
