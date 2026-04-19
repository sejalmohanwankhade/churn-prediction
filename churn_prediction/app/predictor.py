"""
ChurnPredictor — BERT sentiment analysis + Gradient Boosting ensemble
Simulates production BERT pipeline without requiring GPU/heavy dependencies.
"""

import numpy as np
import re
import hashlib
from typing import Dict, Any, List


# ── Sentiment Lexicons ────────────────────────────────────────────────────────

NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "worst", "bad", "poor", "useless",
    "disappointing", "frustrating", "annoying", "slow", "broken", "unfair",
    "expensive", "overpriced", "cheated", "scam", "waste", "rude", "incompetent",
    "cancel", "leaving", "quit", "switch", "unacceptable", "pathetic", "ridiculous",
    "never", "hate", "disgusted", "failed", "ignored", "problem", "issue", "bug"
}

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "fantastic", "love", "happy", "satisfied",
    "wonderful", "best", "perfect", "awesome", "good", "helpful", "efficient",
    "fast", "reliable", "recommend", "pleased", "impressed", "outstanding",
    "brilliant", "superb", "exceptional", "smooth", "easy", "convenient"
}

NEGATION_WORDS = {"not", "never", "no", "don't", "doesn't", "didn't", "won't", "can't"}

RISK_PHRASES = {
    "thinking about cancel": 0.4,
    "going to cancel": 0.5,
    "already cancelled": 0.6,
    "switching to": 0.35,
    "very disappointed": 0.25,
    "extremely frustrated": 0.3,
    "terrible service": 0.3,
    "worst experience": 0.35,
    "not worth": 0.2,
    "waste of money": 0.3,
}


class BERTSentimentSimulator:
    """
    Simulates BERT-based sentiment analysis.
    In production: replace with transformers pipeline.
    
    Production code:
        from transformers import pipeline
        self.nlp = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    """

    def analyze(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        tokens = re.findall(r'\b\w+\b', text_lower)

        neg_count = pos_count = 0
        negation_active = False
        risk_boost = 0.0

        for i, token in enumerate(tokens):
            if token in NEGATION_WORDS:
                negation_active = True
                continue
            if token in NEGATIVE_WORDS:
                if negation_active:
                    pos_count += 0.5
                else:
                    neg_count += 1
                negation_active = False
            elif token in POSITIVE_WORDS:
                if negation_active:
                    neg_count += 0.5
                else:
                    pos_count += 1
                negation_active = False
            else:
                negation_active = False

        for phrase, boost in RISK_PHRASES.items():
            if phrase in text_lower:
                risk_boost += boost

        total = neg_count + pos_count + 1e-9
        raw_score = (pos_count - neg_count) / total  # range [-1, 1]

        # Add deterministic noise based on text hash for realism
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        noise = ((seed % 100) / 100 - 0.5) * 0.15
        sentiment_score = float(np.clip(raw_score + noise, -1, 1))

        label = (
            "VERY_NEGATIVE" if sentiment_score < -0.6 else
            "NEGATIVE"      if sentiment_score < -0.2 else
            "NEUTRAL"       if sentiment_score < 0.2  else
            "POSITIVE"      if sentiment_score < 0.6  else
            "VERY_POSITIVE"
        )

        return {
            "score": round(sentiment_score, 4),
            "label": label,
            "risk_boost": round(min(risk_boost, 0.5), 4),
            "negative_signals": int(neg_count),
            "positive_signals": int(pos_count)
        }


class GradientBoostingSimulator:
    """
    Simulates a trained Gradient Boosting model (XGBoost).
    Feature weights derived from typical churn model importance scores.
    """

    FEATURE_WEIGHTS = {
        "tenure_months":      -0.018,   # longer tenure → less likely to churn
        "monthly_charges":     0.008,   # higher charges → more likely to churn
        "num_complaints":      0.12,
        "num_support_calls":   0.07,
        "contract_score":     -0.25,    # month-to-month vs annual
        "payment_risk":        0.08,    # electronic check = riskier
        "sentiment_score":    -0.30,    # negative sentiment → high churn
    }

    def predict_proba(self, features: Dict[str, float]) -> float:
        logit = -0.5  # base (intercept)
        for feat, weight in self.FEATURE_WEIGHTS.items():
            logit += weight * features.get(feat, 0)

        # Sigmoid
        prob = 1 / (1 + np.exp(-logit))
        return float(np.clip(prob, 0.01, 0.99))


class ChurnPredictor:

    def __init__(self):
        self.sentiment_model = BERTSentimentSimulator()
        self.churn_model = GradientBoostingSimulator()
        self.model_version = "bert-churn-v1.2"

    # ── Feature Engineering ────────────────────────────────────────────────

    def _encode_contract(self, contract: str) -> float:
        mapping = {
            "Month-to-month": 1.0,
            "One year":       0.3,
            "Two year":       0.0
        }
        return mapping.get(contract, 0.5)

    def _encode_payment(self, method: str) -> float:
        risky = {"Electronic check", "Mailed check"}
        return 1.0 if method in risky else 0.0

    def _get_risk_factors(self, features: Dict, sentiment: Dict, prob: float) -> List[str]:
        factors = []
        if features["num_complaints"] >= 3:
            factors.append(f"High complaint frequency ({features['num_complaints']} complaints)")
        if features["num_support_calls"] >= 4:
            factors.append(f"Excessive support calls ({features['num_support_calls']} calls)")
        if sentiment["label"] in ("NEGATIVE", "VERY_NEGATIVE"):
            factors.append(f"Negative customer sentiment ({sentiment['label']})")
        if features["contract_score"] >= 0.8:
            factors.append("Month-to-month contract (high flexibility to leave)")
        if features["monthly_charges"] > 80:
            factors.append(f"High monthly charges (${features['monthly_charges']:.0f}/mo)")
        if features["tenure_months"] < 6:
            factors.append(f"Low tenure ({features['tenure_months']} months) — onboarding risk")
        if features["payment_risk"] == 1.0:
            factors.append("High-risk payment method (electronic/mailed check)")
        if not factors:
            factors.append("Marginal risk profile — monitor usage trends")
        return factors[:4]

    def _get_recommendation(self, risk_level: str, factors: List[str]) -> str:
        recommendations = {
            "HIGH": (
                "🚨 Immediate action required: Assign dedicated retention specialist. "
                "Offer a personalised discount (15–25%) or contract upgrade. "
                "Schedule a proactive call within 24 hours."
            ),
            "MEDIUM": (
                "⚠️ Proactive outreach recommended: Send a satisfaction survey and targeted "
                "retention offer. Consider loyalty rewards or a service upgrade trial."
            ),
            "LOW": (
                "✅ Low risk: Maintain regular engagement. Monitor for sentiment changes. "
                "Include in NPS surveys for feedback collection."
            )
        }
        return recommendations.get(risk_level, "Monitor customer closely.")

    # ── Main Prediction ────────────────────────────────────────────────────

    def predict_single(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        sentiment = self.sentiment_model.analyze(customer["review_text"])

        features = {
            "tenure_months":    customer["tenure_months"],
            "monthly_charges":  customer["monthly_charges"],
            "num_complaints":   customer["num_complaints"],
            "num_support_calls":customer["num_support_calls"],
            "contract_score":   self._encode_contract(customer["contract_type"]),
            "payment_risk":     self._encode_payment(customer["payment_method"]),
            "sentiment_score":  sentiment["score"],
        }

        base_prob = self.churn_model.predict_proba(features)
        final_prob = float(np.clip(base_prob + sentiment["risk_boost"], 0.01, 0.99))
        final_prob = round(final_prob, 4)

        risk_level = (
            "HIGH"   if final_prob >= 0.70 else
            "MEDIUM" if final_prob >= 0.40 else
            "LOW"
        )

        risk_factors = self._get_risk_factors(features, sentiment, final_prob)
        recommendation = self._get_recommendation(risk_level, risk_factors)

        return {
            "customer_id":       customer["customer_id"],
            "churn_probability": final_prob,
            "churn_prediction":  final_prob >= 0.5,
            "risk_level":        risk_level,
            "sentiment_score":   sentiment["score"],
            "sentiment_label":   sentiment["label"],
            "top_risk_factors":  risk_factors,
            "recommendation":    recommendation,
        }
