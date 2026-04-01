from __future__ import annotations

import pandas as pd

from .features import build_features
from .model import RuleBasedModel

MODEL = RuleBasedModel()


def evaluate_deal(payload: dict) -> dict:
    features, financials = build_features(payload)
    X = pd.DataFrame([features])

    probabilities = MODEL.predict_proba(X)[0]
    prediction = MODEL.predict(X)[0]
    probability_map = dict(zip(MODEL.classes_, probabilities))
    confidence = max(probabilities)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_map,
        "features": features,
        "financials": financials,
    }
