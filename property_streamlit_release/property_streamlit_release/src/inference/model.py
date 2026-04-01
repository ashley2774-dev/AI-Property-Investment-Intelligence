from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "investment_screener_pipeline.joblib"


class FallbackAreaModel:
    """Transparent fallback scorer when the trained pipeline artifact is unavailable."""

    classes_ = ["reject", "review", "recommended"]

    def predict_proba(self, X: pd.DataFrame):
        rows = []
        for _, row in X.iterrows():
            score = 0.0
            score += min(max((row.get("gross_yield", 0) - 0.08) / 0.06, -1), 1) * 0.20
            score += min(max((row.get("net_yield", 0) - 0.05) / 0.05, -1), 1) * 0.20
            score += min(max((row.get("dscr", 0) - 1.1) / 0.5, -1), 1) * 0.25
            score += min(max(row.get("roi", 0) / 0.30, -1), 1) * 0.20
            score += min(max(row.get("monthly_cash_flow", 0) / 3000.0, -1), 1) * 0.15

            recommended = max(0.05, min(0.90, 0.20 + max(score, -0.15) * 0.55))
            reject = max(0.05, min(0.90, 0.20 + max(-score, -0.15) * 0.55))
            review = max(0.05, 1 - recommended - reject)
            total = recommended + reject + review
            rows.append([reject / total, review / total, recommended / total])
        return rows



def load_model() -> Tuple[object, str]:
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), "trained_pipeline"
        except Exception:
            pass
    return FallbackAreaModel(), "fallback_area_model"
