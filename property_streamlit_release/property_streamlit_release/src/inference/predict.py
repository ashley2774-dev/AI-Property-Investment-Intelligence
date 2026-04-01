from __future__ import annotations

from typing import Dict

import pandas as pd

from .amortization import build_amortization_schedule
from .benchmarks import build_location_assumptions
from .finance import compute_financials
from .model import load_model
from .recommendation import build_recommendation


def _build_features(assumptions: Dict[str, float | str], financials: Dict[str, float]) -> Dict[str, float | str]:
    floor_area = float(assumptions["floor_area_sqm"])
    purchase_price = float(assumptions["purchase_price"])
    monthly_rent = float(assumptions["monthly_rent"])

    return {
        **assumptions,
        "gross_yield": financials["gross_yield"],
        "net_yield": financials["net_yield"],
        "roi": financials["roi"],
        "dscr": financials["dscr"],
        "monthly_cash_flow": financials["monthly_cash_flow"],
        "price_per_sqm": purchase_price / floor_area if floor_area else 0.0,
        "rent_per_sqm": monthly_rent / floor_area if floor_area else 0.0,
        "bond_to_rent": financials["monthly_bond_payment"] / monthly_rent if monthly_rent else 0.0,
        "opex_to_rent": (financials["opex_annual"] / 12) / monthly_rent if monthly_rent else 0.0,
    }


def evaluate_location(province: str, suburb: str) -> Dict[str, object]:
    assumptions = build_location_assumptions(province=province, suburb=suburb)
    financials = compute_financials(assumptions)
    features = _build_features(assumptions, financials)

    X = pd.DataFrame([features])
    model, model_source = load_model()
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    prob_map = {label: float(prob) for label, prob in zip(classes, probabilities)}
    prediction = max(prob_map, key=prob_map.get)
    confidence = prob_map[prediction]

    recommendation = build_recommendation(
        prediction=prediction,
        confidence=confidence,
        financials=financials,
        assumptions=assumptions,
    )

    schedule, break_even_month = build_amortization_schedule(assumptions, financials)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": prob_map,
        "assumptions": assumptions,
        "features": features,
        "financials": financials,
        "recommendation": recommendation,
        "model_source": model_source,
        "amortization_schedule": schedule,
        "cashflow_break_even_month": break_even_month,
        "cashflow_break_even_month_label": (
            "Not within horizon" if break_even_month is None else f"Month {break_even_month}"
        ),
    }
