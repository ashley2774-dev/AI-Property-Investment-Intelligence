from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .finance import calculate_financials


DEFAULTS = {
    "province": "Gauteng",
    "city": "Johannesburg",
    "suburb": "Unknown",
    "property_type": "Apartment",
    "bedrooms": 2,
    "bathrooms": 1.0,
    "parking": 1,
    "purchase_price": 950000.0,
    "estimated_rent": 8500.0,
    "floor_area_sqm": 65.0,
    "levy": 1200.0,
    "rates_taxes": 700.0,
    "insurance": 350.0,
    "other_opex": 250.0,
    "deposit_pct": 10.0,
    "interest_rate_pct": 11.75,
    "loan_term_years": 20,
    "vacancy_pct": 5.0,
    "management_fee_pct": 8.0,
    "maintenance_pct": 5.0,
    "annual_growth_pct": 6.0,
}


ALIASES = {
    "garage": "parking",
    "garages": "parking",
    "monthly_levy": "levy",
    "monthly_rates_taxes": "rates_taxes",
    "monthly_insurance": "insurance",
    "rent": "estimated_rent",
    "rent_estimate": "estimated_rent",
    "price": "purchase_price",
    "area_sqm": "floor_area_sqm",
    "size_sqm": "floor_area_sqm",
}


def _normalize_payload(payload: dict) -> dict:
    normalized = DEFAULTS.copy()
    for key, value in payload.items():
        std_key = ALIASES.get(key, key)
        normalized[std_key] = value
    return normalized


def _inject_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_rows = []
    for _, row in df.iterrows():
        payload = _normalize_payload(row.to_dict())
        finance = calculate_financials(payload)
        combined = payload.copy()
        combined.update(finance.__dict__)

        combined["bedroom_density_floor"] = float(combined.get("bedrooms", 0)) / max(float(combined.get("floor_area_sqm", 1)), 1.0)
        combined["bathroom_density_floor"] = float(combined.get("bathrooms", 0)) / max(float(combined.get("floor_area_sqm", 1)), 1.0)
        combined["cash_flow_margin"] = finance.monthly_cash_flow / finance.estimated_rent if finance.estimated_rent else 0.0
        combined["debt_service_headroom"] = finance.dscr - 1.0
        combined["rates_to_rent"] = float(combined.get("rates_taxes", 0)) / finance.estimated_rent if finance.estimated_rent else 0.0
        combined["levy_to_rent"] = float(combined.get("levy", 0)) / finance.estimated_rent if finance.estimated_rent else 0.0
        combined["price_per_bedroom"] = finance.purchase_price / max(float(combined.get("bedrooms", 1)), 1.0)
        combined["price_per_bathroom"] = finance.purchase_price / max(float(combined.get("bathrooms", 1)), 1.0)
        combined["rent_per_bedroom"] = finance.estimated_rent / max(float(combined.get("bedrooms", 1)), 1.0)
        combined["gross_rent_to_price"] = (finance.estimated_rent * 12.0) / finance.purchase_price if finance.purchase_price else 0.0
        combined["occupancy_pct"] = 100.0 - float(payload.get("vacancy_pct", 0))
        combined["rent_estimation_source"] = combined.get("rent_estimation_source", "user_input")
        output_rows.append(combined)

    return pd.DataFrame(output_rows)


def _align_to_feature_list(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    aligned = df.copy()
    feature_names = list(feature_names)

    for col in feature_names:
        if col not in aligned.columns:
            if col.endswith("_missing_flag"):
                aligned[col] = 1
            elif col in {"suburb", "city", "province", "property_type", "rent_estimation_source"}:
                aligned[col] = DEFAULTS.get(col, "Unknown")
            else:
                aligned[col] = np.nan

    aligned = aligned[feature_names].copy()
    return aligned


def build_single_property_features(payload: dict, feature_names: Iterable[str]) -> pd.DataFrame:
    normalized = _normalize_payload(payload)
    raw_df = pd.DataFrame([normalized])
    engineered = _inject_engineered_columns(raw_df)
    return _align_to_feature_list(engineered, feature_names)


def coerce_batch_input(batch_df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    if batch_df.empty:
        raise ValueError("Uploaded CSV is empty.")

    renamed = batch_df.rename(columns={c: ALIASES.get(c, c) for c in batch_df.columns})
    for key, value in DEFAULTS.items():
        if key not in renamed.columns:
            renamed[key] = value

    engineered = _inject_engineered_columns(renamed)
    return _align_to_feature_list(engineered, feature_names)
