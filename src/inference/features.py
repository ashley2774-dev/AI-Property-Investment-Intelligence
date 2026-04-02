from __future__ import annotations

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

        floor_area = max(float(combined.get("floor_area_sqm", 1.0)), 1.0)
        bedrooms = max(float(combined.get("bedrooms", 1.0)), 1.0)
        bathrooms = max(float(combined.get("bathrooms", 1.0)), 1.0)
        estimated_rent = float(finance.estimated_rent)
        purchase_price = float(finance.purchase_price)

        combined["bedroom_density_floor"] = float(combined.get("bedrooms", 0.0)) / floor_area
        combined["bathroom_density_floor"] = float(combined.get("bathrooms", 0.0)) / floor_area
        combined["cash_flow_margin"] = finance.monthly_cash_flow / estimated_rent if estimated_rent else 0.0
        combined["debt_service_headroom"] = finance.dscr - 1.0
        combined["rates_to_rent"] = float(combined.get("rates_taxes", 0.0)) / estimated_rent if estimated_rent else 0.0
        combined["levy_to_rent"] = float(combined.get("levy", 0.0)) / estimated_rent if estimated_rent else 0.0
        combined["price_per_bedroom"] = purchase_price / bedrooms if purchase_price else 0.0
        combined["price_per_bathroom"] = purchase_price / bathrooms if purchase_price else 0.0
        combined["rent_per_bedroom"] = estimated_rent / bedrooms if estimated_rent else 0.0
        combined["gross_rent_to_price"] = (estimated_rent * 12.0) / purchase_price if purchase_price else 0.0
        combined["occupancy_pct"] = 100.0 - float(payload.get("vacancy_pct", 0.0))
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

    return aligned[feature_names].copy()


def build_single_property_features(
    payload: dict, feature_names: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        business_df: engineered dataframe with financial outputs and business-facing columns
        model_df: feature-aligned dataframe used strictly for model inference
    """
    normalized = _normalize_payload(payload)
    raw_df = pd.DataFrame([normalized])

    business_df = _inject_engineered_columns(raw_df)
    model_df = _align_to_feature_list(business_df, feature_names)

    return business_df, model_df


def coerce_batch_input(
    batch_df: pd.DataFrame, feature_names: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        business_df: engineered dataframe with financial outputs and business-facing columns
        model_df: feature-aligned dataframe used strictly for model inference
    """
    if batch_df.empty:
        raise ValueError("Uploaded CSV is empty.")

    renamed = batch_df.rename(columns={c: ALIASES.get(c, c) for c in batch_df.columns})

    for key, value in DEFAULTS.items():
        if key not in renamed.columns:
            renamed[key] = value

    business_df = _inject_engineered_columns(renamed)
    model_df = _align_to_feature_list(business_df, feature_names)

    return business_df, model_df
