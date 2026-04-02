from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return np.where((denominator == 0) | pd.isna(denominator), np.nan, numerator / denominator)


def winsorize_series(s, lower=0.01, upper=0.99):
    s = pd.to_numeric(s, errors="coerce")
    non_null = s.dropna()
    if non_null.empty:
        return s
    lower_q = non_null.quantile(lower)
    upper_q = non_null.quantile(upper)
    return s.clip(lower=lower_q, upper=upper_q)


DEFAULTS = {
    "province": "Gauteng",
    "city": "Johannesburg",
    "suburb": "Unknown",
    "property_type": "Apartment",
    "bedrooms": 2,
    "bathrooms": 1.0,
    "parking": 1,
    "garage": 0,
    "purchase_price": 950000.0,
    "estimated_rent": 8500.0,
    "floor_area_sqm": 65.0,
    "land_area_sqm": np.nan,
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
    "rent_estimation_source": "user_input",
}

ALIASES = {
    "garage_spaces": "parking",
    "garages": "garage",
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
        normalized[ALIASES.get(key, key)] = value
    return normalized


def _build_financial_engine(df: pd.DataFrame) -> pd.DataFrame:
    fe = df.copy()

    fe["purchase_price"] = pd.to_numeric(fe["purchase_price"], errors="coerce")
    fe["estimated_monthly_rent"] = pd.to_numeric(fe["estimated_rent"], errors="coerce")
    fe["levies_clean"] = pd.to_numeric(fe.get("levy", 0), errors="coerce").fillna(0)
    fe["rates_taxes_clean"] = pd.to_numeric(fe.get("rates_taxes", 0), errors="coerce").fillna(0)

    fe["deposit_amount"] = fe["purchase_price"] * (pd.to_numeric(fe["deposit_pct"], errors="coerce") / 100.0)
    fe["loan_amount"] = fe["purchase_price"] - fe["deposit_amount"]
    fe["transfer_costs"] = fe["purchase_price"] * 0.08
    fe["total_cash_invested"] = fe["deposit_amount"] + fe["transfer_costs"]

    annual_rate = pd.to_numeric(fe["interest_rate_pct"], errors="coerce") / 100.0
    monthly_rate = annual_rate / 12.0
    n = pd.to_numeric(fe["loan_term_years"], errors="coerce") * 12.0
    fe["bond_payment"] = np.where(
        (fe["loan_amount"].isna()) | (n <= 0),
        np.nan,
        np.where(
            monthly_rate == 0,
            fe["loan_amount"] / n,
            fe["loan_amount"] * (monthly_rate * (1 + monthly_rate) ** n) / (((1 + monthly_rate) ** n) - 1),
        ),
    )

    fe["gross_rent"] = fe["estimated_monthly_rent"]
    fe["effective_rent"] = fe["gross_rent"] * (1 - (pd.to_numeric(fe["vacancy_pct"], errors="coerce") / 100.0))
    fe["management_fee"] = fe["gross_rent"] * (pd.to_numeric(fe["management_fee_pct"], errors="coerce") / 100.0)
    fe["maintenance"] = fe["gross_rent"] * (pd.to_numeric(fe["maintenance_pct"], errors="coerce") / 100.0)

    insurance_monthly = pd.to_numeric(fe.get("insurance", 0), errors="coerce").fillna(0)
    other_opex = pd.to_numeric(fe.get("other_opex", 0), errors="coerce").fillna(0)

    fe["insurance"] = insurance_monthly
    fe["operating_expenses"] = (
        fe["levies_clean"]
        + fe["rates_taxes_clean"]
        + fe["management_fee"]
        + fe["maintenance"]
        + fe["insurance"]
        + other_opex
    )

    fe["noi"] = fe["effective_rent"] - fe["operating_expenses"]
    fe["cash_flow"] = fe["noi"] - fe["bond_payment"]
    fe["rental_yield"] = safe_divide(fe["gross_rent"] * 12.0, fe["purchase_price"])
    fe["ROI"] = safe_divide(fe["cash_flow"] * 12.0, fe["total_cash_invested"])
    fe["DSCR"] = safe_divide(fe["noi"], fe["bond_payment"])

    return fe


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = _build_financial_engine(df)

    fe["parking_spaces"] = pd.to_numeric(fe.get("parking", 0), errors="coerce").fillna(0)
    fe["garage"] = pd.to_numeric(fe.get("garage", 0), errors="coerce").fillna(0)

    fe["total_rooms_proxy"] = pd.to_numeric(fe.get("bedrooms", 0), errors="coerce").fillna(0) + pd.to_numeric(fe.get("bathrooms", 0), errors="coerce").fillna(0)
    fe["total_parking_proxy"] = fe["parking_spaces"] + fe["garage"]

    fe["price_per_sqm"] = safe_divide(fe["purchase_price"], fe.get("floor_area_sqm"))
    fe["price_per_land_sqm"] = safe_divide(fe["purchase_price"], fe.get("land_area_sqm"))
    fe["rent_per_sqm"] = safe_divide(fe["estimated_monthly_rent"], fe.get("floor_area_sqm"))
    fe["rent_per_bedroom"] = safe_divide(fe["estimated_monthly_rent"], fe.get("bedrooms"))
    fe["rent_per_bathroom"] = safe_divide(fe["estimated_monthly_rent"], fe.get("bathrooms"))
    fe["levies_to_rent"] = safe_divide(fe.get("levies_clean"), fe["estimated_monthly_rent"])
    fe["rates_to_rent"] = safe_divide(fe.get("rates_taxes_clean"), fe["estimated_monthly_rent"])
    fe["bond_to_rent"] = safe_divide(fe["bond_payment"], fe["estimated_monthly_rent"])
    fe["opex_to_rent"] = safe_divide(fe["operating_expenses"], fe["estimated_monthly_rent"])
    fe["ltv"] = safe_divide(fe["loan_amount"], fe["purchase_price"])
    fe["deposit_to_price"] = safe_divide(fe["deposit_amount"], fe["purchase_price"])
    fe["cash_invested_to_price"] = safe_divide(fe["total_cash_invested"], fe["purchase_price"])

    source_rank = {
        "suburb_property_type_bedrooms": 5,
        "suburb_bedrooms": 4,
        "suburb_property_type": 3,
        "suburb": 2,
        "user_input": 1,
    }
    fe["rent_source_strength"] = fe["rent_estimation_source"].map(source_rank).fillna(1)

    fe["suburb_listing_count"] = 1
    fe["city_listing_count"] = 1
    fe["property_type_listing_count"] = 1

    for col in ["floor_area_sqm", "land_area_sqm", "levies_clean", "rates_taxes_clean", "estimated_monthly_rent"]:
        fe[f"missing_{col}"] = fe[col].isna().astype(int)

    winsor_cols = [
        "price_per_sqm",
        "price_per_land_sqm",
        "rent_per_sqm",
        "rent_per_bedroom",
        "rent_per_bathroom",
        "rental_yield",
        "ROI",
        "DSCR",
        "bond_to_rent",
        "opex_to_rent",
    ]
    for col in winsor_cols:
        fe[f"{col}_win"] = winsorize_series(fe[col])

    for c in ["suburb", "city", "province", "property_type", "rent_estimation_source"]:
        if c in fe.columns:
            fe[c] = fe[c].astype("string")

    fe["monthly_bond_payment"] = fe["bond_payment"]
    fe["monthly_vacancy_cost"] = fe["gross_rent"] * (pd.to_numeric(fe["vacancy_pct"], errors="coerce") / 100.0)
    fe["monthly_management_fee"] = fe["management_fee"]
    fe["monthly_maintenance_cost"] = fe["maintenance"]
    fe["monthly_total_opex"] = fe["operating_expenses"]
    fe["monthly_noi"] = fe["noi"]
    fe["monthly_cash_flow"] = fe["cash_flow"]
    fe["gross_yield_pct"] = fe["rental_yield"] * 100.0
    fe["net_yield_pct"] = safe_divide((fe["noi"] * 12.0), fe["purchase_price"]) * 100.0
    fe["roi_pct"] = fe["ROI"] * 100.0
    fe["dscr"] = fe["DSCR"]

    return fe


def _align_to_feature_list(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    aligned = df.copy()
    feature_names = list(feature_names)

    categorical_defaults = {
        "suburb": "Unknown",
        "city": "Unknown",
        "province": "Unknown",
        "property_type": "Unknown",
        "rent_estimation_source": "user_input",
    }

    for col in feature_names:
        if col not in aligned.columns:
            if col.startswith("missing_"):
                aligned[col] = 1
            elif col in categorical_defaults:
                aligned[col] = categorical_defaults[col]
            else:
                aligned[col] = np.nan

    return aligned[feature_names].copy()


def build_single_property_features(payload: dict, feature_names: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = _normalize_payload(payload)
    raw_df = pd.DataFrame([normalized])

    business_df = _engineer_features(raw_df)

    direct_leakage_cols = [
        "estimated_monthly_rent",
        "gross_rent",
        "effective_rent",
        "management_fee",
        "maintenance",
        "insurance",
        "operating_expenses",
        "noi",
        "cash_flow",
        "rental_yield",
        "ROI",
        "DSCR",
    ]
    model_base_df = business_df.drop(columns=direct_leakage_cols, errors="ignore").copy()
    model_df = _align_to_feature_list(model_base_df, feature_names)

    return business_df, model_df


def coerce_batch_input(batch_df: pd.DataFrame, feature_names: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if batch_df.empty:
        raise ValueError("Uploaded CSV is empty.")

    renamed = batch_df.rename(columns={c: ALIASES.get(c, c) for c in batch_df.columns})
    for key, value in DEFAULTS.items():
        if key not in renamed.columns:
            renamed[key] = value

    business_df = _engineer_features(renamed)

    direct_leakage_cols = [
        "estimated_monthly_rent",
        "gross_rent",
        "effective_rent",
        "management_fee",
        "maintenance",
        "insurance",
        "operating_expenses",
        "noi",
        "cash_flow",
        "rental_yield",
        "ROI",
        "DSCR",
    ]
    model_base_df = business_df.drop(columns=direct_leakage_cols, errors="ignore").copy()
    model_df = _align_to_feature_list(model_base_df, feature_names)

    return business_df, model_df
