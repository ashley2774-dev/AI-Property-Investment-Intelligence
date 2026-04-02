from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd

# =========================
# Financial Logic (aligned to training)
# =========================

ASSUMPTIONS = {
    "deposit_pct": 0.10,
    "annual_interest_rate": 0.11,
    "loan_term_years": 20,
    "transfer_cost_pct": 0.08,
    "occupancy_rate": 0.95,
    "maintenance_pct": 0.05,
    "insurance_pct": 0.003,
    "management_pct": 0.08,
}


def annuity_payment(principal, annual_rate, term_years):
    monthly_rate = annual_rate / 12
    n = term_years * 12
    return principal * (monthly_rate * (1 + monthly_rate) ** n) / ((1 + monthly_rate) ** n - 1)


def safe_divide(n, d):
    return np.where((d == 0) | pd.isna(d), np.nan, n / d)


# =========================
# Winsorization
# =========================

def winsorize_series(s, lower=0.01, upper=0.99):
    return np.clip(s, s.quantile(lower), s.quantile(upper))


# =========================
# CORE PIPELINE
# =========================

def build_single_property_features(payload: dict, feature_names: Iterable[str]):

    df = pd.DataFrame([payload])

    # =========================
    # Financial Engine
    # =========================

    df["deposit_amount"] = df["purchase_price"] * (df["deposit_pct"] / 100)
    df["loan_amount"] = df["purchase_price"] - df["deposit_amount"]

    df["bond_payment"] = annuity_payment(
        df["loan_amount"],
        df["interest_rate_pct"] / 100,
        df["loan_term_years"],
    )

    df["gross_rent"] = df["estimated_rent"]
    df["effective_rent"] = df["gross_rent"] * (1 - df["vacancy_pct"] / 100)

    df["management_fee"] = df["gross_rent"] * (df["management_fee_pct"] / 100)
    df["maintenance"] = df["gross_rent"] * (df["maintenance_pct"] / 100)
    df["insurance"] = df["purchase_price"] * ASSUMPTIONS["insurance_pct"] / 12

    df["operating_expenses"] = (
        df["levy"]
        + df["rates_taxes"]
        + df["management_fee"]
        + df["maintenance"]
        + df["insurance"]
        + df["other_opex"]
    )

    df["noi"] = df["effective_rent"] - df["operating_expenses"]
    df["cash_flow"] = df["noi"] - df["bond_payment"]

    df["rental_yield"] = safe_divide(df["gross_rent"] * 12, df["purchase_price"])
    df["ROI"] = safe_divide(df["cash_flow"] * 12, df["deposit_amount"])
    df["DSCR"] = safe_divide(df["noi"], df["bond_payment"])

    # =========================
    # Feature Engineering (MATCH TRAINING)
    # =========================

    df["price_per_sqm"] = safe_divide(df["purchase_price"], df["floor_area_sqm"])
    df["rent_per_sqm"] = safe_divide(df["estimated_rent"], df["floor_area_sqm"])

    df["bond_to_rent"] = safe_divide(df["bond_payment"], df["estimated_rent"])
    df["opex_to_rent"] = safe_divide(df["operating_expenses"], df["estimated_rent"])

    df["rent_per_bedroom"] = safe_divide(df["estimated_rent"], df["bedrooms"])

    # Winsorized features (IMPORTANT)
    for col in ["rental_yield", "ROI", "DSCR", "bond_to_rent", "opex_to_rent"]:
        df[f"{col}_win"] = winsorize_series(df[col])

    # =========================
    # MODEL DATASET
    # =========================

    model_df = df.copy()

    # Align to model features
    for col in feature_names:
        if col not in model_df.columns:
            model_df[col] = np.nan

    model_df = model_df[feature_names]

    # =========================
    # BUSINESS DATASET (UI)
    # =========================

    business_df = df.copy()

    # Rename for UI compatibility
    business_df["monthly_cash_flow"] = business_df["cash_flow"]
    business_df["gross_yield_pct"] = business_df["rental_yield"] * 100
    business_df["net_yield_pct"] = business_df["rental_yield"] * 100
    business_df["roi_pct"] = business_df["ROI"] * 100
    business_df["dscr"] = business_df["DSCR"]

    return business_df, model_df


# =========================
# BATCH
# =========================

def coerce_batch_input(df: pd.DataFrame, feature_names: Iterable[str]):
    business_list = []
    model_list = []

    for _, row in df.iterrows():
        b, m = build_single_property_features(row.to_dict(), feature_names)
        business_list.append(b)
        model_list.append(m)

    business_df = pd.concat(business_list, ignore_index=True)
    model_df = pd.concat(model_list, ignore_index=True)

    return business_df, model_df
