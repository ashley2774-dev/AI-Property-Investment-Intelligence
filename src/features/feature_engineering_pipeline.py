from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return np.where((denominator == 0) | pd.isna(denominator), np.nan, numerator / denominator)


def winsorize_series(s, lower=0.01, upper=0.99):
    lower_q = s.quantile(lower)
    upper_q = s.quantile(upper)
    return np.clip(s, lower_q, upper_q)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fe = df.copy()
    fe["total_rooms_proxy"] = fe.get("bedrooms", 0).fillna(0) + fe.get("bathrooms", 0).fillna(0)
    fe["total_parking_proxy"] = fe.get("parking_spaces", 0).fillna(0) + fe.get("garage", 0).fillna(0)
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

    source_rank = {"suburb_property_type_bedrooms": 5, "suburb_bedrooms": 4, "suburb_property_type": 3, "suburb": 2}
    fe["rent_source_strength"] = fe["rent_estimation_source"].map(source_rank).fillna(1)

    suburb_listing_count = fe.groupby("suburb")["listing_id"].transform("count") if "listing_id" in fe.columns else 0
    city_listing_count = fe.groupby("city")["purchase_price"].transform("count") if "city" in fe.columns else 0
    property_type_count = fe.groupby("property_type")["purchase_price"].transform("count") if "property_type" in fe.columns else 0
    fe["suburb_listing_count"] = suburb_listing_count
    fe["city_listing_count"] = city_listing_count
    fe["property_type_listing_count"] = property_type_count

    for col in ["floor_area_sqm", "land_area_sqm", "levies_clean", "rates_taxes_clean", "estimated_monthly_rent"]:
        if col in fe.columns:
            fe[f"missing_{col}"] = fe[col].isna().astype(int)

    winsor_cols = ["price_per_sqm", "price_per_land_sqm", "rent_per_sqm", "rent_per_bedroom", "rent_per_bathroom", "rental_yield", "ROI", "DSCR", "bond_to_rent", "opex_to_rent"]
    for col in winsor_cols:
        if col in fe.columns:
            series = pd.Series(fe[col])
            fe[f"{col}_win"] = winsorize_series(series.dropna().reindex(series.index))

    cat_cols = ["suburb", "city", "province", "property_type", "rent_estimation_source", "investment_label"]
    for c in cat_cols:
        if c in fe.columns:
            fe[c] = fe[c].astype("string")

    full_engineered_df = fe.copy()
    target_col = "investment_label"
    direct_leakage_cols = [
        "estimated_monthly_rent", "gross_rent", "effective_rent", "management_fee", "maintenance", "insurance",
        "operating_expenses", "noi", "cash_flow", "rental_yield", "ROI", "DSCR",
    ]
    id_cols = ["source_site", "listing_id", "listing_url", "title", "description", "listing_date", "scraped_timestamp", "rental_url"]
    metadata_cols = ["expansion_source", "synthetic_row_flag"]
    remove_cols = [c for c in direct_leakage_cols + id_cols + metadata_cols if c in fe.columns]
    model_ready_df = fe.drop(columns=remove_cols, errors="ignore").copy()
    if target_col not in model_ready_df.columns and target_col in fe.columns:
        model_ready_df[target_col] = fe[target_col]
    return full_engineered_df, model_ready_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--input-path", default="data/interim/financial_engine_sales_dataset.csv")
    parser.add_argument("--full-output", default="data/processed/financial_engine_feature_engineered_full.csv")
    parser.add_argument("--model-output", default="data/processed/financial_engine_feature_engineered_model_ready.csv")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input_path))
    full_engineered_df, model_ready_df = engineer_features(df)
    full_output = Path(args.full_output)
    model_output = Path(args.model_output)
    full_output.parent.mkdir(parents=True, exist_ok=True)
    full_engineered_df.to_csv(full_output, index=False)
    model_ready_df.to_csv(model_output, index=False)
    print(f"✅ Full engineered dataset saved to: {full_output}")
    print(f"✅ Model-ready dataset saved to: {model_output}")
    print("Full shape:", full_engineered_df.shape)
    print("Model-ready shape:", model_ready_df.shape)


if __name__ == "__main__":
    main()
