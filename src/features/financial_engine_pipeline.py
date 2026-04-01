from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ASSUMPTIONS = {
    "deposit_pct": 0.10,
    "annual_interest_rate": 0.11,
    "loan_term_years": 20,
    "transfer_cost_pct": 0.08,
    "occupancy_rate": 0.95,
    "maintenance_pct_of_rent": 0.05,
    "insurance_pct_of_price_annual": 0.003,
    "property_management_pct": 0.08,
}


def annuity_payment(principal: pd.Series, annual_rate: float, term_years: int) -> pd.Series:
    monthly_rate = annual_rate / 12
    n = term_years * 12
    return principal * (monthly_rate * (1 + monthly_rate) ** n) / (((1 + monthly_rate) ** n) - 1)


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return np.where((denominator == 0) | denominator.isna(), np.nan, numerator / denominator)


def choose_estimated_rent(df_engine: pd.DataFrame) -> pd.DataFrame:
    rent_candidates = [
        "rent_median_suburb_type_bed",
        "rent_median_suburb_bed",
        "rent_median_suburb_type",
        "rent_median_suburb",
    ]
    source_rank = {
        "rent_median_suburb_type_bed": "suburb_property_type_bedrooms",
        "rent_median_suburb_bed": "suburb_bedrooms",
        "rent_median_suburb_type": "suburb_property_type",
        "rent_median_suburb": "suburb",
    }
    df_engine["estimated_monthly_rent"] = np.nan
    df_engine["rent_estimation_source"] = np.nan
    for col in rent_candidates:
        if col not in df_engine.columns:
            continue
        mask = df_engine["estimated_monthly_rent"].isna() & df_engine[col].notna()
        df_engine.loc[mask, "estimated_monthly_rent"] = df_engine.loc[mask, col]
        df_engine.loc[mask, "rent_estimation_source"] = source_rank[col]
    return df_engine


def build_financial_engine(df_sales: pd.DataFrame, df_rentals: pd.DataFrame) -> pd.DataFrame:
    # fallback merge if the sales frame did not already receive rental benchmarks
    if "rent_median_suburb" not in df_sales.columns and not df_rentals.empty:
        rent_group_exact = df_rentals.groupby(["suburb", "property_type", "bedrooms"], dropna=False)["monthly_rent"].agg(["median", "mean", "count"]).reset_index()
        rent_group_exact = rent_group_exact.rename(columns={"median": "rent_median_suburb_type_bed", "mean": "rent_mean_suburb_type_bed", "count": "rent_count_suburb_type_bed"})
        df_engine = df_sales.merge(rent_group_exact, on=["suburb", "property_type", "bedrooms"], how="left")
    else:
        df_engine = df_sales.copy()

    df_engine = choose_estimated_rent(df_engine)

    for col in ["purchase_price", "levies", "rates_taxes"]:
        if col in df_engine.columns:
            df_engine[col] = pd.to_numeric(df_engine[col], errors="coerce")
    df_engine["purchase_price"] = pd.to_numeric(df_engine["purchase_price"], errors="coerce")
    df_engine["levies_clean"] = df_engine.get("levies", 0).fillna(0)
    df_engine["rates_taxes_clean"] = df_engine.get("rates_taxes", 0).fillna(0)

    df_engine["deposit_amount"] = df_engine["purchase_price"] * ASSUMPTIONS["deposit_pct"]
    df_engine["loan_amount"] = df_engine["purchase_price"] - df_engine["deposit_amount"]
    df_engine["transfer_costs"] = df_engine["purchase_price"] * ASSUMPTIONS["transfer_cost_pct"]
    df_engine["total_cash_invested"] = df_engine["deposit_amount"] + df_engine["transfer_costs"]
    df_engine["bond_payment"] = annuity_payment(df_engine["loan_amount"], ASSUMPTIONS["annual_interest_rate"], ASSUMPTIONS["loan_term_years"])

    df_engine["gross_rent"] = df_engine["estimated_monthly_rent"]
    df_engine["effective_rent"] = df_engine["gross_rent"] * ASSUMPTIONS["occupancy_rate"]
    df_engine["management_fee"] = df_engine["gross_rent"] * ASSUMPTIONS["property_management_pct"]
    df_engine["maintenance"] = df_engine["gross_rent"] * ASSUMPTIONS["maintenance_pct_of_rent"]
    df_engine["insurance"] = (df_engine["purchase_price"] * ASSUMPTIONS["insurance_pct_of_price_annual"]) / 12
    df_engine["operating_expenses"] = df_engine["levies_clean"] + df_engine["rates_taxes_clean"] + df_engine["management_fee"] + df_engine["maintenance"] + df_engine["insurance"]

    df_engine["noi"] = df_engine["effective_rent"] - df_engine["operating_expenses"]
    df_engine["cash_flow"] = df_engine["noi"] - df_engine["bond_payment"]
    df_engine["rental_yield"] = safe_divide(df_engine["gross_rent"] * 12, df_engine["purchase_price"])
    df_engine["ROI"] = safe_divide(df_engine["cash_flow"] * 12, df_engine["total_cash_invested"])
    df_engine["DSCR"] = safe_divide(df_engine["noi"], df_engine["bond_payment"])

    conditions = [
        (df_engine["cash_flow"] > 1000) & (df_engine["DSCR"] >= 1.2) & (df_engine["ROI"] >= 0.08),
        (df_engine["cash_flow"] >= 0) & (df_engine["DSCR"] >= 1.0),
    ]
    choices = ["Good", "Moderate"]
    df_engine["investment_label"] = np.select(conditions, choices, default="Bad")
    return df_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial engine pipeline")
    parser.add_argument("--sales-path", default="data/processed/eda_ready_sales_data.csv")
    parser.add_argument("--rentals-path", default="data/processed/eda_ready_rentals_data.csv")
    parser.add_argument("--output-path", default="data/interim/financial_engine_sales_dataset.csv")
    args = parser.parse_args()

    df_sales = pd.read_csv(Path(args.sales_path))
    df_rentals = pd.read_csv(Path(args.rentals_path)) if Path(args.rentals_path).exists() else pd.DataFrame()
    df_engine = build_financial_engine(df_sales, df_rentals)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_engine.to_csv(output_path, index=False)
    print(f"✅ Financial engine dataset saved to: {output_path}")
    print(df_engine["investment_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
