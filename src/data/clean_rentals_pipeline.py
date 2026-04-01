from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean_rent(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).lower().strip()
    if "per month" in value or "pm" in value:
        value = value.replace("per month", "").replace("pm", "")
    if "k" in value:
        num = re.findall(r"[\d\.]+", value)
        return float(num[0]) * 1_000 if num else np.nan
    value = re.sub(r"[^\d\.]", "", value)
    return float(value) if value else np.nan


def clean_text(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().title()


def normalize_property_type(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower()
    if "townhouse" in x:
        return "Townhouse"
    if "apartment" in x or "flat" in x:
        return "Apartment"
    if "house" in x:
        return "House"
    return "Other"


def clean_rentals_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    if "monthly_rent" in df.columns:
        df["monthly_rent"] = df["monthly_rent"].apply(clean_rent)
    for col in ["suburb", "city", "province"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].apply(normalize_property_type)
    for col in ["bedrooms", "bathrooms", "parking", "parking_spaces", "garage", "rates_taxes", "levies", "floor_area_sqm", "land_area_sqm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "scraped_timestamp" in df.columns:
        df = df.sort_values("scraped_timestamp", ascending=False)
    if "listing_id" in df.columns:
        df = df.drop_duplicates(subset=["listing_id"], keep="first")

    df_clean = df.dropna(subset=[c for c in ["monthly_rent", "suburb", "city"] if c in df.columns]).copy()
    df_clean = df_clean[(df_clean["monthly_rent"] > 0) & (df_clean["monthly_rent"] < 100000)].copy()

    if "parking" in df_clean.columns and "parking_spaces" not in df_clean.columns:
        df_clean["parking_spaces"] = df_clean["parking"]

    rental_summary = (
        df_clean.groupby([c for c in ["suburb", "property_type", "bedrooms"] if c in df_clean.columns], dropna=False)["monthly_rent"]
        .agg(avg_rent="mean", median_rent="median", count="size")
        .reset_index()
    )
    rental_summary["avg_rent"] = rental_summary["avg_rent"].round(0)
    rental_summary["median_rent"] = rental_summary["median_rent"].round(0)

    assert (df_clean["monthly_rent"] > 0).all(), "Invalid rent values found"
    assert df_clean["suburb"].notnull().all(), "Missing suburb detected"
    return df_clean.reset_index(drop=True), rental_summary.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean rentals dataset")
    parser.add_argument("--input-path", default="data/raw/rentals/privateproperty_rentals_final.csv")
    parser.add_argument("--output-path", default="data/processed/clean_rentals_data.csv")
    parser.add_argument("--summary-path", default="data/processed/rental_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input_path))
    df_clean, rental_summary = clean_rentals_dataframe(df)
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    rental_summary.to_csv(summary_path, index=False)
    print(f"✅ Clean rentals data saved to: {output_path}")
    print(f"✅ Rental summary saved to: {summary_path}")
    print("Final shape:", df_clean.shape)


if __name__ == "__main__":
    main()
