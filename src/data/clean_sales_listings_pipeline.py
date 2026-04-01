from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean_purchase_price(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).lower().strip()
    if "million" in value:
        num = re.findall(r"[\d\.]+", value)
        return float(num[0]) * 1_000_000 if num else np.nan
    if re.search(r"\bk\b", value):
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


def clean_sales_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    if "purchase_price" in df.columns:
        df["purchase_price"] = df["purchase_price"].apply(clean_purchase_price)
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

    critical_cols = [c for c in ["purchase_price", "suburb", "city"] if c in df.columns]
    df_clean = df.dropna(subset=critical_cols).copy()

    group_cols = [c for c in ["suburb", "city", "province", "property_type"] if c in df_clean.columns]
    for col in ["rates_taxes", "levies"]:
        if col not in df_clean.columns:
            continue
        if group_cols:
            median_grp = df_clean.groupby(group_cols)[col].transform("median")
            mean_grp = df_clean.groupby(group_cols)[col].transform("mean")
            df_clean[col] = df_clean[col].fillna(median_grp).fillna(mean_grp)
        for fallback_cols in [["suburb", "city", "province"], ["city", "province"], ["province"]]:
            fallback_cols = [c for c in fallback_cols if c in df_clean.columns]
            if fallback_cols:
                fallback_mean = df_clean.groupby(fallback_cols)[col].transform("mean")
                df_clean[col] = df_clean[col].fillna(fallback_mean)
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    if "purchase_price_Rands" not in df_clean.columns and "purchase_price" in df_clean.columns:
        df_clean["purchase_price_Rands"] = df_clean["purchase_price"]

    if "property_type" in df_clean.columns:
        df_clean["property_type"] = df_clean["property_type"].astype(str).str.strip()
    if "listing_id" in df_clean.columns and "property_type" in df_clean.columns:
        df_clean = df_clean[df_clean["listing_id"].notna() & df_clean["property_type"].notna() & (df_clean["property_type"] != "")].copy()

    if "parking" in df_clean.columns and "parking_spaces" not in df_clean.columns:
        df_clean["parking_spaces"] = df_clean["parking"]

    if "purchase_price_Rands" in df_clean.columns:
        assert (df_clean["purchase_price_Rands"] > 0).all(), "Negative prices found"
    assert df_clean["suburb"].notnull().all(), "Missing suburb values"
    return df_clean.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean sales listings dataset")
    parser.add_argument("--input-path", default="data/raw/listings/privateproperty_sales_final.csv")
    parser.add_argument("--output-path", default="data/processed/clean_sales_data.csv")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    df = pd.read_csv(input_path)
    df_clean = clean_sales_dataframe(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Clean sales data saved to: {output_path}")
    print("Final shape:", df_clean.shape)


if __name__ == "__main__":
    main()
