from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def slugify_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "-", regex=True)
        .str.strip("-")
    )


def expand_fast(df: pd.DataFrame, target_rows: int, dataset: str) -> pd.DataFrame:
    df = df.copy()
    if len(df) >= target_rows or len(df) == 0:
        return df.reset_index(drop=True)
    repeats = int(np.ceil(target_rows / len(df)))
    expanded = pd.concat([df] * repeats, ignore_index=True).head(target_rows).copy()
    expanded["expansion_source"] = dataset
    expanded["synthetic_row_flag"] = False
    expanded.loc[len(df):, "synthetic_row_flag"] = True
    if "listing_id" in expanded.columns:
        expanded["listing_id"] = expanded["listing_id"].astype("string") + "_" + expanded.index.astype(str)
    return expanded.reset_index(drop=True)


def build_suburb_rental_benchmarks(df_rentals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols_1 = [c for c in ["suburb", "property_type", "bedrooms"] if c in df_rentals.columns]
    group_cols_2 = [c for c in ["suburb", "bedrooms"] if c in df_rentals.columns]
    group_cols_3 = [c for c in ["suburb", "property_type"] if c in df_rentals.columns]

    exact = df_rentals.groupby(group_cols_1, dropna=False)["monthly_rent"].agg(["median", "mean", "count"]).reset_index()
    exact = exact.rename(columns={"median": "rent_median_suburb_type_bed", "mean": "rent_mean_suburb_type_bed", "count": "rent_count_suburb_type_bed"})

    by_bed = df_rentals.groupby(group_cols_2, dropna=False)["monthly_rent"].agg(["median", "mean", "count"]).reset_index()
    by_bed = by_bed.rename(columns={"median": "rent_median_suburb_bed", "mean": "rent_mean_suburb_bed", "count": "rent_count_suburb_bed"})

    by_type = df_rentals.groupby(group_cols_3, dropna=False)["monthly_rent"].agg(["median", "mean", "count"]).reset_index()
    by_type = by_type.rename(columns={"median": "rent_median_suburb_type", "mean": "rent_mean_suburb_type", "count": "rent_count_suburb_type"})
    return exact, by_bed, by_type


def merge_rental_signals(df_sales: pd.DataFrame, df_rentals: pd.DataFrame) -> pd.DataFrame:
    exact, by_bed, by_type = build_suburb_rental_benchmarks(df_rentals)
    out = df_sales.copy()
    if "suburb" in out.columns:
        out["suburb_slug"] = slugify_series(out["suburb"])
    if "suburb" in df_rentals.columns:
        df_rentals = df_rentals.copy()
        df_rentals["suburb_slug"] = slugify_series(df_rentals["suburb"])
        exact["suburb_slug"] = slugify_series(exact["suburb"])
        by_bed["suburb_slug"] = slugify_series(by_bed["suburb"])
        by_type["suburb_slug"] = slugify_series(by_type["suburb"])

    merge_keys_exact = [c for c in ["suburb_slug", "property_type", "bedrooms"] if c in out.columns and c in exact.columns]
    merge_keys_bed = [c for c in ["suburb_slug", "bedrooms"] if c in out.columns and c in by_bed.columns]
    merge_keys_type = [c for c in ["suburb_slug", "property_type"] if c in out.columns and c in by_type.columns]

    out = out.merge(exact.drop(columns=[c for c in ["suburb"] if c in exact.columns]), on=merge_keys_exact, how="left")
    out = out.merge(by_bed.drop(columns=[c for c in ["suburb"] if c in by_bed.columns]), on=merge_keys_bed, how="left")
    out = out.merge(by_type.drop(columns=[c for c in ["suburb"] if c in by_type.columns]), on=merge_keys_type, how="left")

    suburb_overall = df_rentals.groupby("suburb_slug", dropna=False)["monthly_rent"].agg(["median", "mean", "count"]).reset_index()
    suburb_overall = suburb_overall.rename(columns={"median": "rent_median_suburb", "mean": "rent_mean_suburb", "count": "rent_count_suburb"})
    out = out.merge(suburb_overall, on="suburb_slug", how="left")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand property dataset and merge rental signals")
    parser.add_argument("--sales-path", default="data/processed/clean_sales_data.csv")
    parser.add_argument("--rentals-path", default="data/processed/clean_rentals_data.csv")
    parser.add_argument("--target-rows", type=int, default=10000)
    parser.add_argument("--sales-output", default="data/processed/expanded_sales_data.csv")
    parser.add_argument("--rentals-output", default="data/processed/expanded_rentals_data.csv")
    args = parser.parse_args()

    df_sales = pd.read_csv(Path(args.sales_path))
    df_rentals = pd.read_csv(Path(args.rentals_path))

    expanded_sales = expand_fast(df_sales, args.target_rows, "sales")
    expanded_rentals = expand_fast(df_rentals, args.target_rows, "rentals")
    expanded_sales = merge_rental_signals(expanded_sales, expanded_rentals)

    sales_output = Path(args.sales_output)
    rentals_output = Path(args.rentals_output)
    sales_output.parent.mkdir(parents=True, exist_ok=True)
    expanded_sales.to_csv(sales_output, index=False)
    expanded_rentals.to_csv(rentals_output, index=False)
    print(f"✅ Expanded sales saved to: {sales_output}")
    print(f"✅ Expanded rentals saved to: {rentals_output}")
    print("Sales shape:", expanded_sales.shape)
    print("Rentals shape:", expanded_rentals.shape)


if __name__ == "__main__":
    main()
