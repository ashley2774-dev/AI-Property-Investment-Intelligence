from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.switch_backend("Agg")


def ensure_price_field(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    if "purchase_price_rands" in df.columns:
        df["purchase_price"] = df["purchase_price_rands"]
    df = df[df["purchase_price"].notna() & (df["purchase_price"] > 0)].copy()
    return df


def save_plot(fig_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def run_eda(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = ensure_price_field(df)
    if "suburb" in df.columns:
        df["suburb"] = df["suburb"].astype("string").str.strip().str.title()

    missing_df = pd.DataFrame({"missing_count": df.isnull().sum(), "missing_pct": (df.isnull().sum() / len(df)) * 100}).sort_values("missing_pct", ascending=False)
    missing_df.to_csv(output_dir / "sales_missingness_summary.csv")

    plt.figure(figsize=(10, 5))
    missing_df.head(20)["missing_pct"].sort_values(ascending=False).plot(kind="bar")
    plt.title("Missing Values (%)")
    save_plot(output_dir / "sales_missingness.png")

    plt.figure(figsize=(10, 5))
    df["purchase_price"].hist(bins=50)
    plt.title("Purchase Price Distribution")
    save_plot(output_dir / "sales_purchase_price_distribution.png")

    plt.figure(figsize=(10, 5))
    np.log1p(df["purchase_price"]).hist(bins=50)
    plt.title("Log Purchase Price Distribution")
    save_plot(output_dir / "sales_log_purchase_price_distribution.png")

    corr = df.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Matrix")
    save_plot(output_dir / "sales_correlation_matrix.png")

    if "suburb" in df.columns:
        suburb_price_summary = df.groupby("suburb", dropna=False).agg(listings=("suburb", "count"), avg_price=("purchase_price", "mean"), median_price=("purchase_price", "median")).reset_index().sort_values("listings", ascending=False)
        suburb_price_summary.to_csv(output_dir / "sales_suburb_price_summary.csv", index=False)

    if "bedrooms" in df.columns:
        bedroom_df = df[df["bedrooms"].notna()].copy()
        bedroom_price_summary = bedroom_df.groupby("bedrooms").agg(listings=("bedrooms", "count"), avg_price=("purchase_price", "mean"), median_price=("purchase_price", "median")).reset_index().sort_values("bedrooms")
        bedroom_price_summary.to_csv(output_dir / "sales_bedroom_price_summary.csv", index=False)

    if {"property_type", "purchase_price"}.issubset(df.columns):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="property_type", y="purchase_price")
        plt.xticks(rotation=30)
        plt.title("Purchase Price by Property Type")
        save_plot(output_dir / "sales_price_by_property_type.png")

        property_type_summary = df.groupby("property_type").agg(listings=("property_type", "count"), avg_price=("purchase_price", "mean")).reset_index().sort_values("listings", ascending=False)
        property_type_summary.to_csv(output_dir / "sales_property_type_summary.csv", index=False)

    eda_ready = df.copy()
    eda_ready.to_csv(output_dir.parent / "eda_ready_sales_data.csv", index=False)
    return eda_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA pipeline for expanded sales data")
    parser.add_argument("--input-path", default="data/processed/expanded_sales_data.csv")
    parser.add_argument("--output-dir", default="data/processed/eda_sales_outputs")
    args = parser.parse_args()
    df = pd.read_csv(Path(args.input_path))
    out = run_eda(df, Path(args.output_dir))
    print("✅ Sales EDA complete")
    print("EDA-ready shape:", out.shape)


if __name__ == "__main__":
    main()
