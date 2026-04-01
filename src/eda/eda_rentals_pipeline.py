from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.switch_backend("Agg")


def detect_outliers_iqr(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def cap_outliers(series: pd.Series) -> pd.Series:
    lower, upper = detect_outliers_iqr(series)
    return series.clip(lower=lower, upper=upper)


def run_eda(df_rentals: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    df_rentals = df_rentals.copy()
    df_rentals = df_rentals[df_rentals["monthly_rent"].notna() & (df_rentals["monthly_rent"] > 0)].copy()

    plt.figure(figsize=(10, 5))
    df_rentals["monthly_rent"].hist(bins=50)
    plt.title("Monthly Rent Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "rentals_monthly_rent_distribution.png")
    plt.close()

    df_rentals["log_rent"] = np.log1p(df_rentals["monthly_rent"])
    plt.figure(figsize=(10, 5))
    df_rentals["log_rent"].hist(bins=50)
    plt.title("Log Rent Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "rentals_log_rent_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df_rentals["monthly_rent"])
    plt.title("Monthly Rent Boxplot")
    plt.tight_layout()
    plt.savefig(output_dir / "rentals_boxplot.png")
    plt.close()

    suburb_rent_summary = df_rentals.groupby("suburb").agg(listings=("monthly_rent", "count"), avg_rent=("monthly_rent", "mean"), median_rent=("monthly_rent", "median")).reset_index().sort_values("listings", ascending=False)
    suburb_rent_summary.to_csv(output_dir / "rentals_suburb_summary.csv", index=False)

    lower_rent, upper_rent = detect_outliers_iqr(df_rentals["monthly_rent"])
    df_rentals_clean = df_rentals[(df_rentals["monthly_rent"] >= lower_rent) & (df_rentals["monthly_rent"] <= upper_rent)].copy()
    df_rentals["monthly_rent_capped"] = cap_outliers(df_rentals["monthly_rent"])

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df_rentals["monthly_rent_capped"])
    plt.title("Capped Monthly Rent")
    plt.tight_layout()
    plt.savefig(output_dir / "rentals_capped_boxplot.png")
    plt.close()

    df_rentals.to_csv(output_dir.parent / "eda_ready_rentals_data.csv", index=False)
    return df_rentals_clean


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA pipeline for expanded rentals data")
    parser.add_argument("--input-path", default="data/processed/expanded_rentals_data.csv")
    parser.add_argument("--output-dir", default="data/processed/eda_rentals_outputs")
    args = parser.parse_args()
    df = pd.read_csv(Path(args.input_path))
    cleaned = run_eda(df, Path(args.output_dir))
    print("✅ Rentals EDA complete")
    print("IQR-filtered shape:", cleaned.shape)


if __name__ == "__main__":
    main()
