from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

DEFAULTS: Dict[str, float | str] = {
    "property_type": "Apartment",
    "bedrooms": 2.0,
    "bathrooms": 1.0,
    "parking": 1.0,
    "floor_area_sqm": 75.0,
    "purchase_price": 850000.0,
    "monthly_rent": 8500.0,
    "deposit_pct": 10.0,
    "interest_rate": 11.75,
    "loan_term_years": 20.0,
    "vacancy_pct": 5.0,
    "management_pct": 8.0,
    "maintenance_pct": 5.0,
    "rates_taxes_monthly": 1200.0,
    "insurance_monthly": 350.0,
    "annual_growth_pct": 6.0,
    "annual_rent_growth_pct": 5.0,
}

COLUMN_ALIASES = {
    "province": ["province"],
    "suburb": ["suburb"],
    "property_type": ["property_type"],
    "bedrooms": ["bedrooms", "bedroom", "beds"],
    "bathrooms": ["bathrooms", "bathroom", "baths"],
    "parking": ["parking", "parking_bays", "garages", "parking_spaces"],
    "floor_area_sqm": ["floor_area_sqm", "floor_area", "erf_size_sqm", "sqm"],
    "purchase_price": ["purchase_price", "purchase price", "purchase_price_rands", "price"],
    "monthly_rent": ["monthly_rent", "rent", "average_rent", "estimated_rent", "rent_estimate"],
    "vacancy_pct": ["vacancy_pct"],
    "management_pct": ["management_pct"],
    "maintenance_pct": ["maintenance_pct"],
    "rates_taxes_monthly": ["rates_taxes_monthly", "rates_and_taxes", "rates_taxes"],
    "insurance_monthly": ["insurance_monthly", "insurance"],
    "annual_growth_pct": ["annual_growth_pct", "property_growth_pct", "capital_growth_pct"],
    "annual_rent_growth_pct": ["annual_rent_growth_pct", "rent_growth_pct"],
}


def _normalise_text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _first_matching_column(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _load_csv_if_exists(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _mode_or_default(series: pd.Series, default):
    clean = series.dropna()
    if clean.empty:
        return default
    mode = clean.mode(dropna=True)
    if mode.empty:
        return default
    return mode.iloc[0]


def _median_or_default(series: pd.Series, default: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float(default)
    return float(vals.median())


def _extract_area_rows(df: pd.DataFrame, province: str, suburb: str) -> pd.DataFrame:
    if df.empty:
        return df
    province_col = _first_matching_column(df, COLUMN_ALIASES["province"])
    suburb_col = _first_matching_column(df, COLUMN_ALIASES["suburb"])
    if not province_col or not suburb_col:
        return pd.DataFrame()

    province_match = _normalise_text_series(df[province_col]) == province.strip().lower()
    suburb_match = _normalise_text_series(df[suburb_col]) == suburb.strip().lower()
    out = df.loc[province_match & suburb_match].copy()
    if not out.empty:
        return out

    out = df.loc[suburb_match].copy()
    if not out.empty:
        return out

    return df.loc[province_match].copy()


def build_location_assumptions(province: str, suburb: str) -> Dict[str, float | str]:
    sales = _load_csv_if_exists("financial_engine_feature_engineered_full.csv")
    if sales.empty:
        sales = _load_csv_if_exists("feature_engineered_dataset_v1.csv")
    if sales.empty:
        sales = _load_csv_if_exists("expanded_sales_data.csv")

    rental = _load_csv_if_exists("rental_summary.csv")
    if rental.empty:
        rental = _load_csv_if_exists("expanded_rentals_data.csv")

    sales_rows = _extract_area_rows(sales, province, suburb)
    rental_rows = _extract_area_rows(rental, province, suburb)

    assumptions: Dict[str, float | str] = {
        "province": province,
        "suburb": suburb,
        "rent_estimation_source": "area_benchmark",
    }

    for field, default in DEFAULTS.items():
        if field == "monthly_rent":
            source_df = rental_rows if not rental_rows.empty else sales_rows
        else:
            source_df = sales_rows

        col = _first_matching_column(source_df, COLUMN_ALIASES.get(field, [field])) if not source_df.empty else None
        if col is None:
            assumptions[field] = default
            continue

        if isinstance(default, str):
            assumptions[field] = _mode_or_default(source_df[col], default)
        else:
            assumptions[field] = _median_or_default(source_df[col], float(default))

    assumptions["data_match_level"] = (
        "suburb"
        if (not sales_rows.empty or not rental_rows.empty)
        else "province_or_global_default"
    )

    assumptions["suburb_sample_sales"] = int(len(sales_rows))
    assumptions["suburb_sample_rentals"] = int(len(rental_rows))

    assumptions["property_type"] = str(assumptions["property_type"]).title()
    return assumptions
