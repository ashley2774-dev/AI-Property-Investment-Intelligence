from __future__ import annotations

from src.inference.finance import compute_financials


def build_features(payload: dict) -> tuple[dict, dict]:
    financials = compute_financials(payload)

    purchase_price = float(payload["purchase_price"])
    monthly_rent = float(payload["monthly_rent"])
    floor_area_sqm = float(payload["floor_area_sqm"])
    bedrooms = float(payload["bedrooms"])
    bathrooms = float(payload["bathrooms"])
    parking = float(payload["parking"])

    price_per_sqm = purchase_price / floor_area_sqm if floor_area_sqm > 0 else 0.0
    rent_per_sqm = monthly_rent / floor_area_sqm if floor_area_sqm > 0 else 0.0
    bond_to_rent = (
        financials["monthly_bond_payment"] / monthly_rent if monthly_rent > 0 else 0.0
    )
    opex_to_rent = (
        (financials["opex_annual"] / 12) / monthly_rent if monthly_rent > 0 else 0.0
    )

    features = {
        "province": payload["province"],
        "suburb": payload["suburb"].strip().title(),
        "property_type": payload["property_type"],
        "rent_estimation_source": payload["rent_estimation_source"],
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "floor_area_sqm": floor_area_sqm,
        "purchase_price": purchase_price,
        "monthly_rent": monthly_rent,
        "deposit_pct": float(payload["deposit_pct"]),
        "interest_rate": float(payload["interest_rate"]),
        "loan_term_years": float(payload["loan_term_years"]),
        "vacancy_pct": float(payload["vacancy_pct"]),
        "management_pct": float(payload["management_pct"]),
        "maintenance_pct": float(payload["maintenance_pct"]),
        "rates_taxes_monthly": float(payload["rates_taxes_monthly"]),
        "insurance_monthly": float(payload["insurance_monthly"]),
        "gross_yield": financials["gross_yield"],
        "net_yield": financials["net_yield"],
        "roi": financials["roi"],
        "dscr": financials["dscr"],
        "monthly_cash_flow": financials["monthly_cash_flow"],
        "price_per_sqm": price_per_sqm,
        "rent_per_sqm": rent_per_sqm,
        "bond_to_rent": bond_to_rent,
        "opex_to_rent": opex_to_rent,
    }

    return features, financials
