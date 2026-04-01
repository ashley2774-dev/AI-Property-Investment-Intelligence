from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def build_amortization_schedule(
    assumptions: Dict[str, float | str],
    financials: Dict[str, float],
    years: int = 20,
) -> Tuple[List[Dict[str, float]], Optional[int]]:
    property_value = float(assumptions["purchase_price"])
    monthly_rent = float(assumptions["monthly_rent"])
    annual_growth = float(assumptions.get("annual_growth_pct", 6.0)) / 100
    annual_rent_growth = float(assumptions.get("annual_rent_growth_pct", 5.0)) / 100
    loan_balance = float(financials["loan_amount"])
    monthly_payment = float(financials["monthly_bond_payment"])
    monthly_rate = float(assumptions["interest_rate"]) / 100 / 12

    monthly_cash_flow = float(financials["monthly_cash_flow"])
    break_even_month: Optional[int] = 1 if monthly_cash_flow >= 0 else None

    rows: List[Dict[str, float]] = []

    current_value = property_value
    current_rent = monthly_rent
    current_balance = loan_balance
    month_counter = 0

    for year in range(1, years + 1):
        for _ in range(12):
            month_counter += 1
            interest = current_balance * monthly_rate
            principal = max(monthly_payment - interest, 0)
            current_balance = max(current_balance - principal, 0)
            if break_even_month is None and monthly_cash_flow >= 0:
                break_even_month = month_counter

        current_value *= (1 + annual_growth)
        current_rent *= (1 + annual_rent_growth)
        monthly_cash_flow = monthly_cash_flow + (current_rent * 0.92 - monthly_rent * 0.92) / 12

        equity = current_value - current_balance
        ltv = current_balance / current_value if current_value else 0.0
        rows.append(
            {
                "year": year,
                "property_value": current_value,
                "loan_balance": current_balance,
                "equity": equity,
                "ltv": ltv,
                "monthly_cash_flow": monthly_cash_flow,
            }
        )

        if break_even_month is None and monthly_cash_flow >= 0:
            break_even_month = month_counter

    return rows, break_even_month
