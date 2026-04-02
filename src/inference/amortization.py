from __future__ import annotations

import math
import pandas as pd

from .finance import monthly_bond_payment


def build_amortization_schedule(
    purchase_price: float,
    loan_amount: float,
    annual_interest_rate: float,
    loan_term_years: int,
    monthly_cash_flow: float,
    annual_growth_rate: float,
) -> pd.DataFrame:
    payment = monthly_bond_payment(loan_amount, annual_interest_rate, loan_term_years)
    monthly_rate = annual_interest_rate / 12.0
    balance = loan_amount
    records = []

    for year in range(0, loan_term_years + 1):
        if year == 0:
            property_value = purchase_price
            equity = property_value - balance
            cumulative_cash_flow = 0.0
        else:
            for _ in range(12):
                interest = balance * monthly_rate
                principal = max(payment - interest, 0.0)
                balance = max(balance - principal, 0.0)
            property_value = purchase_price * ((1.0 + annual_growth_rate) ** year)
            equity = property_value - balance
            cumulative_cash_flow = monthly_cash_flow * 12.0 * year

        records.append(
            {
                "year": year,
                "property_value": round(property_value, 2),
                "loan_balance": round(balance, 2),
                "equity": round(equity, 2),
                "cumulative_cash_flow": round(cumulative_cash_flow, 2),
                "total_equity_plus_cash_flow": round(equity + cumulative_cash_flow, 2),
            }
        )

    return pd.DataFrame(records)


def summarize_amortization(schedule: pd.DataFrame) -> dict:
    ending = schedule.iloc[-1].to_dict()
    positive_rows = schedule.loc[schedule["cumulative_cash_flow"] >= 0]
    break_even_year = int(positive_rows.iloc[0]["year"]) if not positive_rows.empty else None

    return {
        "ending_equity": float(ending["equity"]),
        "ending_property_value": float(ending["property_value"]),
        "break_even_year_display": "Already positive" if break_even_year == 0 else (str(break_even_year) if break_even_year is not None else "Not within model term"),
    }
