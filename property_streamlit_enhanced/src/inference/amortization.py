from __future__ import annotations

import pandas as pd

from .finance import monthly_bond_payment


def build_projection_table(
    payload: dict,
    projection_years: int,
    annual_property_growth_pct: float,
    annual_rent_growth_pct: float,
    annual_expense_growth_pct: float,
) -> pd.DataFrame:
    purchase_price = float(payload["purchase_price"])
    deposit_pct = float(payload["deposit_pct"]) / 100
    interest_rate = float(payload["interest_rate"])
    loan_term_years = int(payload["loan_term_years"])
    monthly_rent = float(payload["monthly_rent"])
    vacancy_pct = float(payload["vacancy_pct"]) / 100
    management_pct = float(payload["management_pct"]) / 100
    maintenance_pct = float(payload["maintenance_pct"]) / 100
    rates_taxes_monthly = float(payload["rates_taxes_monthly"])
    insurance_monthly = float(payload["insurance_monthly"])

    loan_amount = purchase_price * (1 - deposit_pct)
    monthly_payment = monthly_bond_payment(loan_amount, interest_rate, loan_term_years)
    monthly_rate = interest_rate / 100 / 12
    total_months = loan_term_years * 12

    property_growth = annual_property_growth_pct / 100
    rent_growth = annual_rent_growth_pct / 100
    expense_growth = annual_expense_growth_pct / 100

    annual_gross_rent_base = monthly_rent * 12
    fixed_costs_base = (rates_taxes_monthly + insurance_monthly) * 12

    rows = []
    current_balance = loan_amount

    for year in range(1, projection_years + 1):
        months_to_apply = min(12, max(total_months - (year - 1) * 12, 0))
        interest_paid = 0.0
        principal_paid = 0.0
        bond_cost = 0.0

        for _ in range(months_to_apply):
            if current_balance <= 0:
                break
            interest_component = current_balance * monthly_rate
            principal_component = min(monthly_payment - interest_component, current_balance)
            payment = interest_component + principal_component
            current_balance -= principal_component
            interest_paid += interest_component
            principal_paid += principal_component
            bond_cost += payment

        annual_gross_rent = annual_gross_rent_base * ((1 + rent_growth) ** (year - 1))
        effective_annual_rent = annual_gross_rent * (1 - vacancy_pct)
        management_annual = annual_gross_rent * management_pct
        maintenance_annual = annual_gross_rent * maintenance_pct
        fixed_costs_annual = fixed_costs_base * ((1 + expense_growth) ** (year - 1))
        annual_opex = management_annual + maintenance_annual + fixed_costs_annual
        annual_noi = effective_annual_rent - annual_opex
        annual_cash_flow = annual_noi - bond_cost
        property_value = purchase_price * ((1 + property_growth) ** year)
        equity = property_value - current_balance

        rows.append(
            {
                "year": year,
                "property_value": property_value,
                "loan_balance": max(current_balance, 0.0),
                "equity": equity,
                "interest_paid": interest_paid,
                "principal_paid": principal_paid,
                "annual_rent": annual_gross_rent,
                "annual_effective_rent": effective_annual_rent,
                "annual_opex": annual_opex,
                "annual_noi": annual_noi,
                "annual_bond_cost": bond_cost,
                "annual_cash_flow": annual_cash_flow,
            }
        )

    return pd.DataFrame(rows)


def summarize_projection(projection_df: pd.DataFrame) -> dict:
    positive_years = projection_df.loc[projection_df["annual_cash_flow"] > 0, "year"]
    cashflow_positive_year = int(positive_years.iloc[0]) if not positive_years.empty else None

    last_row = projection_df.iloc[-1]
    return {
        "cashflow_positive_year": cashflow_positive_year,
        "ending_property_value": float(last_row["property_value"]),
        "ending_balance": float(last_row["loan_balance"]),
        "ending_equity": float(last_row["equity"]),
    }
