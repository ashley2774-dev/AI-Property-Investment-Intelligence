from __future__ import annotations

from typing import Dict


def monthly_bond_payment(loan_amount: float, annual_rate_pct: float, term_years: float) -> float:
    monthly_rate = annual_rate_pct / 100 / 12
    months = int(term_years * 12)
    if loan_amount <= 0:
        return 0.0
    if monthly_rate == 0:
        return loan_amount / max(months, 1)
    return loan_amount * (monthly_rate * (1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)


def compute_financials(assumptions: Dict[str, float | str]) -> Dict[str, float]:
    purchase_price = float(assumptions["purchase_price"])
    monthly_rent = float(assumptions["monthly_rent"])
    deposit_pct = float(assumptions["deposit_pct"])
    interest_rate = float(assumptions["interest_rate"])
    loan_term_years = float(assumptions["loan_term_years"])
    vacancy_pct = float(assumptions["vacancy_pct"])
    management_pct = float(assumptions["management_pct"])
    maintenance_pct = float(assumptions["maintenance_pct"])
    rates_taxes_monthly = float(assumptions["rates_taxes_monthly"])
    insurance_monthly = float(assumptions["insurance_monthly"])

    deposit_amount = purchase_price * deposit_pct / 100
    loan_amount = purchase_price - deposit_amount
    monthly_payment = monthly_bond_payment(loan_amount, interest_rate, loan_term_years)

    annual_gross_rent = monthly_rent * 12
    vacancy_allowance_annual = annual_gross_rent * vacancy_pct / 100
    effective_annual_rent = annual_gross_rent - vacancy_allowance_annual
    management_annual = annual_gross_rent * management_pct / 100
    maintenance_annual = annual_gross_rent * maintenance_pct / 100
    fixed_costs_annual = (rates_taxes_monthly + insurance_monthly) * 12
    opex_annual = management_annual + maintenance_annual + fixed_costs_annual
    annual_bond_cost = monthly_payment * 12
    annual_noi = effective_annual_rent - opex_annual
    annual_cash_flow = annual_noi - annual_bond_cost
    monthly_cash_flow = annual_cash_flow / 12

    gross_yield = annual_gross_rent / purchase_price if purchase_price else 0.0
    net_yield = annual_noi / purchase_price if purchase_price else 0.0
    roi = annual_cash_flow / deposit_amount if deposit_amount else 0.0
    dscr = annual_noi / annual_bond_cost if annual_bond_cost else 0.0

    return {
        "deposit_amount": deposit_amount,
        "loan_amount": loan_amount,
        "monthly_bond_payment": monthly_payment,
        "annual_gross_rent": annual_gross_rent,
        "vacancy_allowance_annual": vacancy_allowance_annual,
        "effective_annual_rent": effective_annual_rent,
        "management_annual": management_annual,
        "maintenance_annual": maintenance_annual,
        "fixed_costs_annual": fixed_costs_annual,
        "opex_annual": opex_annual,
        "annual_bond_cost": annual_bond_cost,
        "annual_net_operating_income": annual_noi,
        "annual_cash_flow": annual_cash_flow,
        "monthly_cash_flow": monthly_cash_flow,
        "gross_yield": gross_yield,
        "net_yield": net_yield,
        "roi": roi,
        "dscr": dscr,
    }
