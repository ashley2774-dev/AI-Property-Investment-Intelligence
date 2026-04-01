from __future__ import annotations


def monthly_bond_payment(principal: float, annual_rate_pct: float, term_years: int) -> float:
    monthly_rate = annual_rate_pct / 100 / 12
    n_payments = term_years * 12
    if principal <= 0:
        return 0.0
    if monthly_rate == 0:
        return principal / n_payments
    factor = (1 + monthly_rate) ** n_payments
    return principal * (monthly_rate * factor) / (factor - 1)


def compute_financials(payload: dict) -> dict:
    purchase_price = float(payload["purchase_price"])
    monthly_rent = float(payload["monthly_rent"])
    deposit_pct = float(payload["deposit_pct"]) / 100
    interest_rate = float(payload["interest_rate"])
    loan_term_years = int(payload["loan_term_years"])
    vacancy_pct = float(payload["vacancy_pct"]) / 100
    management_pct = float(payload["management_pct"]) / 100
    maintenance_pct = float(payload["maintenance_pct"]) / 100
    rates_taxes_monthly = float(payload["rates_taxes_monthly"])
    insurance_monthly = float(payload["insurance_monthly"])

    deposit_amount = purchase_price * deposit_pct
    loan_amount = purchase_price - deposit_amount
    bond_payment = monthly_bond_payment(loan_amount, interest_rate, loan_term_years)

    annual_gross_rent = monthly_rent * 12
    vacancy_allowance_annual = annual_gross_rent * vacancy_pct
    effective_annual_rent = annual_gross_rent - vacancy_allowance_annual

    management_annual = annual_gross_rent * management_pct
    maintenance_annual = annual_gross_rent * maintenance_pct
    fixed_costs_annual = (rates_taxes_monthly + insurance_monthly) * 12
    opex_annual = management_annual + maintenance_annual + fixed_costs_annual

    annual_bond_cost = bond_payment * 12
    annual_net_operating_income = effective_annual_rent - opex_annual
    annual_cash_flow = annual_net_operating_income - annual_bond_cost
    monthly_cash_flow = annual_cash_flow / 12

    gross_yield = annual_gross_rent / purchase_price if purchase_price else 0.0
    net_yield = annual_net_operating_income / purchase_price if purchase_price else 0.0
    roi = annual_cash_flow / deposit_amount if deposit_amount > 0 else 0.0
    dscr = annual_net_operating_income / annual_bond_cost if annual_bond_cost > 0 else 0.0

    return {
        "deposit_amount": deposit_amount,
        "loan_amount": loan_amount,
        "monthly_bond_payment": bond_payment,
        "annual_gross_rent": annual_gross_rent,
        "vacancy_allowance_annual": vacancy_allowance_annual,
        "effective_annual_rent": effective_annual_rent,
        "management_annual": management_annual,
        "maintenance_annual": maintenance_annual,
        "fixed_costs_annual": fixed_costs_annual,
        "opex_annual": opex_annual,
        "annual_bond_cost": annual_bond_cost,
        "annual_net_operating_income": annual_net_operating_income,
        "annual_cash_flow": annual_cash_flow,
        "monthly_cash_flow": monthly_cash_flow,
        "gross_yield": gross_yield,
        "net_yield": net_yield,
        "roi": roi,
        "dscr": dscr,
    }
