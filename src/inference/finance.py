from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FinanceOutputs:
    purchase_price: float
    estimated_rent: float
    loan_amount: float
    deposit_amount: float
    monthly_bond_payment: float
    monthly_vacancy_cost: float
    monthly_management_fee: float
    monthly_maintenance_cost: float
    monthly_total_opex: float
    monthly_noi: float
    monthly_cash_flow: float
    gross_yield_pct: float
    net_yield_pct: float
    roi_pct: float
    dscr: float
    bond_to_rent: float
    opex_to_rent: float
    price_per_sqm: float
    rent_per_sqm: float
    ltv: float


def monthly_bond_payment(loan_amount: float, annual_rate: float, years: int) -> float:
    monthly_rate = annual_rate / 12.0
    periods = years * 12
    if loan_amount <= 0:
        return 0.0
    if monthly_rate == 0:
        return loan_amount / periods
    return loan_amount * (monthly_rate * (1 + monthly_rate) ** periods) / (((1 + monthly_rate) ** periods) - 1)


def calculate_financials(payload: dict) -> FinanceOutputs:
    purchase_price = float(payload.get("purchase_price", 0))
    estimated_rent = float(payload.get("estimated_rent", 0))
    floor_area_sqm = max(float(payload.get("floor_area_sqm", 0) or 0), 1.0)
    levy = float(payload.get("levy", 0))
    rates_taxes = float(payload.get("rates_taxes", 0))
    insurance = float(payload.get("insurance", 0))
    other_opex = float(payload.get("other_opex", 0))
    deposit_pct = float(payload.get("deposit_pct", 0)) / 100.0
    interest_rate = float(payload.get("interest_rate_pct", 0)) / 100.0
    vacancy_pct = float(payload.get("vacancy_pct", 0)) / 100.0
    management_fee_pct = float(payload.get("management_fee_pct", 0)) / 100.0
    maintenance_pct = float(payload.get("maintenance_pct", 0)) / 100.0
    loan_term_years = int(payload.get("loan_term_years", 20))

    deposit_amount = purchase_price * deposit_pct
    loan_amount = max(purchase_price - deposit_amount, 0.0)
    bond_payment = monthly_bond_payment(loan_amount, interest_rate, loan_term_years)

    vacancy_cost = estimated_rent * vacancy_pct
    management_fee = estimated_rent * management_fee_pct
    maintenance_cost = estimated_rent * maintenance_pct
    total_opex = levy + rates_taxes + insurance + other_opex + vacancy_cost + management_fee + maintenance_cost
    noi = estimated_rent - total_opex
    cash_flow = noi - bond_payment

    gross_yield = ((estimated_rent * 12.0) / purchase_price * 100.0) if purchase_price else 0.0
    net_yield = ((noi * 12.0) / purchase_price * 100.0) if purchase_price else 0.0
    roi = ((cash_flow * 12.0) / max(deposit_amount, 1.0) * 100.0) if deposit_amount else 0.0
    dscr = (noi / bond_payment) if bond_payment else 0.0
    bond_to_rent = (bond_payment / estimated_rent) if estimated_rent else 0.0
    opex_to_rent = (total_opex / estimated_rent) if estimated_rent else 0.0
    price_per_sqm = purchase_price / floor_area_sqm if floor_area_sqm else 0.0
    rent_per_sqm = estimated_rent / floor_area_sqm if floor_area_sqm else 0.0
    ltv = (loan_amount / purchase_price) if purchase_price else 0.0

    return FinanceOutputs(
        purchase_price=purchase_price,
        estimated_rent=estimated_rent,
        loan_amount=loan_amount,
        deposit_amount=deposit_amount,
        monthly_bond_payment=bond_payment,
        monthly_vacancy_cost=vacancy_cost,
        monthly_management_fee=management_fee,
        monthly_maintenance_cost=maintenance_cost,
        monthly_total_opex=total_opex,
        monthly_noi=noi,
        monthly_cash_flow=cash_flow,
        gross_yield_pct=gross_yield,
        net_yield_pct=net_yield,
        roi_pct=roi,
        dscr=dscr,
        bond_to_rent=bond_to_rent,
        opex_to_rent=opex_to_rent,
        price_per_sqm=price_per_sqm,
        rent_per_sqm=rent_per_sqm,
        ltv=ltv,
    )
