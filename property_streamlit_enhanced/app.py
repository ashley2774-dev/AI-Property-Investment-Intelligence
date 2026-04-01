from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference.amortization import build_projection_table, summarize_projection
from src.inference.predict import evaluate_deal
from src.inference.rules import build_recommendation

st.set_page_config(page_title="Property Investment Screener", layout="wide")

PROVINCES = [
    "Gauteng",
    "Western Cape",
    "KwaZulu-Natal",
    "Eastern Cape",
    "Free State",
    "Limpopo",
    "Mpumalanga",
    "North West",
    "Northern Cape",
]

PROPERTY_TYPES = ["Apartment", "House", "Townhouse", "Cluster", "Duplex", "Studio", "Other"]
RENT_SOURCES = ["user_input", "suburb_average", "listing_average", "agent_estimate", "unknown"]


def money(value: float) -> str:
    return f"R {value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.2%}"


def badge_html(text: str, bg: str, fg: str = "#111827") -> str:
    return (
        f"<div style='display:inline-block;padding:0.35rem 0.7rem;border-radius:999px;"
        f"background:{bg};color:{fg};font-weight:600;margin-bottom:0.5rem'>{text}</div>"
    )


def recommendation_color(recommendation: str) -> tuple[str, str]:
    if recommendation == "Proceed":
        return "#DCFCE7", "#166534"
    if recommendation == "Review Carefully":
        return "#FEF3C7", "#92400E"
    return "#FEE2E2", "#991B1B"


st.title("South Africa Property Investment Screener")
st.caption("Financed buy-to-let screening MVP")

with st.sidebar:
    st.header("About")
    st.write(
        "This MVP screens residential buy-to-let opportunities using the same investment "
        "logic as the modelling workflow. It turns a deal into an investor-readable summary, "
        "financial breakdown, and forward-looking equity view."
    )
    st.info(
        "This is a first-pass screening tool for decision support. It is not legal, tax, credit, or investment advice."
    )

    st.header("Projection assumptions")
    projection_years = st.slider("Projection horizon (years)", min_value=5, max_value=30, value=20)
    annual_property_growth_pct = st.slider("Annual property growth (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.5)
    annual_rent_growth_pct = st.slider("Annual rent growth (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    annual_expense_growth_pct = st.slider("Annual operating cost growth (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

st.subheader("Deal Inputs")
col1, col2, col3 = st.columns(3)

with col1:
    province = st.selectbox("Province", PROVINCES)
    suburb = st.text_input("Suburb", value="Randburg")
    property_type = st.selectbox("Property Type", PROPERTY_TYPES)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=1)
    parking = st.number_input("Parking Bays", min_value=0, max_value=20, value=1)

with col2:
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=10.0, max_value=5000.0, value=75.0)
    purchase_price = st.number_input("Purchase Price (R)", min_value=50000.0, value=850000.0, step=10000.0)
    deposit_pct = st.slider("Deposit (%)", min_value=0, max_value=80, value=10)
    interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=11.75, step=0.1)
    loan_term_years = st.number_input("Loan Term (years)", min_value=5, max_value=40, value=20)
    monthly_rent = st.number_input("Expected Monthly Rent (R)", min_value=1000.0, value=8500.0, step=100.0)

with col3:
    vacancy_pct = st.slider("Vacancy Allowance (%)", min_value=0, max_value=30, value=5)
    management_pct = st.slider("Management Fee (%)", min_value=0, max_value=20, value=8)
    maintenance_pct = st.slider("Maintenance Allowance (%)", min_value=0, max_value=20, value=5)
    rates_taxes_monthly = st.number_input("Rates & Taxes / Levies Monthly (R)", min_value=0.0, value=1200.0, step=100.0)
    insurance_monthly = st.number_input("Insurance Monthly (R)", min_value=0.0, value=350.0, step=50.0)
    rent_estimation_source = st.selectbox("Rent Estimation Source", RENT_SOURCES)

submitted = st.button("Evaluate Deal", type="primary")

if submitted:
    input_payload = {
        "province": province,
        "suburb": suburb,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "floor_area_sqm": floor_area_sqm,
        "purchase_price": purchase_price,
        "deposit_pct": deposit_pct,
        "interest_rate": interest_rate,
        "loan_term_years": loan_term_years,
        "monthly_rent": monthly_rent,
        "vacancy_pct": vacancy_pct,
        "management_pct": management_pct,
        "maintenance_pct": maintenance_pct,
        "rates_taxes_monthly": rates_taxes_monthly,
        "insurance_monthly": insurance_monthly,
        "rent_estimation_source": rent_estimation_source,
    }

    result = evaluate_deal(input_payload)
    recommendation = build_recommendation(result)
    projection_df = build_projection_table(
        payload=input_payload,
        projection_years=projection_years,
        annual_property_growth_pct=annual_property_growth_pct,
        annual_rent_growth_pct=annual_rent_growth_pct,
        annual_expense_growth_pct=annual_expense_growth_pct,
    )
    projection_summary = summarize_projection(projection_df)
    fin = result["financials"]

    st.divider()
    bg, fg = recommendation_color(recommendation["recommendation"])
    st.markdown(badge_html(recommendation["recommendation"], bg=bg, fg=fg), unsafe_allow_html=True)
    st.header("Investment Recommendation")
    st.write(recommendation["summary"])

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Predicted Class", result["prediction"].title())
    top2.metric("Confidence", f"{result['confidence']:.0%}")
    top3.metric("Monthly Cash Flow", money(fin["monthly_cash_flow"]))
    top4.metric("DSCR", f"{fin['dscr']:.2f}")

    prob_df = pd.DataFrame(
        {
            "investment_label": list(result["probabilities"].keys()),
            "probability": list(result["probabilities"].values()),
        }
    ).sort_values("probability", ascending=False)

    tab_summary, tab_financials, tab_amort = st.tabs(["Summary", "Financials", "Amortization & equity"])

    with tab_summary:
        s1, s2 = st.columns([1.15, 1])
        with s1:
            st.subheader("Deal summary")
            summary_df = pd.DataFrame(
                {
                    "Item": [
                        "Province",
                        "Suburb",
                        "Property type",
                        "Purchase price",
                        "Monthly rent",
                        "Deposit",
                        "Loan amount",
                        "Bond payment",
                    ],
                    "Value": [
                        province,
                        suburb.title(),
                        property_type,
                        money(float(purchase_price)),
                        money(float(monthly_rent)),
                        f"{deposit_pct:.0f}% ({money(fin['deposit_amount'])})",
                        money(fin["loan_amount"]),
                        money(fin["monthly_bond_payment"]),
                    ],
                }
            )
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

            st.subheader("Class probabilities")
            st.bar_chart(prob_df.set_index("investment_label"), use_container_width=True)

        with s2:
            st.subheader("Green flags")
            for item in recommendation["positives"]:
                st.success(item)

            st.subheader("Red flags")
            for item in recommendation["risks"]:
                st.error(item)

            st.subheader("Forward view")
            if projection_summary["cashflow_positive_year"] is None:
                st.warning(
                    f"Cash flow stays negative through year {projection_years} under the current projection assumptions."
                )
            elif projection_summary["cashflow_positive_year"] == 1:
                st.success("Cash flow is already positive in year 1 under the current assumptions.")
            else:
                st.info(
                    f"Cash flow is projected to turn positive in year {projection_summary['cashflow_positive_year']}."
                )

            st.write(
                f"Projected property value in year {projection_years}: {money(projection_summary['ending_property_value'])}."
            )
            st.write(
                f"Projected investor equity in year {projection_years}: {money(projection_summary['ending_equity'])}."
            )

    with tab_financials:
        st.subheader("Financials")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Gross Yield", pct(fin["gross_yield"]))
        r2.metric("Net Yield", pct(fin["net_yield"]))
        r3.metric("ROI", pct(fin["roi"]))
        r4.metric("DSCR", f"{fin['dscr']:.2f}")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Annual Gross Rent", money(fin["annual_gross_rent"]))
        r6.metric("Annual NOI", money(fin["annual_net_operating_income"]))
        r7.metric("Annual Bond Cost", money(fin["annual_bond_cost"]))
        r8.metric("Annual Cash Flow", money(fin["annual_cash_flow"]))

        fin_table = pd.DataFrame(
            {
                "Metric": [
                    "Deposit amount",
                    "Loan amount",
                    "Monthly bond payment",
                    "Annual gross rent",
                    "Vacancy allowance annual",
                    "Effective annual rent",
                    "Management annual",
                    "Maintenance annual",
                    "Fixed costs annual",
                    "Operating expenses annual",
                    "Annual net operating income",
                    "Annual bond cost",
                    "Annual cash flow",
                    "Monthly cash flow",
                ],
                "Value": [
                    money(fin["deposit_amount"]),
                    money(fin["loan_amount"]),
                    money(fin["monthly_bond_payment"]),
                    money(fin["annual_gross_rent"]),
                    money(fin["vacancy_allowance_annual"]),
                    money(fin["effective_annual_rent"]),
                    money(fin["management_annual"]),
                    money(fin["maintenance_annual"]),
                    money(fin["fixed_costs_annual"]),
                    money(fin["opex_annual"]),
                    money(fin["annual_net_operating_income"]),
                    money(fin["annual_bond_cost"]),
                    money(fin["annual_cash_flow"]),
                    money(fin["monthly_cash_flow"]),
                ],
            }
        )
        st.dataframe(fin_table, hide_index=True, use_container_width=True)

    with tab_amort:
        st.subheader("Amortization & equity")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Equity in final year", money(projection_summary["ending_equity"]))
        a2.metric("Property value in final year", money(projection_summary["ending_property_value"]))
        a3.metric("Outstanding balance in final year", money(projection_summary["ending_balance"]))
        positive_year_label = (
            "Not within horizon"
            if projection_summary["cashflow_positive_year"] is None
            else str(projection_summary["cashflow_positive_year"])
        )
        a4.metric("Cash flow positive by year", positive_year_label)

        chart_df = projection_df.set_index("year")[["property_value", "loan_balance", "equity", "annual_cash_flow"]]
        st.line_chart(chart_df, use_container_width=True)

        display_cols = [
            "year",
            "property_value",
            "loan_balance",
            "equity",
            "annual_rent",
            "annual_opex",
            "annual_bond_cost",
            "annual_cash_flow",
        ]
        display_df = projection_df[display_cols].copy()
        for col in display_cols[1:]:
            display_df[col] = display_df[col].map(lambda x: round(float(x), 2))
        st.dataframe(display_df, hide_index=True, use_container_width=True)
else:
    st.info("Complete the form and click Evaluate Deal to screen the property.")
