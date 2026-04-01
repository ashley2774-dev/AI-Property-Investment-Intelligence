from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.inference.predict import evaluate_location

st.set_page_config(
    page_title="AI Property Investment Screener",
    page_icon="🏘️",
    layout="wide",
)

st.title("AI Property Investment Screener")
st.caption("South Africa buy-to-let screening MVP")

with st.sidebar:
    st.header("About")
    st.write(
        "This MVP automates a first-pass financed buy-to-let screen from area-level "
        "benchmarks built from the project pipeline. The user only selects a province "
        "and suburb; the app then estimates the likely deal profile, rebuilds the financial "
        "logic, scores the opportunity, and returns an investor-readable recommendation."
    )
    st.info(
        "This is a screening tool for fast deal triage. It does not replace legal, tax, "
        "valuation, or credit advice."
    )

    st.subheader("Assumptions used")
    st.write(
        "Where a direct trained model artifact is not available, the app falls back to a "
        "transparent scoring layer built on the same financial logic used during modelling."
    )

st.subheader("Location inputs")
col1, col2 = st.columns(2)
with col1:
    province = st.selectbox(
        "Province",
        [
            "Gauteng",
            "Western Cape",
            "KwaZulu-Natal",
            "Eastern Cape",
            "Free State",
            "Limpopo",
            "Mpumalanga",
            "North West",
            "Northern Cape",
        ],
        index=0,
    )
with col2:
    suburb = st.text_input("Suburb", value="Randburg")

clicked = st.button("Screen suburb opportunity", type="primary")

if clicked:
    result = evaluate_location(province=province, suburb=suburb)
    rec = result["recommendation"]
    fin = result["financials"]
    assumptions = result["assumptions"]
    amort = pd.DataFrame(result["amortization_schedule"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommendation", rec["label"])
    c2.metric("Model confidence", f"{result['confidence']:.0%}")
    c3.metric("Estimated monthly cash flow", f"R {fin['monthly_cash_flow']:,.0f}")
    c4.metric("Estimated DSCR", f"{fin['dscr']:.2f}")

    tabs = st.tabs([
        "Summary",
        "Financials",
        "Amortization & equity",
        "Area assumptions",
    ])

    with tabs[0]:
        st.subheader("Recommendation summary")
        st.write(rec["summary"])

        a, b = st.columns(2)
        with a:
            st.markdown("**Green flags**")
            for item in rec["green_flags"]:
                st.write(f"- {item}")
        with b:
            st.markdown("**Red flags**")
            for item in rec["red_flags"]:
                st.write(f"- {item}")

        st.markdown("**Next action**")
        st.write(rec["next_action"])

        st.subheader("Probability view")
        probs = pd.DataFrame(
            {
                "class": list(result["probabilities"].keys()),
                "probability": list(result["probabilities"].values()),
            }
        )
        probs["probability"] = probs["probability"].map(lambda x: f"{x:.0%}")
        st.dataframe(probs, use_container_width=True, hide_index=True)

    with tabs[1]:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estimated purchase price", f"R {assumptions['purchase_price']:,.0f}")
        m2.metric("Estimated monthly rent", f"R {assumptions['monthly_rent']:,.0f}")
        m3.metric("Bond repayment", f"R {fin['monthly_bond_payment']:,.0f}")
        m4.metric("Break-even month", result["cashflow_break_even_month_label"])

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Gross yield", f"{fin['gross_yield']:.2%}")
        m6.metric("Net yield", f"{fin['net_yield']:.2%}")
        m7.metric("ROI", f"{fin['roi']:.2%}")
        m8.metric("Loan amount", f"R {fin['loan_amount']:,.0f}")

        financial_table = pd.DataFrame(
            {
                "Metric": [
                    "Deposit amount",
                    "Annual gross rent",
                    "Vacancy allowance annual",
                    "Operating expenses annual",
                    "Annual bond cost",
                    "Annual NOI",
                    "Annual cash flow",
                ],
                "Value": [
                    fin["deposit_amount"],
                    fin["annual_gross_rent"],
                    fin["vacancy_allowance_annual"],
                    fin["opex_annual"],
                    fin["annual_bond_cost"],
                    fin["annual_net_operating_income"],
                    fin["annual_cash_flow"],
                ],
            }
        )
        financial_table["Value"] = financial_table["Value"].map(lambda x: f"R {x:,.0f}")
        st.dataframe(financial_table, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Equity growth and loan amortization")
        if not amort.empty:
            st.line_chart(
                amort.set_index("year")[["property_value", "loan_balance", "equity"]],
                use_container_width=True,
            )
            st.line_chart(
                amort.set_index("year")[["monthly_cash_flow"]],
                use_container_width=True,
            )

            show_cols = [
                "year",
                "property_value",
                "loan_balance",
                "equity",
                "ltv",
                "monthly_cash_flow",
            ]
            display = amort[show_cols].copy()
            money_cols = ["property_value", "loan_balance", "equity", "monthly_cash_flow"]
            for c in money_cols:
                display[c] = display[c].map(lambda x: f"R {x:,.0f}")
            display["ltv"] = display["ltv"].map(lambda x: f"{x:.1%}")
            st.dataframe(display, use_container_width=True, hide_index=True)

            if result["cashflow_break_even_month"] is None:
                st.warning(
                    "Under the current area-level assumptions, monthly cash flow does not turn positive "
                    "within the projection horizon."
                )
            else:
                st.success(
                    f"Projected monthly cash flow turns positive in month {result['cashflow_break_even_month']} "
                    f"({result['cashflow_break_even_month_label']})."
                )

    with tabs[3]:
        st.subheader("Auto-filled area assumptions")
        area_df = pd.DataFrame([assumptions]).T.reset_index()
        area_df.columns = ["Field", "Value"]
        st.dataframe(area_df, use_container_width=True, hide_index=True)
        st.caption(
            "These values are inferred from processed project data where available, then filled with conservative defaults."
        )
else:
    st.info("Select a province and suburb, then click the button to generate the screening memo.")
