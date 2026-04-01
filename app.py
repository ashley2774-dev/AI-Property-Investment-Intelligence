import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference.predict import evaluate_deal
from src.inference.rules import build_recommendation

st.set_page_config(page_title="Property Investment Screener", layout="wide")

st.title("South Africa Property Investment Screener")
st.caption("Financed buy-to-let screening MVP")

with st.sidebar:
    st.header("About")
    st.write(
        "This app screens residential buy-to-let deals using the same financial logic and "
        "feature engineering philosophy as the modelling workflow."
    )
    st.info(
        "This is a screening tool for first-pass decision support. It is not legal, tax, or investment advice."
    )

st.subheader("Deal Inputs")

col1, col2, col3 = st.columns(3)

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
    )
    suburb = st.text_input("Suburb", value="Randburg")
    property_type = st.selectbox(
        "Property Type",
        ["Apartment", "House", "Townhouse", "Cluster", "Duplex", "Studio", "Other"],
    )
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
    rent_estimation_source = st.selectbox(
        "Rent Estimation Source",
        ["user_input", "suburb_average", "listing_average", "agent_estimate", "unknown"],
    )

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

    st.divider()
    st.header("Investment Recommendation")

    rec_col1, rec_col2, rec_col3 = st.columns(3)
    rec_col1.metric("Recommendation", recommendation["recommendation"])
    rec_col2.metric("Predicted Class", result["prediction"]) 
    rec_col3.metric("Confidence", f"{result['confidence']:.1%}")

    prob_df = pd.DataFrame(
        {
            "investment_label": list(result["probabilities"].keys()),
            "probability": list(result["probabilities"].values()),
        }
    ).sort_values("probability", ascending=False)

    st.subheader("Class Probabilities")
    st.bar_chart(prob_df.set_index("investment_label"))

    fin = result["financials"]
    st.subheader("ROI Breakdown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Monthly Bond", f"R {fin['monthly_bond_payment']:,.0f}")
    m2.metric("Monthly Cash Flow", f"R {fin['monthly_cash_flow']:,.0f}")
    m3.metric("Gross Yield", f"{fin['gross_yield']:.2%}")
    m4.metric("ROI", f"{fin['roi']:.2%}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Net Yield", f"{fin['net_yield']:.2%}")
    m6.metric("DSCR", f"{fin['dscr']:.2f}")
    m7.metric("Deposit", f"R {fin['deposit_amount']:,.0f}")
    m8.metric("Loan Amount", f"R {fin['loan_amount']:,.0f}")

    st.subheader("Investor Memo")
    st.write(recommendation["summary"])

    left, right = st.columns(2)
    with left:
        st.markdown("**Positive drivers**")
        for item in recommendation["positives"]:
            st.write(f"- {item}")
    with right:
        st.markdown("**Risk flags**")
        for item in recommendation["risks"]:
            st.write(f"- {item}")

    st.subheader("Underlying Inputs and Engineered Features")
    st.dataframe(pd.DataFrame([result["features"]]))

    st.subheader("Raw Result JSON")
    st.code(json.dumps(result, indent=2), language="json")

else:
    st.info("Complete the form and click Evaluate Deal to screen the property.")
