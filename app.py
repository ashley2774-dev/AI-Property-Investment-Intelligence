from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference.amortization import build_amortization_schedule, summarize_amortization
from src.inference.features import build_single_property_features, coerce_batch_input
from src.inference.model import load_runtime_artifacts, check_required_artifacts
from src.inference.predict import score_properties
from src.inference.rules import generate_flags

st.set_page_config(
    page_title="AI Property Investment Intelligence",
    page_icon="🏠",
    layout="wide",
)

ARTIFACTS = {
    "model": "models/investment_model_pipeline.joblib",
    "features": "models/model_features.joblib",
    "label_encoder": "models/label_encoder.joblib",
    "threshold_config": "models/threshold_config.json",
}


@st.cache_resource(show_spinner=False)
def get_runtime():
    return load_runtime_artifacts(base_dir=Path("."))


def _fmt_money(value):
    return f"R {float(value):,.0f}"


def _fmt_pct(value):
    return f"{float(value):,.2f}%"


def _render_artifact_warning():
    missing = check_required_artifacts(Path("."))
    if missing:
        st.error(
            "The deployed app is missing one or more model artifacts. "
            "Upload the exported notebook outputs listed below into the repo root so Streamlit can load the real model."
        )
        st.code("\n".join(missing))
        st.stop()


def _single_property_form():
    st.subheader("Single property screening")

    col1, col2, col3 = st.columns(3)

    with col1:
        province = st.text_input("Province", value="Gauteng")
        city = st.text_input("City", value="Johannesburg")
        suburb = st.text_input("Suburb", value="Bryanston")
        property_type = st.selectbox(
            "Property type",
            ["Apartment", "House", "Townhouse", "Cluster", "Duplex", "Studio"],
            index=0,
        )
        purchase_price = st.number_input(
            "Purchase price (R)",
            min_value=100000.0,
            value=950000.0,
            step=10000.0,
        )
        estimated_rent = st.number_input(
            "Estimated monthly rent (R)",
            min_value=1000.0,
            value=8500.0,
            step=250.0,
        )
        floor_area_sqm = st.number_input(
            "Floor area (sqm)",
            min_value=15.0,
            value=65.0,
            step=1.0,
        )

    with col2:
        bedrooms = st.number_input("Bedrooms", min_value=0, value=2, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, value=1.0, step=0.5)
        parking = st.number_input("Parking / garage spaces", min_value=0, value=1, step=1)
        levy = st.number_input("Monthly levy (R)", min_value=0.0, value=1200.0, step=100.0)
        rates_taxes = st.number_input(
            "Monthly rates & taxes (R)",
            min_value=0.0,
            value=700.0,
            step=50.0,
        )
        insurance = st.number_input(
            "Monthly insurance (R)",
            min_value=0.0,
            value=350.0,
            step=50.0,
        )
        other_opex = st.number_input(
            "Other monthly opex (R)",
            min_value=0.0,
            value=250.0,
            step=50.0,
        )

    with col3:
        deposit_pct = st.slider(
            "Deposit (%)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
        )
        interest_rate_pct = st.slider(
            "Interest rate (%)",
            min_value=5.0,
            max_value=18.0,
            value=11.75,
            step=0.25,
        )
        loan_term_years = st.slider(
            "Loan term (years)",
            min_value=5,
            max_value=30,
            value=20,
            step=1,
        )
        vacancy_pct = st.slider(
            "Vacancy allowance (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
        )
        management_fee_pct = st.slider(
            "Management fee (%)",
            min_value=0.0,
            max_value=20.0,
            value=8.0,
            step=0.5,
        )
        maintenance_pct = st.slider(
            "Maintenance allowance (%)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
        )
        annual_growth_pct = st.slider(
            "Annual property growth (%)",
            min_value=0.0,
            max_value=20.0,
            value=6.0,
            step=0.5,
        )

    return {
        "province": province,
        "city": city,
        "suburb": suburb,
        "property_type": property_type,
        "purchase_price": purchase_price,
        "estimated_rent": estimated_rent,
        "floor_area_sqm": floor_area_sqm,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "levy": levy,
        "rates_taxes": rates_taxes,
        "insurance": insurance,
        "other_opex": other_opex,
        "deposit_pct": deposit_pct,
        "interest_rate_pct": interest_rate_pct,
        "loan_term_years": loan_term_years,
        "vacancy_pct": vacancy_pct,
        "management_fee_pct": management_fee_pct,
        "maintenance_pct": maintenance_pct,
        "annual_growth_pct": annual_growth_pct,
    }


def _normalize_result_fields(result: dict) -> dict:
    if "estimated_rent" not in result and "monthly_rent" in result:
        result["estimated_rent"] = result["monthly_rent"]

    if "gross_yield_pct" not in result and "gross_yield" in result:
        result["gross_yield_pct"] = float(result["gross_yield"]) * 100

    if "net_yield_pct" not in result and "net_yield" in result:
        result["net_yield_pct"] = float(result["net_yield"]) * 100

    if "roi_pct" not in result and "roi" in result:
        result["roi_pct"] = float(result["roi"]) * 100

    return result


def _confidence_display(result: dict) -> str:
    confidence_value = float(result.get("top_probability_pct", 0))
    if confidence_value <= 1:
        confidence_value *= 100
    return _fmt_pct(confidence_value)


def _render_amortization_table(schedule: pd.DataFrame):
    st.markdown("**Projection snapshot**")
    projection_cols = ["year", "property_value", "loan_balance", "equity"]
    available_cols = [c for c in projection_cols if c in schedule.columns]

    if available_cols:
        display_df = schedule[available_cols].copy()
        for col in ["property_value", "loan_balance", "equity"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(float).map(_fmt_money)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(schedule, use_container_width=True, hide_index=True)


def main():
    st.title("🏠 AI Property Investment Intelligence")
    st.caption("Real-model Streamlit deployment for financed buy-to-let screening")

    st.sidebar.header("About")
    st.sidebar.write(
        "This app screens South African buy-to-let properties using the exported trained pipeline, "
        "threshold policy, and notebook artifacts from your modelling workflow."
    )
    st.sidebar.info(
        "Upload the model artifacts into the repo root under /models and optional reports under /reports. "
        "The app will then score deals with the real pipeline instead of the earlier rule-based stand-in."
    )

    _render_artifact_warning()
    runtime = get_runtime()

    tabs = st.tabs(["Summary", "Financials", "Amortization & equity", "Batch scoring", "Artifacts"])

    with tabs[0]:
        user_input = _single_property_form()
        score_clicked = st.button("Screen property", type="primary")

        if score_clicked:
            property_df = build_single_property_features(user_input, runtime["feature_names"])
            scored = score_properties(property_df, runtime)

            full_result_df = pd.concat(
                [
                    property_df.reset_index(drop=True),
                    scored.reset_index(drop=True),
                ],
                axis=1,
            )

            result = full_result_df.iloc[0].to_dict()
            result = _normalize_result_fields(result)

            flag_pack = generate_flags(result)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recommended label", str(result.get("recommended_label", "unknown")).title())
            c2.metric("Confidence", _confidence_display(result))
            c3.metric("Monthly cash flow", _fmt_money(float(result.get("monthly_cash_flow", 0))))
            c4.metric("DSCR", f"{float(result.get('dscr', 0)):.2f}")

            st.subheader("Probability breakdown")
            prob_cols = [c for c in full_result_df.columns if c.startswith("prob_")]
            if prob_cols:
                prob_display = (
                    full_result_df[prob_cols]
                    .rename(columns=lambda x: x.replace("prob_", "").title())
                    .T
                    .reset_index()
                    .rename(columns={"index": "Class", 0: "Probability"})
                )
                prob_display["Probability"] = (prob_display["Probability"] * 100).round(2)
                prob_display["Probability"] = prob_display["Probability"].map(lambda x: f"{x:.2f}%")
                st.dataframe(prob_display, use_container_width=True, hide_index=True)

            st.subheader("Green flags")
            if flag_pack["green_flags"]:
                for item in flag_pack["green_flags"]:
                    st.success(item)
            else:
                st.write("No strong green flags identified from the current inputs.")

            st.subheader("Red flags")
            if flag_pack["red_flags"]:
                for item in flag_pack["red_flags"]:
                    st.error(item)
            else:
                st.write("No major red flags identified from the current inputs.")

            st.subheader("Model-ready preview")
            st.dataframe(full_result_df, use_container_width=True)

            st.session_state["latest_result"] = result
            st.session_state["latest_input"] = user_input

    with tabs[1]:
        st.subheader("Financials")
        if "latest_result" not in st.session_state:
            st.info("Screen a property in the Summary tab to populate the financial breakdown.")
        else:
            result = st.session_state["latest_result"]

            financial_rows = [
                ("Purchase price", result.get("purchase_price", 0)),
                ("Estimated rent", result.get("estimated_rent", 0)),
                ("Loan amount", result.get("loan_amount", 0)),
                ("Monthly bond payment", result.get("monthly_bond_payment", 0)),
                ("Monthly vacancy allowance", result.get("monthly_vacancy_cost", 0)),
                ("Monthly management fee", result.get("monthly_management_fee", 0)),
                ("Monthly maintenance", result.get("monthly_maintenance_cost", 0)),
                ("Monthly total opex", result.get("monthly_total_opex", 0)),
                ("Monthly NOI", result.get("monthly_noi", 0)),
                ("Monthly cash flow", result.get("monthly_cash_flow", 0)),
            ]
            financial_df = pd.DataFrame(financial_rows, columns=["Metric", "Value"])
            financial_df["Value"] = financial_df["Value"].astype(float).map(_fmt_money)
            st.dataframe(financial_df, use_container_width=True, hide_index=True)

            ratio_rows = [
                ("Gross yield (%)", result.get("gross_yield_pct", 0)),
                ("Net yield (%)", result.get("net_yield_pct", 0)),
                ("ROI (%)", result.get("roi_pct", 0)),
                ("DSCR", result.get("dscr", 0)),
                ("Bond-to-rent", result.get("bond_to_rent", 0)),
                ("Opex-to-rent", result.get("opex_to_rent", 0)),
            ]
            ratio_df = pd.DataFrame(ratio_rows, columns=["Ratio", "Value"])
            ratio_df["Value"] = ratio_df.apply(
                lambda r: _fmt_pct(float(r["Value"])) if "%" in r["Ratio"] else f"{float(r['Value']):,.2f}",
                axis=1,
            )
            st.dataframe(ratio_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Amortization & equity")
        if "latest_input" not in st.session_state:
            st.info("Screen a property first to build the amortization projection.")
        else:
            latest_input = st.session_state["latest_input"]
            latest_result = st.session_state["latest_result"]

            schedule = build_amortization_schedule(
                purchase_price=float(latest_result.get("purchase_price", 0)),
                loan_amount=float(latest_result.get("loan_amount", 0)),
                annual_interest_rate=float(latest_input["interest_rate_pct"]) / 100.0,
                loan_term_years=int(latest_input["loan_term_years"]),
                monthly_cash_flow=float(latest_result.get("monthly_cash_flow", 0)),
                annual_growth_rate=float(latest_input["annual_growth_pct"]) / 100.0,
            )
            summary = summarize_amortization(schedule)

            c1, c2, c3 = st.columns(3)
            c1.metric("Equity after full term", _fmt_money(summary["ending_equity"]))
            c2.metric("Property value at term", _fmt_money(summary["ending_property_value"]))
            c3.metric("Cash-flow break-even year", summary["break_even_year_display"])

            _render_amortization_table(schedule)

            st.markdown("**Full amortization schedule**")
            schedule_display = schedule.copy()
            for col in ["property_value", "loan_balance", "equity"]:
                if col in schedule_display.columns:
                    schedule_display[col] = schedule_display[col].astype(float).map(_fmt_money)
            st.dataframe(schedule_display, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Batch scoring")
        st.write(
            "Upload a CSV using the provided template. The app will calculate financial features, "
            "align the frame to the training feature list, and score each property with the real model."
        )

        template_path = Path("sample_batch_input.csv")
        if template_path.exists():
            with open(template_path, "rb") as f:
                st.download_button(
                    "Download CSV template",
                    data=f.read(),
                    file_name="sample_batch_input.csv",
                    mime="text/csv",
                )

        uploaded = st.file_uploader("Upload property CSV", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            prepared = coerce_batch_input(batch_df, runtime["feature_names"])
            scored_batch = score_properties(prepared, runtime)
            st.dataframe(scored_batch, use_container_width=True)
            st.download_button(
                "Download scored results",
                data=scored_batch.to_csv(index=False).encode("utf-8"),
                file_name="investment_scored_results.csv",
                mime="text/csv",
            )

    with tabs[4]:
        st.subheader("Loaded artifacts")
        manifest = {
            "model_path": str(runtime["paths"]["model"]),
            "feature_list_path": str(runtime["paths"]["features"]),
            "label_encoder_path": str(runtime["paths"]["label_encoder"]),
            "threshold_config_path": str(runtime["paths"]["threshold_config"]),
        }
        st.json(manifest)

        optional_reports = [
            Path("reports/metrics/final_model_metrics_summary.csv"),
            Path("reports/features/final_model_feature_importance.csv"),
            Path("reports/explainability/shap_global_importance.csv"),
        ]
        for report_path in optional_reports:
            if report_path.exists():
                st.write(f"Preview: {report_path}")
                st.dataframe(pd.read_csv(report_path).head(20), use_container_width=True)


if __name__ == "__main__":
    main()
