from __future__ import annotations


def generate_flags(result: dict) -> dict:
    green_flags = []
    red_flags = []

    monthly_cash_flow = float(result.get("monthly_cash_flow", 0) or 0)
    dscr = float(result.get("dscr", 0) or 0)
    roi_pct = float(result.get("roi_pct", 0) or 0)
    gross_yield = float(result.get("gross_yield_pct", 0) or 0)
    confidence = float(result.get("top_probability_pct", 0) or 0)
    recommended = str(result.get("recommended_label", "")).lower()

    if monthly_cash_flow > 0:
        green_flags.append(f"Positive monthly cash flow of R {monthly_cash_flow:,.0f}.")
    else:
        red_flags.append(f"Negative monthly cash flow of R {monthly_cash_flow:,.0f}.")

    if dscr >= 1.2:
        green_flags.append(f"Debt service cover is healthy at {dscr:.2f}.")
    elif dscr < 1.0:
        red_flags.append(f"Debt service cover is below 1.00 at {dscr:.2f}.")

    if roi_pct >= 8:
        green_flags.append(f"Cash-on-cash style ROI is attractive at {roi_pct:.2f}%.")
    elif roi_pct < 0:
        red_flags.append(f"ROI is negative at {roi_pct:.2f}%.")

    if gross_yield >= 10:
        green_flags.append(f"Gross yield is strong at {gross_yield:.2f}%.")
    elif gross_yield < 7:
        red_flags.append(f"Gross yield is thin at {gross_yield:.2f}%.")

    if confidence >= 70:
        green_flags.append(f"Model confidence is relatively strong at {confidence:.1f}%.")
    elif confidence < 45:
        red_flags.append(f"Model confidence is low at {confidence:.1f}%; treat the output as uncertain.")

    if recommended == "strong":
        green_flags.append("The deployed policy classifies this deal as a strong candidate for further investigation.")
    elif recommended == "weak":
        red_flags.append("The deployed policy classifies this deal as weak based on current assumptions.")

    return {"green_flags": green_flags, "red_flags": red_flags}
