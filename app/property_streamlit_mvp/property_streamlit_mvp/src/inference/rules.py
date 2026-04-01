from __future__ import annotations


def build_recommendation(result: dict) -> dict:
    fin = result["financials"]
    pred = result["prediction"]

    positives = []
    risks = []

    if fin["monthly_cash_flow"] > 0:
        positives.append("Deal produces positive monthly cash flow under the current assumptions.")
    else:
        risks.append("Deal is cash-flow negative under the current assumptions.")

    if fin["dscr"] >= 1.25:
        positives.append("Debt service coverage is healthy for a financed buy-to-let deal.")
    else:
        risks.append("Debt service coverage is thin and may not comfortably absorb shocks.")

    if fin["gross_yield"] >= 0.09:
        positives.append("Gross yield is in a stronger screening range.")
    else:
        risks.append("Gross yield looks soft relative to the purchase price.")

    if fin["roi"] >= 0.08:
        positives.append("Return on invested cash is attractive at screening level.")
    else:
        risks.append("Return on invested cash is modest for the deposit committed.")

    if pred == "recommended":
        recommendation = "Proceed"
        summary = (
            "This deal screens as attractive for first-pass review. The financing burden appears manageable "
            "relative to rent, and the property produces acceptable investment metrics under the current assumptions."
        )
    elif pred == "review":
        recommendation = "Review Carefully"
        summary = (
            "This deal is borderline. It may still work, but only if the rent assumption is reliable, operating costs "
            "are well controlled, or the purchase price can be negotiated."
        )
    else:
        recommendation = "Do Not Proceed"
        summary = (
            "This deal currently screens poorly. On the present assumptions it does not look strong enough as a financed "
            "buy-to-let opportunity and should not move forward without a material change in price, rent, or funding terms."
        )

    if not positives:
        positives.append("No major financial strength flags were triggered by the current assumptions.")
    if not risks:
        risks.append("No major screening risk flags were triggered by the current assumptions.")

    return {
        "recommendation": recommendation,
        "summary": summary,
        "positives": positives,
        "risks": risks,
    }
