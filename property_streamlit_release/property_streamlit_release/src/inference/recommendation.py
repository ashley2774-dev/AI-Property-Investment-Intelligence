from __future__ import annotations

from typing import Dict, List


def build_recommendation(prediction: str, confidence: float, financials: Dict[str, float], assumptions: Dict[str, float | str]) -> Dict[str, object]:
    green_flags: List[str] = []
    red_flags: List[str] = []

    if financials["gross_yield"] >= 0.10:
        green_flags.append("Gross yield is comfortably above the common first-pass threshold.")
    if financials["net_yield"] >= 0.07:
        green_flags.append("Net yield remains healthy after operating costs.")
    if financials["dscr"] >= 1.10:
        green_flags.append("Debt service coverage suggests rent can support the bond.")
    if financials["monthly_cash_flow"] >= 0:
        green_flags.append("Monthly cash flow is non-negative under the current assumptions.")
    if assumptions.get("suburb_sample_rentals", 0) >= 5:
        green_flags.append("The suburb has enough rental observations to support a benchmark estimate.")

    if financials["monthly_cash_flow"] < 0:
        red_flags.append("Monthly cash flow is negative, so the deal may need owner support in the early years.")
    if financials["dscr"] < 1.0:
        red_flags.append("Debt service coverage is below 1.0, which is weak for a financed rental deal.")
    if financials["roi"] < 0:
        red_flags.append("Cash-on-cash ROI is currently negative on the deposit used.")
    if assumptions.get("data_match_level") != "suburb":
        red_flags.append("The app had to fall back to province-level or global defaults for some assumptions.")
    if confidence < 0.60:
        red_flags.append("Confidence is modest, so the result should be treated as a triage signal rather than a decision.")

    label_map = {
        "recommended": "Green light",
        "review": "Needs review",
        "reject": "Red light",
    }

    if prediction == "recommended":
        summary = (
            "This suburb screens as potentially investable on a first pass. The area-level rent and price "
            "assumptions produce a financing profile that looks workable, but the deal still needs listing-level verification."
        )
        next_action = "Pull live listings in this suburb, compare asking prices to the benchmark, and validate actual rental comps before making an offer."
    elif prediction == "review":
        summary = (
            "This suburb sits in the middle band. The opportunity may work, but it is highly sensitive to pricing, financing, and rent assumptions."
        )
        next_action = "Manually test best-case and worst-case rent and purchase-price scenarios before progressing."
    else:
        summary = (
            "This suburb does not currently screen well for a financed buy-to-let deal under the benchmark assumptions."
        )
        next_action = "Only revisit this suburb if you find an unusually discounted purchase price or much stronger rent evidence than the benchmark."

    return {
        "label": label_map[prediction],
        "summary": summary,
        "next_action": next_action,
        "green_flags": green_flags or ["No major green flags were triggered under the current benchmark assumptions."],
        "red_flags": red_flags or ["No major red flags were triggered under the current benchmark assumptions."],
    }
