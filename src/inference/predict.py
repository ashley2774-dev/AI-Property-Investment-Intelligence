from __future__ import annotations

import numpy as np
import pandas as pd


BUSINESS_COLUMNS = [
    "province",
    "city",
    "suburb",
    "property_type",
    "bedrooms",
    "bathrooms",
    "parking",
    "purchase_price",
    "estimated_rent",
    "floor_area_sqm",
    "levy",
    "rates_taxes",
    "insurance",
    "other_opex",
    "deposit_pct",
    "interest_rate_pct",
    "loan_term_years",
    "vacancy_pct",
    "management_fee_pct",
    "maintenance_pct",
    "annual_growth_pct",
    "deposit_amount",
    "loan_amount",
    "monthly_bond_payment",
    "monthly_vacancy_cost",
    "monthly_management_fee",
    "monthly_maintenance_cost",
    "monthly_total_opex",
    "monthly_noi",
    "monthly_cash_flow",
    "gross_yield_pct",
    "net_yield_pct",
    "roi_pct",
    "dscr",
    "bond_to_rent",
    "opex_to_rent",
]


def _find_class_index(classes, target_keyword):
    for idx, cls in enumerate(classes):
        value = str(cls).strip().lower()
        if value == target_keyword:
            return idx
    for idx, cls in enumerate(classes):
        if target_keyword in str(cls).strip().lower():
            return idx
    raise ValueError(f"Could not find class containing keyword: {target_keyword}")


def apply_threshold_policy(prob_array, class_names, strong_threshold=0.50, weak_threshold=0.50):
    strong_idx = _find_class_index(class_names, "strong")
    weak_idx = _find_class_index(class_names, "weak")

    predictions = []
    for row in prob_array:
        if row[strong_idx] >= strong_threshold:
            predictions.append(strong_idx)
        elif row[weak_idx] >= weak_threshold:
            predictions.append(weak_idx)
        else:
            predictions.append(int(np.argmax(row)))
    return np.array(predictions)


def _safe_business_frame(business_df: pd.DataFrame) -> pd.DataFrame:
    results = business_df.copy()

    for col in BUSINESS_COLUMNS:
        if col not in results.columns:
            results[col] = np.nan

    return results


def score_properties(model_df: pd.DataFrame, business_df: pd.DataFrame, runtime: dict) -> pd.DataFrame:
    model = runtime["model"]
    label_encoder = runtime["label_encoder"]
    threshold_config = runtime["threshold_config"]
    class_names = [str(c) for c in label_encoder.classes_]

    pred_default = model.predict(model_df)
    prob_array = model.predict_proba(model_df)

    pred_threshold = apply_threshold_policy(
        prob_array,
        class_names=class_names,
        strong_threshold=float(threshold_config.get("strong_threshold", 0.5)),
        weak_threshold=float(threshold_config.get("weak_threshold", 0.5)),
    )

    deployed_policy = str(threshold_config.get("deployed_prediction_policy", "threshold_tuned"))
    recommended = pred_threshold if deployed_policy == "threshold_tuned" else pred_default

    results = _safe_business_frame(business_df)

    for class_name, class_probs in zip(class_names, prob_array.T):
        results[f"prob_{class_name}"] = class_probs

    results["predicted_label_default"] = label_encoder.inverse_transform(pred_default)
    results["predicted_label_threshold"] = label_encoder.inverse_transform(pred_threshold)
    results["recommended_label"] = label_encoder.inverse_transform(recommended)
    results["deployed_prediction_policy"] = deployed_policy
    results["top_probability_pct"] = prob_array.max(axis=1) * 100.0

    return results
