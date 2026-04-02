from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_label(value: str) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _find_first_matching_index(classes, keywords):
    normalized = [_normalize_label(c) for c in classes]

    for keyword in keywords:
        keyword_norm = _normalize_label(keyword)
        for idx, cls in enumerate(normalized):
            if cls == keyword_norm:
                return idx

    for keyword in keywords:
        keyword_norm = _normalize_label(keyword)
        for idx, cls in enumerate(normalized):
            if keyword_norm in cls:
                return idx

    return None


def _resolve_threshold_indices(class_names):
    """
    Try to map the model's class labels to:
    - strong / best class
    - weak / worst class

    Supports labels like:
    - Good / Moderate / Bad
    - Strong Investment / Average Investment / Weak Investment
    - Recommended / Review / Reject
    """

    strong_keywords = [
        "good",
        "strong",
        "recommended",
        "proceed",
        "good investment",
        "strong investment",
    ]

    weak_keywords = [
        "bad",
        "weak",
        "reject",
        "do not proceed",
        "bad investment",
        "weak investment",
    ]

    strong_idx = _find_first_matching_index(class_names, strong_keywords)
    weak_idx = _find_first_matching_index(class_names, weak_keywords)

    return strong_idx, weak_idx


def apply_threshold_policy(prob_array, class_names, strong_threshold=0.50, weak_threshold=0.50):
    strong_idx, weak_idx = _resolve_threshold_indices(class_names)

    predictions = []
    for row in prob_array:
        if strong_idx is not None and row[strong_idx] >= strong_threshold:
            predictions.append(strong_idx)
        elif weak_idx is not None and row[weak_idx] >= weak_threshold:
            predictions.append(weak_idx)
        else:
            predictions.append(int(np.argmax(row)))

    return np.array(predictions)


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

    results = business_df.reset_index(drop=True).copy()

    for class_name, class_probs in zip(class_names, prob_array.T):
        results[f"prob_{class_name}"] = class_probs * 100.0

    results["predicted_label_default"] = label_encoder.inverse_transform(pred_default)
    results["predicted_label_threshold"] = label_encoder.inverse_transform(pred_threshold)
    results["recommended_label"] = label_encoder.inverse_transform(recommended)
    results["deployed_prediction_policy"] = deployed_policy
    results["top_probability_pct"] = prob_array.max(axis=1) * 100.0

    return results
