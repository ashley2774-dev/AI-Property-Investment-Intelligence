from __future__ import annotations

import numpy as np
import pandas as pd


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
    strong_idx = _find_class_index(class_names, "good")
    weak_idx = _find_class_index(class_names, "bad")

    predictions = []
    for row in prob_array:
        if row[strong_idx] >= strong_threshold:
            predictions.append(strong_idx)
        elif row[weak_idx] >= weak_threshold:
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
