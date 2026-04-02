from __future__ import annotations

import json
from pathlib import Path

import joblib


REQUIRED = {
    "model": Path("models/investment_model_pipeline.joblib"),
    "features": Path("models/model_features.joblib"),
    "label_encoder": Path("models/label_encoder.joblib"),
    "threshold_config": Path("models/threshold_config.json"),
}


def check_required_artifacts(base_dir: Path) -> list[str]:
    missing = []
    for rel_path in REQUIRED.values():
        abs_path = base_dir / rel_path
        if not abs_path.exists():
            missing.append(str(rel_path))
    return missing


def load_runtime_artifacts(base_dir: Path) -> dict:
    missing = check_required_artifacts(base_dir)
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    paths = {name: base_dir / rel for name, rel in REQUIRED.items()}

    model = joblib.load(paths["model"])
    feature_names = joblib.load(paths["features"])
    label_encoder = joblib.load(paths["label_encoder"])

    with open(paths["threshold_config"], "r", encoding="utf-8") as f:
        threshold_config = json.load(f)

    return {
        "model": model,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "threshold_config": threshold_config,
        "paths": paths,
    }
