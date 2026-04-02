# Model artifacts upload guide

After re-running the modelling notebook, upload or commit these exported files into the matching repository paths.

## Required for the app to run

- `models/investment_model_pipeline.joblib`
- `models/model_features.joblib`
- `models/label_encoder.joblib`
- `models/threshold_config.json`

## Recommended supporting files

- `models/class_labels.json`
- `reports/metrics/final_model_metrics_summary.csv`
- `reports/metrics/model_selection_logic_summary.csv`
- `reports/metrics/hyperparameter_tuning_summary.csv`
- `reports/metrics/threshold_tuning_summary.csv`
- `reports/features/final_model_feature_importance.csv`
- `reports/explainability/shap_global_importance.csv`
- `reports/explainability/shap_local_examples.csv`

## Expected repo structure

```text
AI-Property-Investment-Intelligence/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── models/
│   ├── investment_model_pipeline.joblib
│   ├── model_features.joblib
│   ├── label_encoder.joblib
│   ├── threshold_config.json
│   └── class_labels.json
├── reports/
│   ├── metrics/
│   ├── features/
│   └── explainability/
└── src/
    └── inference/
```

## Notes

- The app will stop with a clear error message if any required model artifact is missing.
- The optional report files are only used for previews inside the Artifacts tab.
- If your Streamlit deployment fails on package resolution, re-check that the root `requirements.txt` is present in GitHub.
