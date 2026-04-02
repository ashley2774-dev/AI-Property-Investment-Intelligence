# Phase 3 Streamlit deployment package

This package replaces the earlier rule-based Streamlit MVP with a real-model deployment flow.

## What this package does

- loads the exported trained pipeline from `models/investment_model_pipeline.joblib`
- loads the notebook threshold policy from `models/threshold_config.json`
- aligns app inputs to the model feature list from `models/model_features.joblib`
- scores single-property and batch CSV inputs with the real trained model
- shows summary, financials, amortization, and artifact previews in Streamlit

## Upload these files to the **root of your GitHub repository**

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `src/__init__.py`
- `src/inference/__init__.py`
- `src/inference/finance.py`
- `src/inference/features.py`
- `src/inference/model.py`
- `src/inference/predict.py`
- `src/inference/rules.py`
- `src/inference/amortization.py`
- `sample_batch_input.csv`

## Upload these notebook-exported artifacts too

Place these at the repo root in the exact folders below:

- `models/investment_model_pipeline.joblib`
- `models/model_features.joblib`
- `models/label_encoder.joblib`
- `models/threshold_config.json`
- `models/class_labels.json` *(optional but recommended)*
- `reports/metrics/final_model_metrics_summary.csv` *(optional)*
- `reports/metrics/model_selection_logic_summary.csv` *(optional)*
- `reports/metrics/hyperparameter_tuning_summary.csv` *(optional)*
- `reports/metrics/threshold_tuning_summary.csv` *(optional)*
- `reports/features/final_model_feature_importance.csv` *(optional)*
- `reports/explainability/shap_global_importance.csv` *(optional)*
- `reports/explainability/shap_local_examples.csv` *(optional)*

## Streamlit Community Cloud settings

- Repository: `ashley2774-dev/AI-Property-Investment-Intelligence`
- Branch: `main`
- Main file path: `app.py`

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```
