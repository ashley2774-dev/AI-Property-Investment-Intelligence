# Streamlit deployment files for AI Property Investment Intelligence

## Files to upload to the root of the GitHub repository

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `src/inference/__init__.py`
- `src/inference/benchmarks.py`
- `src/inference/finance.py`
- `src/inference/amortization.py`
- `src/inference/model.py`
- `src/inference/recommendation.py`
- `src/inference/predict.py`

## Optional but strongly recommended

- `models/investment_screener_pipeline.joblib`
  - add this later when the final trained pipeline is exported
  - until then the app uses the fallback transparent area model

## Data files the app tries to read automatically

Place these in `data/processed/` if they are not already in the repo:

- `financial_engine_feature_engineered_full.csv`
- `feature_engineered_dataset_v1.csv`
- `expanded_sales_data.csv`
- `rental_summary.csv`
- `expanded_rentals_data.csv`

The app reads the first available processed files and infers suburb-level assumptions.

## Streamlit Community Cloud settings

- Repo: `ashley2774-dev/AI-Property-Investment-Intelligence`
- Branch: `main`
- Main file path: `app.py`

## Why this version is cleaner than the current repo state

- It uses a single root app entrypoint.
- It removes the raw JSON debug display.
- It no longer requires the user to type every deal assumption manually.
- It auto-fills assumptions from processed project data where possible.
- It includes an amortization and equity view.
