# Property Investment Streamlit MVP

This Streamlit package is a cleaned and upgraded version of the property investment screener MVP.

## What is included

- About section in the sidebar
- Summary tab
- Financials tab
- Amortization & equity tab
- Green flags and red flags
- Property growth and equity projection
- Cash flow break-even year projection

## Suggested GitHub structure

Upload the contents of this folder to the root of your GitHub repository:

- `app.py`
- `requirements.txt`
- `src/__init__.py`
- `src/inference/__init__.py`
- `src/inference/finance.py`
- `src/inference/features.py`
- `src/inference/model.py`
- `src/inference/predict.py`
- `src/inference/rules.py`
- `src/inference/amortization.py`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Use these settings:

- Repository: your GitHub repository
- Branch: `main`
- Main file path: `app.py`

## Current model note

This version still uses the rule-based stand-in model in `src/inference/model.py`.
When your trained pipeline artifact is ready, replace that file's logic with a `joblib.load(...)` flow.
