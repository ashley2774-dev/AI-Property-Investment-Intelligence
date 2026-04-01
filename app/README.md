# Property Investment Streamlit MVP

This MVP turns the financed buy-to-let investment model into a usable screening app.

## What it does

The user enters a property deal, and the app:

1. captures the deal assumptions
2. rebuilds the financial logic
3. creates inference-time features
4. scores the deal
5. returns an investor-readable recommendation

## Current design

This version uses a rule-based stand-in model so the full app flow works immediately.

When your final trained model is ready, replace the temporary model in:

- `src/inference/model.py`

with your saved pipeline artifact, for example:

- `joblib.load("models/final_model.joblib")`

## Suggested next upgrade

Refactor your notebook outputs into reusable modules:

- `src/training/` for training code
- `src/inference/` for live scoring
- `src/features/` for shared feature logic
- `src/financial_engine/` for shared investment calculations

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Replace the temporary model with your real model

Your production-ready inference flow should become:

- load the exact saved preprocessing + model pipeline
- enforce column order
- apply the same feature engineering used during training
- predict class and probabilities
- apply a business rules layer for investor messaging

## Strong portfolio add-ons

- probability explanation panel
- scenario analysis sliders
- suburb benchmarks
- PDF investment memo export
- FastAPI backend
- Docker deployment
- logging and monitoring
