# Workflow - Property Investment Intelligence SA

## 1. Overview

In this project, I am building an end-to-end workflow that transforms raw property and rental data into financed investment recommendations.

The workflow is designed to move through the following stages:

1. Data ingestion
2. Data cleaning and validation
3. Feature engineering
4. Financial metric calculation
5. Label generation
6. Model training and evaluation
7. Deployment
8. Monitoring

---

## 2. End-to-End Workflow

### Stage 1: Ingestion

At this stage, I collect raw data from:
- property listing sources
- rental listing sources
- area-level sources
- workbook exports
- user assumptions

Raw files are stored in:
- `data/raw/listings/`
- `data/raw/rentals/`
- `data/raw/area_data/`
- `data/raw/workbook_exports/`

---

### Stage 2: Cleaning and Validation

After ingestion, I clean and standardize the data by:
- removing duplicates
- standardizing suburb and city names
- converting fields to usable types
- handling missing values
- validating numeric ranges
- checking data consistency

Cleaned files are stored in:
- `data/interim/`

---

### Stage 3: Dataset Integration

Once each source is cleaned, I merge them into a modeling base table.

This includes:
- matching sale listings to rental comparables
- joining area-level averages
- attaching financing assumptions
- preparing a single property-level record per candidate investment

Integrated outputs are stored in:
- `data/processed/`

---

### Stage 4: Financial Metric Engine

Next, I apply the financial logic defined in the project.

This includes calculating:
- loan_amount
- bond_payment
- maintenance
- vacancy_cost
- total_expenses
- NOI
- cash_flow
- DSCR
- rental_yield
- ROI

This stage converts a property record into an investment evaluation record.

---

### Stage 5: Feature Engineering

I then engineer the features used by the model.

These include:
- property features
- rental features
- financing features
- expense features
- area comparison features
- text-derived features

The result is a modeling table ready for training and inference.

---

### Stage 6: Label Creation

Using the financial engine outputs, I assign rule-based proxy labels such as:
- Strong Investment
- Moderate Investment
- Weak Investment

These labels are used as the target for supervised learning.

---

### Stage 7: Model Training and Evaluation

I use the prepared dataset to:
- train a property viability classifier
- train a rental prediction model
- compare baseline and advanced models
- evaluate performance
- generate explainability outputs

Model artifacts are stored in:
- `models/`

---

### Stage 8: Deployment

The trained logic is exposed through a Streamlit application.

The app allows a user to:
- input or review a property
- see financial outputs
- receive a prediction
- view the reasons behind the result

Deployment-related logic is stored in:
- `app/`
- `src/deployment/`

---

### Stage 9: Monitoring

After deployment, I monitor:
- data quality
- feature drift
- model performance
- retraining triggers

This makes the system more realistic and production-oriented.

---

## 3. Workflow Philosophy

I am designing this workflow to achieve three goals:

### Reproducibility
I want the same inputs to produce the same outputs consistently.

### Modularity
Each stage should be independently testable and maintainable.

### Business alignment
Every transformation should support better investment decision-making.

---

## 4. Summary

This workflow turns a fragmented manual process into a structured, end-to-end data product.

Instead of evaluating one property at a time in a spreadsheet, I am building a system that can ingest, evaluate, and explain investment opportunities at scale.
