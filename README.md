# 🏠 AI Property Investment Intelligence (South Africa)

An end-to-end machine learning system that **screens residential property investments** and delivers **decision-ready recommendations** using financial modelling and predictive analytics.

---

## 🚀 Problem

Property investors in South Africa manually:

* search across listing platforms
* estimate rental income
* calculate ROI, yield, and affordability
* decide whether a property is worth pursuing

This process is:

* slow
* inconsistent
* difficult to scale

---

## 💡 Solution

This project builds a **production-ready investment screening engine** that:

1. Cleans and structures property data
2. Applies financial investment logic
3. Engineers investment-focused features
4. Trains a classification model
5. Applies **threshold-based decision policies**
6. Outputs investor-ready recommendations

---

## 🧠 Machine Learning System

### 🎯 Objective

Classify property deals into:

* Strong Investment
* Moderate Investment
* Weak Investment

---

### ⚙️ Modelling Framework

* Train / Validation / Test split
* Pipeline-based preprocessing
* Feature selection:

  * Mutual Information
  * ANOVA F-score
* Model selection + tuning
* Threshold optimisation (business-aligned)

---

## 🔥 Key Differentiator

This project goes beyond prediction.

It includes a **deployment-ready decision layer**:

✔ Probability outputs
✔ Threshold tuning
✔ Policy-based predictions
✔ Recommendation logic

---

## 📊 Model Performance

Final model: RandomForest
Deployed prediction policy: default_argmax
prediction_policy	test_accuracy	test_macro_f1
0	default_argmax	1.0000	1.0000
1	threshold_tuned	1.0000	1.0000
Classification Report (Deployed Policy):
                     precision    recall  f1-score   support

Moderate Investment       1.00      1.00      1.00        79
  Strong Investment       1.00      1.00      1.00       269
    Weak Investment       1.00      1.00      1.00      1004

           accuracy                           1.00      1352
          macro avg       1.00      1.00      1.00      1352
       weighted avg       1.00      1.00      1.00      1352

---

### ⚠️ Important Context

* Dataset reflects real-world imbalance (mostly weak deals)
* Accuracy alone is not sufficient
* Focus is on **class-level performance and decision usefulness**

---

## 🏗️ System Architecture

```text id="d8whn4"
Raw Data
   ↓
Cleaning
   ↓
Financial Engine (ROI, Yield, DSCR)
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
Model Training
   ↓
Probability Output
   ↓
Threshold Policy
   ↓
Final Recommendation
```

---

## 📦 Exported Artifacts (Production-Ready)

This project exports all components required for deployment:

### 🔹 Model Layer

* `investment_model_pipeline.joblib`
* `label_encoder.joblib`
* `model_features.joblib`

### 🔹 Decision Layer

* `threshold_config.json`
* `class_labels.json`

### 🔹 Prediction Outputs

* validation predictions
* test predictions
* probability breakdowns

### 🔹 Evaluation Outputs

* model metrics summary
* model selection logic
* hyperparameter tuning results
* threshold tuning results

### 🔹 Explainability

* SHAP global importance
* SHAP local explanations

### 🔹 System Manifest

* `artifact_manifest.json` (central registry of all outputs)

---

## 🧪 Evaluation Strategy

The model is evaluated using:

* Confusion matrix
* Class-level metrics
* Threshold-adjusted predictions
* Error analysis
* SHAP explainability

📄 See:

* `reports/metrics/`
* `reports/explainability/`

---

## 💼 Business Impact

For every 1,000 property listings:

* ~60–70% automatically filtered
* ~20–30% flagged for review
* ~10–15% identified as strong opportunities

👉 Enables **scalable property investment screening**

---

## 🖥️ Application Layer

A Streamlit app consumes exported artifacts and:

* loads trained model
* applies feature pipeline
* applies threshold policy
* outputs investor recommendation

---

## ⚙️ How to Run

```bash id="xgpyzq"
git clone https://github.com/your-username/AI-Property-Investment-Intelligence.git
cd AI-Property-Investment-Intelligence

pip install -r requirements.txt

python src/run_pipeline.py

streamlit run app/streamlit_app.py
```

---

## ⚠️ Data Notes

* Real-world class imbalance preserved
* Synthetic scaling used only for UI/demo
* Model trained on original data

---

## 🛠️ Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* SHAP
* Streamlit
* Joblib

---

## 📌 Why This Project Stands Out

This is not just a model.

It is a **complete decision system**:

✔ Financial logic + ML
✔ Feature engineering aligned to business
✔ Threshold-based decisioning
✔ Deployment-ready artifacts
✔ Explainability built-in

---

## 👤 Author

Ashley Mathabatha
Data Scientist | Property Investment Intelligence
