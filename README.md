# 🏠 AI Property Investment Intelligence (South Africa)

An end-to-end machine learning system that **screens residential property deals** and provides **investment recommendations based on financial viability**.

---

## 🚀 Problem

South African property investors manually evaluate deals by:

* estimating rental income
* calculating ROI, yield, and DSCR
* assessing affordability and cash flow
* comparing properties across suburbs

This process is:

* time-consuming
* inconsistent
* difficult to scale

---

## 💡 Solution

This project builds a **data-driven screening engine** that:

1. Processes property listing data
2. Applies structured financial investment logic
3. Engineers investment-focused features
4. Uses machine learning to classify deal quality
5. Outputs clear investor recommendations

---

## 🧠 Machine Learning Approach

### Problem Type

Multi-class classification:

* Weak Investment
* Moderate Investment
* Strong Investment

---

### Modelling Strategy (Aligned to Notebook)

* Train / Validation / Test split
* Pipeline-based preprocessing
* Baseline model: Logistic Regression
* Feature selection:

  * Mutual Information
  * ANOVA F-score
* Final model: *(update with your best model)*

---

## 📊 Model Performance

*(Replace with your real values from notebook)*

| Metric                    | Score |
| ------------------------- | ----- |
| Accuracy                  | 0.97  |
| Balanced Accuracy         | 0.XX  |
| Strong Investment Recall  | 0.XX  |
| Weak Investment Precision | 0.XX  |

---

### ⚠️ Important Interpretation

Due to class imbalance:

* High accuracy is driven by majority class (Weak investments)
* True performance must be evaluated per class

👉 The model is designed as a **screening tool**, not a final decision-maker

---

## 📈 Key Features Driving Predictions

From feature selection analysis:

Top drivers include:

* Debt service affordability (DSCR)
* Cash flow strength
* Rental yield
* Cost burden ratios
* Financing structure

👉 This confirms the model aligns with **real-world investment logic**

---

## 🏗️ System Architecture

```text
Data → Cleaning → Financial Engine → Feature Engineering → Model → Decision Engine → App
```

---

## 🧪 Model Evaluation Approach

The model is evaluated using:

* Confusion matrix
* Class-level performance
* Error analysis
* Feature importance
* Financial interpretation

📄 See: `reports/model_evaluation_summary.md`

---

## 💼 Business Impact

If applied to 1,000 properties:

* ~60–70% filtered as weak investments
* ~20–30% flagged for further review
* ~10–15% identified as strong opportunities

👉 Enables **fast, scalable deal screening**

---

## 🖥️ Streamlit App

The app allows users to:

* input property details
* run full financial + ML evaluation
* receive:

  * investment classification
  * financial breakdown
  * investor recommendation

---

## ⚙️ How to Run

```bash
git clone https://github.com/your-username/AI-Property-Investment-Intelligence.git
cd AI-Property-Investment-Intelligence
pip install -r requirements.txt
python src/run_pipeline.py
streamlit run app/streamlit_app.py
```

---

## ⚠️ Data Notes

* Dataset reflects real market imbalance (mostly weak deals)
* Synthetic expansion used only for demo scaling
* Model trained on original data

---

## 🛠️ Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Streamlit
* SHAP
* Joblib

---

## 📌 Key Strength of This Project

This project does not just predict outcomes — it:

✔ embeds financial logic
✔ reflects real investment decision-making
✔ outputs actionable recommendations

---

## 👤 Author

Ashley Mathabatha
Data Scientist | Property Investment Intelligence
