# Financial Logic — Property Investment Intelligence SA

## 1. Overview

In this project, I am building a financial engine that replicates and improves the way property investors in South Africa evaluate investment opportunities using spreadsheets.

Instead of treating this as a cash purchase model, I am explicitly modeling **real-world financed property investments**, where properties are acquired using a bank loan (bond). This introduces key dynamics such as:
- leverage
- interest costs
- loan amortization
- debt-driven risk

My goal is to translate a manual, spreadsheet-based workflow into a structured and scalable system that can:
- evaluate multiple properties consistently
- automate financial calculations
- support rule-based and machine learning decision-making

---

## 2. How I Structured the Model

To make the system modular and reusable, I break the financial logic into five components:

1. Purchase & Financing  
2. Rental Income  
3. Operating Expenses  
4. Bond Repayment (Debt)  
5. Investment Performance Metrics  

Each component is designed to be independently computed and then combined into a final investment evaluation.

---

## 3. Data Sources and Lineage

As I designed this system, I made a clear distinction between where each piece of data comes from:

### Property Listings (Scraped)
I use listing data to capture:
- purchase_price
- location
- levies (where available)
- property characteristics

### Rental Listings (Scraped)
I estimate income using:
- monthly_rent
- rental comparables in the same area

### User Inputs and Assumptions
Some values are not directly observable, so I define them as inputs:
- deposit
- interest_rate
- loan_term_years
- maintenance_rate
- vacancy_rate

### Derived Fields (Calculated by the Model)
From these inputs, I compute:
- loan_amount
- bond_payment
- total_expenses
- NOI
- cash_flow
- DSCR
- ROI
- equity

This separation helps me keep the system transparent, debuggable, and production-ready.

---

## 4. Purchase and Financing Logic

### 4.1 Total Cash Invested

I define the investor’s upfront capital as:

total_cash_invested = deposit + transfer_costs

This reflects the actual cash required to enter the deal.

---

### 4.2 Loan Amount

Since the property is financed, I calculate:

loan_amount = purchase_price − deposit

This represents the amount borrowed from the bank.

---

## 5. Bond Repayment (Core of the Model)

Financing is the most important part of this model because it determines whether the property is sustainable.

### 5.1 Monthly Interest Rate

monthly_interest_rate = interest_rate / 12

---

### 5.2 Total Number of Payments

total_payments = loan_term_years × 12

---

### 5.3 Monthly Bond Payment

I calculate the monthly repayment using the standard amortization formula:

bond_payment = loan_amount × [r(1+r)^n] / [(1+r)^n − 1]

Where:
- r is the monthly interest rate
- n is the total number of payments

This value becomes the **largest fixed monthly cost**, and it directly drives cash flow outcomes.

---

## 6. Rental Income

### 6.1 Gross Monthly Income

I calculate total income as:

gross_income = monthly_rent + other_income

In most cases, rental income is the dominant component.

---

## 7. Operating Expenses

To reflect real-world ownership, I include recurring costs.

### 7.1 Expense Components

I model:
- levies
- municipal rates
- maintenance
- vacancy

---

### 7.2 Maintenance

I estimate maintenance as:

maintenance = monthly_rent × maintenance_rate

---

### 7.3 Vacancy

I account for rental gaps:

vacancy_cost = monthly_rent × vacancy_rate

---

### 7.4 Total Expenses

I combine all costs:

total_expenses = levies + rates + maintenance + vacancy_cost

---

## 8. Net Operating Income (NOI)

To understand the property’s performance before debt, I calculate:

NOI = gross_income − total_expenses

This helps me isolate the property’s operational strength independent of financing.

---

## 9. Cash Flow (Primary Decision Metric)

The most important metric I use is:

cash_flow = NOI − bond_payment

This tells me whether the property can sustain itself.

### Interpretation:
- cash_flow ≥ 0 → the property pays for itself  
- cash_flow < 0 → the investor must cover the shortfall  

This is the core metric used in screening decisions.

---

## 10. Debt Service Coverage Ratio (DSCR)

To assess risk, I calculate:

DSCR = NOI / bond_payment

### Interpretation:
- DSCR > 1.2 → strong investment  
- DSCR ≈ 1 → borderline  
- DSCR < 1 → high risk  

This metric aligns closely with how lenders and experienced investors evaluate deals.

---

## 11. Rental Yield

I calculate yield as a simple benchmark:

annual_rent = monthly_rent × 12  

rental_yield = annual_rent / purchase_price  

This helps me compare properties quickly, although it does not account for financing.

---

## 12. Return on Investment (ROI)

Since the investment is leveraged, I calculate ROI based on actual cash invested:

annual_cash_flow = cash_flow × 12  

ROI = annual_cash_flow / total_cash_invested  

This reflects the true return experienced by the investor.

---

## 13. Amortization and Equity Growth

Each bond payment consists of:
- an interest component
- a principal repayment component

Over time, I track:
- decreasing loan balance
- increasing equity

---

### 13.1 Remaining Balance

I update the loan balance each period based on:
- interest charged
- principal repaid

---

### 13.2 Equity

I define equity as:

equity = property_value − remaining_balance

This allows me to capture long-term wealth creation, not just monthly performance.

---

## 14. Investment Decision Logic

Before introducing machine learning, I define rule-based screening logic.

### Strong Investment
- cash_flow ≥ 0  
- DSCR > 1.2  

### Moderate Investment
- slightly negative cash flow  
- DSCR close to 1  

### Weak Investment
- strongly negative cash flow  
- DSCR < 1  

I will use these rules later to generate labels for supervised learning.

---

## 15. How This Feeds the ML Pipeline

This financial engine is central to the entire system.

I use it to:
- generate core financial features
- create labeled training data
- standardize how all properties are evaluated

It acts as the bridge between:
- raw data collection
- feature engineering
- machine learning predictions

---

## 16. Key Assumptions

To make the model practical, I assume:

- rental income is relatively stable (adjusted using vacancy rate)
- maintenance can be approximated as a percentage of rent
- interest rates remain constant over the loan term
- early-stage decisions are driven primarily by cash flow and debt coverage

These assumptions can be refined as the system evolves.

---

## 17. Summary

Through this financial model, I am transforming a manual spreadsheet process into a scalable, data-driven system.

By explicitly modeling:
- financing (bond repayments)
- operating costs
- income generation
- investment risk

I am building a realistic foundation for:
- automated deal screening
- consistent property evaluation
- machine learning-driven investment recommendations
