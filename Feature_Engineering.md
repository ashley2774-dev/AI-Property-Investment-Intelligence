# Feature Definition - Property Investment Intelligence SA

## 1. Overview

In this project, I am designing a feature set that helps the model evaluate whether a financed residential property is likely to be a strong, moderate, or weak investment opportunity.

The features are built from four main sources:

- property listing data
- rental data
- area-level signals
- engineered financial metrics

My goal is to create a feature set that captures:
- affordability
- rental income potential
- financing pressure
- operating cost burden
- investment sustainability
- property and area attractiveness

---

## 2. Feature Design Principles

As I define the feature set, I follow these principles:

### 2.1 Business relevance
Every feature should reflect something an investor would actually care about.

### 2.2 Predictive usefulness
Features should help distinguish good deals from weak ones.

### 2.3 Avoid leakage
I do not include variables that directly reveal the final label in a way that would make training unrealistic.

### 2.4 Scalability
Features should be reproducible from collected data and assumptions.

### 2.5 Interpretability
The final model should be explainable, so I prefer features that can be understood by investors.

---

## 3. Feature Groups

### 3.1 Property Features

These describe the property itself.

- purchase_price
- property_type
- bedrooms
- bathrooms
- parking_spaces
- floor_area_sqm
- suburb
- city
- province

These features capture the structure, size, and location of the property.

---

### 3.2 Rental Features

These describe the expected income potential.

- monthly_rent
- other_income
- gross_income
- area_avg_rent
- rent_per_sqm
- rent_gap_to_area_avg

These features help quantify how strong the property’s income profile is relative to the local market.

---

### 3.3 Financing Features

Because this is a financed investment model, financing variables are central.

- deposit
- deposit_pct
- loan_amount
- loan_to_value_ratio
- interest_rate
- loan_term_years
- monthly_bond_payment

These features help represent leverage, affordability, and debt burden.

---

### 3.4 Expense Features

These represent recurring operating costs.

- levies
- rates
- maintenance
- vacancy_cost
- total_expenses
- expense_ratio

These features help measure the cost burden associated with the property.

---

### 3.5 Financial Performance Features

These are the most important engineered features in the project.

- NOI
- cash_flow
- annual_cash_flow
- DSCR
- rental_yield
- ROI

These variables capture the financial quality of the deal.

---

### 3.6 Area and Market Features

These represent local market context.

- suburb_avg_listing_price
- suburb_avg_rent
- price_to_area_avg_ratio
- rent_to_area_avg_ratio
- listing_density
- rental_density
- demand_proxy
- vacancy_proxy

These features help the model understand whether a deal is strong relative to its surrounding market.

---

### 3.7 Text-Derived Features

If listing descriptions are available, I may engineer simple text features such as:

- description_length
- premium_keyword_count
- renovation_keyword_flag
- urgent_sale_flag
- furnished_flag

These features can capture hidden signals from the listing text.

---

## 4. Engineered Feature Definitions

### 4.1 Deposit Percentage

deposit_pct = deposit / purchase_price

This represents how much of the purchase is funded by the investor.

---

### 4.2 Loan-to-Value Ratio

loan_to_value_ratio = loan_amount / purchase_price

This reflects leverage intensity.

---

### 4.3 Rent per Square Meter

rent_per_sqm = monthly_rent / floor_area_sqm

This allows comparison across differently sized units.

---

### 4.4 Expense Ratio

expense_ratio = total_expenses / gross_income

This measures how much income is consumed by operating costs.

---

### 4.5 Rent Gap to Area Average

rent_gap_to_area_avg = monthly_rent - area_avg_rent

This helps compare a property to local rental norms.

---

### 4.6 Price to Area Average Ratio

price_to_area_avg_ratio = purchase_price / suburb_avg_listing_price

This shows whether the purchase price is above or below the local norm.

---

## 5. Features to Exclude from Training

To avoid leakage or poor modeling decisions, I will exclude some fields from training.

### 5.1 Identifiers
- property_id
- listing_url
- title
- raw_address

These are useful operationally but not as model inputs.

### 5.2 Post-decision metadata
- scraped_timestamp
- manual_notes

These do not reflect the property itself.

### 5.3 Direct target duplicates
If my target label is built directly from a metric, I will be careful about whether that metric should be included in the model.

For example:
- if labels are created mostly from cash_flow and DSCR,
- then training the model on those exact same features may create a model that simply reproduces rules instead of learning broader patterns.

I may use two setups:
- a rules benchmark model
- a broader predictive model with limited direct label leakage

---

## 6. Final Modeling View

My modeling table will likely include:

### Raw features
- purchase_price
- bedrooms
- bathrooms
- floor_area_sqm
- suburb
- property_type
- monthly_rent
- levies
- rates
- deposit
- interest_rate
- loan_term_years

### Engineered features
- loan_amount
- deposit_pct
- monthly_bond_payment
- maintenance
- vacancy_cost
- total_expenses
- NOI
- expense_ratio
- rental_yield
- DSCR
- ROI
- price_to_area_avg_ratio
- rent_gap_to_area_avg

### Target
- investment_label

---

## 7. Summary

In this project, I am designing features that reflect how real investors think about property deals.

Rather than relying only on generic listing variables, I am combining:
- structural property features
- rental market signals
- debt and financing metrics
- operating cost measures
- investment performance indicators

This gives the model a much stronger foundation for identifying which properties deserve further investigation.
