# Labeling Strategy - Property Investment Intelligence SA

## 1. Overview

In this project, I do not have a perfect historical dataset showing which properties ultimately became successful investments over many years.

Because of that, I am creating an initial supervised learning target using **rule-based proxy labels**.

This is a practical and realistic approach because the project is designed to support early-stage deal screening, not long-term realized return measurement.

The goal of the label is to classify each property into one of three categories:

- Strong Investment
- Moderate Investment
- Weak Investment

---

## 2. Why I Am Using Proxy Labels

A truly perfect label would require future outcomes such as:
- realized rental performance
- actual occupancy
- final sale value
- long-term cash returns

That kind of data is usually unavailable at the start of a project.

To solve this, I am using the financial engine from Day 2 to generate decision-oriented labels based on the metrics an investor would use during screening.

This allows me to:
- create a consistent target variable
- train baseline models
- test whether the model can replicate investment reasoning at scale

---

## 3. Core Labeling Metrics

I will base the label primarily on:

- cash_flow
- DSCR
- rental_yield
- ROI

These metrics reflect:
- sustainability
- debt coverage
- return potential
- capital efficiency

---

## 4. Label Definitions

### 4.1 Strong Investment

A property is labeled **Strong Investment** when it shows healthy financed performance.

Example rule logic:
- cash_flow >= 0
- DSCR > 1.2
- rental_yield >= target threshold
- ROI >= target threshold

This means the property is likely able to sustain itself and generate acceptable returns.

---

### 4.2 Moderate Investment

A property is labeled **Moderate Investment** when it is not clearly strong, but may still deserve further review.

Example rule logic:
- slightly negative cash_flow
- DSCR close to 1
- moderate yield
- acceptable but not strong ROI

This means the property may still work depending on investor strategy, negotiation, or improved assumptions.

---

### 4.3 Weak Investment

A property is labeled **Weak Investment** when it performs poorly under financing pressure.

Example rule logic:
- strongly negative cash_flow
- DSCR < 1
- low yield
- weak ROI

This means the investor is likely taking on too much financing pressure relative to income.

---

## 5. Example Threshold Framework

The exact thresholds may be refined after exploratory analysis, but the initial starting point could be:

### Strong Investment
- cash_flow >= 0
- DSCR >= 1.20
- rental_yield >= 0.10
- ROI >= 0.08

### Moderate Investment
- cash_flow between -1000 and 0
- DSCR between 1.00 and 1.20
- rental_yield between 0.08 and 0.10
- ROI between 0.04 and 0.08

### Weak Investment
- cash_flow < -1000
- DSCR < 1.00
- rental_yield < 0.08
- ROI < 0.04

These are starting business rules, not fixed truths.

---

## 6. Labeling Logic Options

### Option A: Strict rule intersection
A property must satisfy all strong conditions to receive the strong label.

### Option B: Score-based logic
I assign points for each metric and label based on total score.

Example:
- strong condition met = 2 points
- moderate condition met = 1 point
- weak condition met = 0 points

This approach may produce more balanced labels.

### Option C: Hybrid logic
I use hard filters for critical risk metrics such as DSCR, then score the remaining dimensions.

This is likely the most realistic approach.

---

## 7. Recommended Approach

For this project, I recommend a **hybrid rule-based labeling strategy**:

1. Use DSCR and cash_flow as hard risk filters
2. Use rental_yield and ROI as scoring metrics
3. Assign final labels based on overall financed investment quality

This reflects how investors actually think:
- first check survival
- then check attractiveness

---

## 8. Important Modeling Note

Because these labels are derived from financial metrics, I need to be careful when training the classifier.

If I train the model on the exact same variables used to generate the label, then the model may simply learn the rule logic rather than broader investment patterns.

To manage this, I can build two model versions:

### Version 1: Rules replication model
Includes core financial metrics and tests whether the model can reproduce the screening framework.

### Version 2: Broader predictive model
Uses upstream property, rental, and area variables to predict the label without relying too heavily on direct rule variables.

This gives me both:
- interpretability
- a more realistic predictive setup

---

## 9. Label Quality Checks

After creating labels, I will validate them by checking:

- class balance
- average cash_flow by label
- average DSCR by label
- average yield by label
- sample properties from each class

If the label distribution is too skewed or unrealistic, I will refine thresholds.

---

## 10. Summary

In this project, I am using financially grounded proxy labels to create a practical supervised learning target.

This allows me to convert investor reasoning into a repeatable training framework by classifying each property as:

- Strong Investment
- Moderate Investment
- Weak Investment

The labeling strategy is designed to reflect real financed property decision-making while remaining flexible enough to improve as more data becomes available.
