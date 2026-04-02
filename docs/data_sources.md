# Data Sources Property Investment Intelligence SA

## 1. Overview

In this project, I am building a property investment intelligence system that depends on multiple types of data.

To evaluate a financed property investment properly, I need more than just a sale listing. I need to combine:

- property listing data
- rental market data
- area-level signals
- financing assumptions
- spreadsheet-derived business logic

This document defines the data sources I plan to use, what fields I expect from each source, and how each source contributes to the overall system.

---

## 2. Source Categories

I divide the required data into four main source groups:

1. Property listings
2. Rental listings
3. Area-level signals
4. User inputs and assumptions

---

## 3. Property Listings

### Purpose
I use property listings to identify candidate investment properties and capture their structural and pricing characteristics.

### Expected fields
- listing_id
- listing_url
- title
- purchase_price
- suburb
- city
- province
- property_type
- bedrooms
- bathrooms
- parking_spaces
- floor_area_sqm
- levies
- rates
- description
- listing_date

### Why this source matters
This is the starting point of the pipeline. It defines the actual properties I want to screen.

### Output destination
`data/raw/listings/`

---

## 4. Rental Listings

### Purpose
I use rental listings to estimate achievable monthly rent and derive area rental comparables.

### Expected fields
- rental_listing_id
- rental_url
- monthly_rent
- suburb
- city
- province
- property_type
- bedrooms
- bathrooms
- floor_area_sqm
- furnished_flag
- listing_date

### Why this source matters
A property investment decision depends heavily on rental income. Without reliable rental comparisons, the investment model becomes weak.

### Output destination
`data/raw/rentals/`

---

## 5. Area-Level Signals

### Purpose
I use area-level signals to give the model local market context.

### Example signals
- suburb average listing price
- suburb average rent
- listing density
- rental density
- rent-to-price ratio by area
- demand proxy
- vacancy proxy
- nearby amenities (if available)

### Why this source matters
A property should not only be evaluated in isolation. I want the model to understand whether the deal is attractive relative to its local market.

### Output destination
`data/raw/area_data/`

---

## 6. User Inputs and Assumptions

### Purpose
Not every important variable can be scraped. Some inputs must be provided directly by the investor or set as modeling assumptions.

### Examples
- deposit
- interest_rate
- loan_term_years
- maintenance_rate
- vacancy_rate
- other_income
- investor thresholds

### Why this source matters
Because this project models financed investments, financing assumptions are central to the output.

### Output destination
- workbook exports
- config files
- application form inputs

---

## 7. Spreadsheet-Derived Logic

### Purpose
The uploaded rental workbook provides the original investment logic that I am converting into code.

### Role in the project
I use it to define:
- financial formulas
- assumptions
- output metrics
- amortization logic

### Why this source matters
It provides the business logic foundation for the system and ensures the Python implementation remains aligned to investor reasoning.

### Output destination
`data/raw/workbook_exports/`

---

## 8. Data Source Roles in the Pipeline

Each source plays a different role:

### Property listings
Define the opportunity set

### Rental listings
Estimate income potential

### Area data
Provide local market context

### Assumptions
Allow financing and cost modeling

### Workbook logic
Defines decision calculations

Together, these sources allow me to model a full financed property screening workflow.

---

## 9. Source Limitations and Risks

As I build the system, I need to account for several data risks:

- missing levies and rates in listings
- inconsistent formatting across sources
- duplicates across listing pages
- incomplete floor area information
- rental listings that are not true comparables
- proxies instead of true vacancy/demand data
- assumptions that vary by investor

These risks will need to be handled during cleaning, validation, and feature engineering.

---

## 10. Summary

In this project, I am deliberately combining listing data, rental comparables, area context, and financing assumptions to create a richer investment screening system.

This is what allows the project to go beyond a simple listing classifier and become a true investment intelligence pipeline.
