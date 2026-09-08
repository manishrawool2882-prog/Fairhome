# Fairhome — Explainable ML House Price Valuation

# Fairhome

An explainable machine-learning web application that estimates house prices in
England and Wales, explains each estimate in plain English using SHAP, and
stress-tests mortgage affordability under rising interest rates.

**MSc Data Analytics with Banking & Finance dissertation project — Sheffield Hallam University.**

Live app: https://fairhome.onrender.com

## What it does
- Predicts a property's value from its characteristics (town, floor area, type,
  energy rating, tenure, age) using an XGBoost model.
- Explains the estimate factor-by-factor in plain English (SHAP TreeExplainer),
  showing how each feature raised or lowered the price relative to the national average.
- Checks mortgage affordability, showing monthly repayments at the current rate
  and at +1%, +2%, and +3%.

## Research question
Does a tool that explains the price and checks affordability help ordinary users
understand and trust a house-buying decision more than a single unexplained price estimate?

## Tech stack
Python · pandas · scikit-learn · XGBoost · SHAP · Flask · HTML/CSS · deployed on Render.

## Model
XGBoost, selected by comparing four models (Linear Regression, Random Forest,
Extra Trees, XGBoost). Final performance: MAE ≈ £52,000, R² ≈ 0.83 on a held-out test set.

## Data
UCL linked HM Land Registry Price Paid Data + EPC dataset (England and Wales,
2015–2024 subset). Open Government Licence v3.0 / CC BY 4.0. The raw dataset and
trained model files are not included in this repository.

## Running locally
1. `pip install -r requirements.txt`
2. `python app.py`
3. Open `http://localhost:5000`

## Project structure
- `app.py` — Flask application (prediction, SHAP explanation, affordability)
- `templates/index.html` — front-end
- `requirements.txt`, `Procfile`, `runtime.txt` — deployment config
- (model bundle and data files excluded via `.gitignore`)

## Author
Manish Anand Rawool — MSc Data Analytics with Banking & Finance, Sheffield Hallam University.
Supervisor: Caren Fernandes.
