from flask import Flask, render_template, request
import joblib
import pandas as pd
import shap

app = Flask(__name__)

bundle = joblib.load("app_bundle.pkl")
model = bundle["model"]
town_means = bundle["town_means"]
county_means = bundle["county_means"]
global_mean = bundle["global_mean"]
rating_map = bundle["rating_map"]
band_to_year = bundle["band_to_year"]
feature_order = bundle["feature_order"]
town_to_county = bundle["town_to_county"]
base_value = bundle["base_value"]

explainer = shap.TreeExplainer(model)
towns = sorted(town_means.index.tolist())
PREDICT_YEAR = 2024

def build_features(form):
    town = form["towncity"]
    ptype = form["propertytype"]
    county = town_to_county.get(town)
    row = {
        "CURRENT_ENERGY_RATING": rating_map[form["energy"]],
        "TOTAL_FLOOR_AREA": float(form["floor_area"]),
        "year": PREDICT_YEAR,
        "propertytype_Flat": 1 if ptype == "Flat" else 0,
        "propertytype_Semi-Detached": 1 if ptype == "Semi-Detached" else 0,
        "propertytype_Terraced": 1 if ptype == "Terraced" else 0,
        "duration_Leasehold": 1 if form["duration"] == "Leasehold" else 0,
        "oldnew_New Build": 0,
        "build_year": band_to_year.get(form["age_band"], 2010),
        "age_known": 1,
        "towncity_enc": town_means.get(town, global_mean),
        "county_enc": county_means.get(county, global_mean),
    }
    return pd.DataFrame([row])[feature_order]

def explain(X, top_n=3):
    shap_row = explainer.shap_values(X)[0]
    impact = pd.Series(shap_row, index=feature_order)
    impact = impact.reindex(impact.abs().sort_values(ascending=False).index)
    reasons = []
    for feat, val in impact.items():
        if feat == "year":
            continue
        up = val > 0
        amount = round(abs(val) / 1000) * 1000
        v = X.iloc[0][feat]
        if feat == "TOTAL_FLOOR_AREA":
            phrase = f"At {int(v)} m2, it's {'larger' if up else 'smaller'} than average"
        elif feat == "towncity_enc":
            phrase = "The area sells above average" if up else "The area sells below average"
        elif feat == "county_enc":
            phrase = "The wider region sells above average" if up else "The wider region sells below average"
        elif feat == "build_year":
            phrase = "A newer property" if up else "An older property"
        elif feat == "CURRENT_ENERGY_RATING":
            phrase = "A good energy rating" if up else "A lower energy rating"
        elif feat == "propertytype_Semi-Detached" and v:
            phrase = "It's semi-detached"
        elif feat == "propertytype_Terraced" and v:
            phrase = "It's terraced"
        elif feat == "propertytype_Flat" and v:
            phrase = "It's a flat"
        elif feat.startswith("propertytype"):
            phrase = "Its property type"
        elif feat == "duration_Leasehold":
            phrase = "It's leasehold" if v else "It's freehold"
        else:
            phrase = "Its property details"
        verb = "adds" if up else "reduces"
        reasons.append(f"{phrase} \u2014 {verb} about \u00a3{amount:,.0f}.")
        if len(reasons) >= top_n:
            break
    return reasons

@app.route("/")
def home():
    return render_template("index.html", towns=towns, selected={})

@app.route("/predict", methods=["POST"])
def predict():
    X = build_features(request.form)
    price = model.predict(X)[0]
    reasons = explain(X)
    return render_template("index.html", towns=towns,
                           price=f"\u00a3{price:,.0f}", reasons=reasons,
                           selected=request.form)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
