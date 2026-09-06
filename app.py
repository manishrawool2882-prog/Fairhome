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
town_pairs = [[t, town_to_county.get(t, "").title()] for t in towns]
NAT_AVG = round(global_mean / 1000) * 1000
PREDICT_YEAR = 2024


def resolve_town(raw):
    if not raw:
        return None
    t = raw.strip().upper()
    if t in town_means.index:
        return t
    for name in towns:
        if name.startswith(t):
            return name
    return None


def build_features(form):
    town = resolve_town(form.get("towncity", ""))
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
    return pd.DataFrame([row])[feature_order], town


def explain(X, ptype, town, top_n=4):
    shap_row = explainer.shap_values(X)[0]
    impact = pd.Series(shap_row, index=feature_order)
    impact = impact.reindex(impact.abs().sort_values(ascending=False).index)
    ptype_done = False
    reasons = []
    town_avg = round(town_means.get(town, global_mean) / 1000) * 1000
    for feat, val in impact.items():
        if feat == "year":
            continue
        up = val > 0
        amount = round(abs(val) / 1000) * 1000
        v = X.iloc[0][feat]
        detail = ""
        if feat == "TOTAL_FLOOR_AREA":
            phrase = f"At {int(v)} m\u00b2, it's {'larger' if up else 'smaller'} than average"
            detail = "Compared with a typical UK home"
        elif feat == "towncity_enc":
            phrase = "This area sells above average" if up else "This area sells below average"
            detail = f"Local average \u2248 \u00a3{town_avg:,.0f} vs \u00a3{NAT_AVG:,.0f} nationally"
        elif feat == "county_enc":
            phrase = "The wider region sells above average" if up else "The wider region sells below average"
            detail = f"Compared with the \u00a3{NAT_AVG:,.0f} national average"
        elif feat == "build_year":
            yr = int(v)
            decade = (yr // 10) * 10
            if yr >= 2000:
                phrase = "A relatively modern property"
            elif yr >= 1970:
                phrase = "A mid-to-late 20th century property"
            else:
                phrase = "An older, period property"
            detail = f"Built around the {decade}s" + (" or later" if yr >= 2010 else "")
        elif feat == "CURRENT_ENERGY_RATING":
            phrase = "A better energy rating" if up else "A lower energy rating"
            detail = "Most UK homes are rated D"
        elif feat.startswith("propertytype"):
            if ptype_done:
                continue
            ptype_done = True
            phrase = f"It's {ptype.lower()}"
            if ptype == "Detached":
                detail = "Detached homes usually sell for the most"
            elif ptype == "Flat":
                detail = "Flats usually sell for less than houses"
            elif ptype == "Terraced":
                detail = "Terraced homes are often more affordable"
            elif ptype == "Semi-Detached":
                detail = "Semi-detached homes sit in the mid-range"
            else:
                detail = "Property type affects the typical price"
        elif feat == "duration_Leasehold":
            phrase = "It's leasehold" if v else "It's freehold"
            detail = "Leasehold can lower value" if v else "Freehold is usually valued higher"
        else:
            continue
        reasons.append({"phrase": phrase, "detail": detail, "up": up, "amount": f"\u00a3{amount:,.0f}"})
        if len(reasons) >= top_n:
            break
    return reasons


def monthly_payment(loan, annual_rate_pct, years=25):
    r = (annual_rate_pct / 100) / 12
    n = years * 12
    if r == 0:
        return loan / n
    return loan * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def affordability(price, deposit_pct=10, base_rate=5.5, years=25):
    loan = price * (1 - deposit_pct / 100)
    rows = []
    for bump in [0, 1, 2, 3]:
        rate = base_rate + bump
        pay = monthly_payment(loan, rate, years)
        rows.append({
            "label": f"{rate:.1f}%" + (" (current)" if bump == 0 else f" (+{bump}%)"),
            "payment": f"\u00a3{pay:,.0f}",
        })
    return {
        "deposit": f"\u00a3{price * deposit_pct/100:,.0f}",
        "loan": f"\u00a3{loan:,.0f}",
        "rows": rows,
    }


@app.route("/")
def home():
    return render_template("index.html", town_pairs=town_pairs, selected={})


@app.route("/predict", methods=["POST"])
def predict():
    X, matched_town = build_features(request.form)
    price = model.predict(X)[0]
    reasons = explain(X, request.form["propertytype"], matched_town)
    afford = affordability(price)
    return render_template("index.html", town_pairs=town_pairs, price=f"\u00a3{price:,.0f}",
                           reasons=reasons, afford=afford, selected=request.form,
                           matched_town=matched_town)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
