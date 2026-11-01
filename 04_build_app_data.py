"""
Step 4 - Build small data files for the Streamlit app

The full cleaned dataset is ~80 MB, too big to ship with the app. This script creates small
summary tables (city stats, heatmap tables) and a 25,000-listing sample with model predictions.

Run (after 03_train_models.py):
    python 04_build_app_data.py
Output: outputs/app_data/*.csv
"""
import os

import joblib
import numpy as np
import pandas as pd

BASE = "outputs"
OUT = os.path.join(BASE, "app_data")
os.makedirs(OUT, exist_ok=True)

state = joblib.load(os.path.join(BASE, "preprocessor.joblib"))
clf = joblib.load(os.path.join(BASE, "models", "classifier.joblib"))
reg = joblib.load(os.path.join(BASE, "models", "regressor.joblib"))
clean = pd.read_csv(os.path.join(BASE, "cleaned_real_estate.csv"))
ready = pd.read_csv(os.path.join(BASE, "model_ready_real_estate.csv"))
P = "Price_per_SqFt_Final"

# ---- aggregates over the FULL dataset ----
city = clean.groupby(["State", "City"]).agg(
    listings=("ID", "size"), avg_price_lakhs=("Price_in_Lakhs", "mean"),
    avg_ppsf=(P, "mean"), pct_good=("Good_Investment", lambda s: s.mean() * 100),
    avg_infra=("Infrastructure_Score", "mean")).reset_index()
city.round(2).to_csv(os.path.join(OUT, "city_stats.csv"), index=False)

state_t = clean.pivot_table(index="State", columns="Property_Type", values=P, aggfunc="mean")
state_t.round(0).to_csv(os.path.join(OUT, "heat_state_type_ppsf.csv"))

city_bhk = clean.pivot_table(index="City", columns="BHK", values="Price_in_Lakhs", aggfunc="mean")
city_bhk.round(1).to_csv(os.path.join(OUT, "heat_city_bhk_price.csv"))

state_good = clean.pivot_table(index="State", columns="Public_Transport_Accessibility",
                               values="Good_Investment", aggfunc="mean") * 100
state_good.round(1).to_csv(os.path.join(OUT, "heat_state_transport_good.csv"))

trend = clean.groupby(["Year_Built", "Property_Type"])[P].mean().unstack()
trend.round(0).to_csv(os.path.join(OUT, "trend_year_built_ppsf.csv"))

infra = clean.assign(infra_band=pd.cut(clean["Infrastructure_Score"], [0, 20, 40, 60, 80, 100],
                                       include_lowest=True).astype(str)) \
    .groupby("infra_band").agg(pct_good=("Good_Investment", lambda s: s.mean() * 100),
                               avg_ppsf=(P, "mean"), listings=("ID", "size")).reset_index()
infra.round(2).to_csv(os.path.join(OUT, "infra_bands.csv"), index=False)

# ---- sample listings with model predictions ----
idx = np.random.RandomState(7).choice(len(clean), 25000, replace=False)
s = clean.loc[idx, ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
                    "Price_in_Lakhs", "Year_Built", "Age_of_Property", "Furnished_Status",
                    "Public_Transport_Accessibility", "Availability_Status", "Owner_Type"]].copy()
s["Price_per_SqFt"] = clean.loc[idx, P].round(0)
s["Infrastructure_Score"] = clean.loc[idx, "Infrastructure_Score"].round(1)
s["Model_P_Good"] = clf.predict_proba(ready.loc[idx, state["feature_cols_classification"]])[:, 1].round(3)
s["Model_Price_5Y_Lakhs"] = reg.predict(ready.loc[idx, state["feature_cols_regression"]]).round(1)
s.sort_values("Model_P_Good", ascending=False).to_csv(os.path.join(OUT, "listings_sample.csv"), index=False)

for f in sorted(os.listdir(OUT)):
    print(f"{f:<34}{os.path.getsize(os.path.join(OUT, f)) / 1024:>9.0f} KB")
