"""
Helpers shared by the Streamlit app and the tests.

build_features() turns the details a user types into a form into exactly the same feature
columns that 01_preprocessing.py created for training (same formulas, same scaler, same
encodings), so predictions are consistent with the trained models.
"""

import os

import joblib
import numpy as np
import pandas as pd

CURRENT_YEAR = 2025            # must match 01_preprocessing.py
BASE_GROWTH, MAX_EXTRA_GROWTH, AGE_PENALTY = 0.05, 0.04, 0.01
TRANSPORT_MAP = {"low": 1, "medium": 2, "high": 3}
HERE = os.path.dirname(os.path.abspath(__file__))


def load_artifacts(base=None):
    base = base or os.path.join(HERE, "outputs")
    state = joblib.load(os.path.join(base, "preprocessor.joblib"))
    clf = joblib.load(os.path.join(base, "models", "classifier.joblib"))
    reg = joblib.load(os.path.join(base, "models", "regressor.joblib"))
    return state, clf, reg


def _mm(x, lo_hi):
    lo, hi = lo_hi
    return (x - lo) / (hi - lo) if hi > lo else 0.0


def derive_raw_features(inp, state):
    """Compute the un-scaled engineered features for one property (dict in, dict out)."""
    mm = state["minmax"]
    size = float(inp["size"])
    price = float(inp["price"])
    bhk = int(inp["bhk"])
    floor_no = int(inp["floor_no"])
    total_floors = max(int(inp["total_floors"]), floor_no)       # same repair as preprocessing
    age = max(CURRENT_YEAR - int(inp["year_built"]), 0)

    ppsf = price * 100000 / size                                  # Rs per sqft (uncapped)
    b = state["outlier_bounds"]
    lo_s, hi_s = b["Size_in_SqFt"]
    lo_p, hi_p = b["Price_in_Lakhs"]
    lo_q, hi_q = b["Price_per_SqFt_Calc"]

    transport = TRANSPORT_MAP[str(inp["transport"]).lower()]
    furn = str(inp["furnished"]).lower()
    furn_level = 0 if "unfurn" in furn else 1 if "semi" in furn else 2
    amenities = list(inp.get("amenities", []))

    schools_mm = _mm(inp["schools"], mm["schools"])
    infra = 100 * (0.40 * schools_mm
                   + 0.30 * _mm(inp["hospitals"], mm["hospitals"])
                   + 0.30 * _mm(transport, mm["transport"]))
    age01 = _mm(min(age, 60), mm["age"])
    growth = BASE_GROWTH + MAX_EXTRA_GROWTH * infra / 100 - AGE_PENALTY * age01
    city_med = state["city_median_ppsf"][inp["city"]]

    raw = {
        "BHK": bhk,
        "Size_in_SqFt_Capped": min(max(size, lo_s), hi_s),
        "Price_in_Lakhs_Capped": min(max(price, lo_p), hi_p),
        "Price_per_SqFt_Calc_Capped": min(max(ppsf, lo_q), hi_q),
        "Age_of_Property": age,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Floor_Ratio": floor_no / total_floors if total_floors else 0,
        "Nearby_Schools": inp["schools"],
        "Nearby_Hospitals": inp["hospitals"],
        "Transport_Score": transport,
        "Furnished_Level": furn_level,
        "Parking_Count": 1 if inp["parking"] == "Yes" else 0,
        "Has_Security": 1 if inp["security"] == "Yes" else 0,
        "Amenity_Count": len(amenities),
        "Rooms_per_1000SqFt": bhk / size * 1000,
        "Avg_Room_Size": size / bhk if bhk else size,
        "School_Density_Score": schools_mm * 10,
        "Infrastructure_Score": infra,
        "PPSF_vs_City_Median": ppsf / city_med,
        "City_Median_PPSF": city_med,
        "Annual_Growth_Rate": growth,
    }
    info = dict(ppsf=ppsf, city_median_ppsf=city_med, ppsf_vs_city=ppsf / city_med,
                infrastructure_score=infra, rule_growth=growth, age=age)
    return raw, info


def build_features(inp, state):
    """Return (X_classifier, X_regressor, info) as one-row DataFrames in the training column order."""
    raw, info = derive_raw_features(inp, state)

    num = pd.DataFrame([raw])[state["numeric_cols"]]
    scaled = pd.DataFrame(state["scaler"].transform(num), columns=state["numeric_cols"])
    row = {c: float(scaled[c].iloc[0]) for c in state["numeric_cols"]}

    row["Locality_Freq"] = state["freq_maps"]["Locality"].get(inp["locality"], 0.0)
    row["City_Freq"] = state["freq_maps"]["City"].get(inp["city"], 0.0)

    for a in state["top_amenities"]:                                # amenity flags
        row["Amn_" + a.replace(" ", "_")] = 1 if a in inp.get("amenities", []) else 0

    wanted = {                                                      # one-hot columns (drop_first)
        f"State_{inp['state']}", f"Property_Type_{inp['property_type']}",
        f"Facing_{inp['facing']}", f"Owner_Type_{inp['owner_type']}",
        f"Availability_Status_{inp['availability']}", f"City_{inp['city']}",
    }
    for c in state["onehot_columns"]:
        row[c] = 1 if c in wanted else 0

    full = pd.DataFrame([row])
    Xc = full.reindex(columns=state["feature_cols_classification"], fill_value=0)
    Xr = full.reindex(columns=state["feature_cols_regression"], fill_value=0)
    return Xc, Xr, info


def predict(inp, state, clf, reg):
    Xc, Xr, info = build_features(inp, state)
    p_good = float(clf.predict_proba(Xc)[0, 1])
    future = float(reg.predict(Xr)[0])
    price = float(inp["price"])
    return dict(p_good=p_good, is_good=p_good >= 0.5, future_price=future,
                gain_lakhs=future - price, gain_pct=(future / price - 1) * 100,
                cagr_pct=((future / price) ** (1 / 5) - 1) * 100, **info)
