"""
Step 1 - Data Preprocessing
Project: Real Estate Investment Advisor (Predicting Property Profitability & Future Value)

What this script does (in order):
  1. Load india_housing_prices.csv and write a data-quality report
  2. Remove duplicates, clean text, fix data types
  3. Handle missing values (median / mode imputation)
  4. Validate and repair inconsistent values (age, price per sqft, floors)
  5. Detect and cap outliers (IQR rule)
  6. Feature engineering (Infrastructure score, amenity count, etc.)
  7. Create the two target variables:
        Good_Investment  (classification, 0/1)
        Future_Price_5Y_Lakhs (regression)
  8. Encode categorical features and scale numeric features
  9. Save outputs

Run:
    python 01_preprocessing.py --input india_housing_prices.csv --outdir outputs

Outputs (in --outdir):
    cleaned_real_estate.csv      readable cleaned data + engineered features + targets (use for EDA)
    model_ready_real_estate.csv  encoded + scaled data (use for model training)
    preprocessor.joblib          saved encoders/scalers/medians (use inside Streamlit app)
    data_quality_report.txt      what was found and what was fixed
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# CONFIG  - change these if your mentor gives different business rules
# ----------------------------------------------------------------------------
CURRENT_YEAR = 2025          # Age_of_Property = CURRENT_YEAR - Year_Built
FORECAST_YEARS = 5           # forecast horizon from the project brief
BASE_GROWTH = 0.05           # baseline yearly appreciation (5%)
MAX_EXTRA_GROWTH = 0.04      # extra yearly growth earned by good infrastructure / transport
AGE_PENALTY = 0.01           # max yearly growth lost for very old buildings
GOOD_INVEST_MIN_SCORE = 3    # a property must pass at least this many of the 4 rules
OUTLIER_CAP = True           # cap extreme values (IQR rule) instead of deleting rows
RANDOM_SEED = 42

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


def is_numeric(s):
    return pd.api.types.is_numeric_dtype(s)


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    log("=" * 70)
    log("1. RAW DATA OVERVIEW")
    log("=" * 70)
    log(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    log("\nMissing values per column (raw):")
    miss = df.isna().sum()
    log(miss[miss > 0].to_string() if miss.sum() else "  none")
    log(f"\nDuplicate rows (all columns): {df.duplicated().sum():,}")
    if "ID" in df.columns:
        log(f"Duplicate IDs: {df['ID'].duplicated().sum():,}")
    return df


# ----------------------------------------------------------------------------
# 2. CLEAN BASICS
# ----------------------------------------------------------------------------
def clean_basics(df):
    log("\n" + "=" * 70)
    log("2. DUPLICATES & TEXT CLEANING")
    log("=" * 70)
    n0 = len(df)

    # strip whitespace / unify text columns
    for c in df.columns:
        if not is_numeric(df[c]):
            df[c] = df[c].astype("string").str.strip()
            df.loc[df[c].isin(["", "nan", "NaN", "None", "NA", "N/A", "null"]), c] = pd.NA

    # drop exact duplicates, then duplicate IDs
    df = df.drop_duplicates()
    if "ID" in df.columns:
        df = df.drop_duplicates(subset="ID", keep="first")
    log(f"Rows removed as duplicates: {n0 - len(df):,}  ->  {len(df):,} rows remain")

    # numeric columns that may have been read as text
    numeric_cols = [
        "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
        "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
    ]
    for c in numeric_cols:
        if c in df.columns and not is_numeric(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # title-case the main categorical columns so "delhi" and "Delhi" are one category
    for c in ["State", "City", "Locality", "Property_Type", "Furnished_Status", "Facing",
              "Owner_Type", "Availability_Status", "Public_Transport_Accessibility"]:
        if c in df.columns and not is_numeric(df[c]):
            df[c] = df[c].str.title()
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. MISSING VALUES
# ----------------------------------------------------------------------------
def impute(df, state):
    log("\n" + "=" * 70)
    log("3. MISSING VALUE TREATMENT")
    log("=" * 70)
    medians, modes = {}, {}

    # rows with no price or no size cannot be used for anything -> drop
    key = [c for c in ["Price_in_Lakhs", "Size_in_SqFt"] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=key)
    log(f"Dropped {before - len(df):,} rows missing Price_in_Lakhs or Size_in_SqFt")

    for c in df.columns:
        if c == "ID" or df[c].isna().sum() == 0:
            continue
        n = int(df[c].isna().sum())
        if is_numeric(df[c]):
            # median of the same City is more realistic than one global median
            if "City" in df.columns:
                df[c] = df[c].fillna(df.groupby("City")[c].transform("median"))
            med = df[c].median()
            df[c] = df[c].fillna(med)
            medians[c] = float(med)
            log(f"  {c:<32} {n:>6,} missing -> City median / global median")
        else:
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            df[c] = df[c].fillna(fill)
            modes[c] = str(fill)
            log(f"  {c:<32} {n:>6,} missing -> mode '{fill}'")
    state["medians"], state["modes"] = medians, modes
    return df


# ----------------------------------------------------------------------------
# 4. VALIDATE / REPAIR
# ----------------------------------------------------------------------------
def validate(df):
    log("\n" + "=" * 70)
    log("4. CONSISTENCY CHECKS")
    log("=" * 70)

    # invalid values
    bad = (df["Size_in_SqFt"] <= 0) | (df["Price_in_Lakhs"] <= 0)
    log(f"Rows with size <= 0 or price <= 0 removed: {int(bad.sum()):,}")
    df = df[~bad].copy()

    # Age should equal CURRENT_YEAR - Year_Built
    if "Year_Built" in df.columns:
        new_age = (CURRENT_YEAR - df["Year_Built"]).clip(lower=0)
        if "Age_of_Property" in df.columns:
            mismatch = int((df["Age_of_Property"] != new_age).sum())
            log(f"Age_of_Property recomputed from Year_Built "
                f"(rows that differed: {mismatch:,})")
        df["Age_of_Property"] = new_age

    # Price_per_SqFt: Price_in_Lakhs * 100,000 / Size
    calc_pps = df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
    if "Price_per_SqFt" in df.columns:
        # the given column may be in Rs/sqft or in Lakhs/sqft -> detect which one it is
        ratio = (df["Price_per_SqFt"] / calc_pps).median()
        if abs(ratio - 1) < abs(ratio - 1e-5 * 1) and ratio > 0.1:
            unit, ref = "Rs per sqft", calc_pps
        else:
            unit, ref = "Lakhs per sqft", calc_pps / 100000
        diff = (df["Price_per_SqFt"] - ref).abs() / ref
        log(f"Price_per_SqFt given in: {unit} (auto-detected)")
        log(f"  Rows differing >5% from Price/Size: {int((diff > 0.05).sum()):,}")
        log("  Price_per_SqFt_Calc (Rs per sqft) is recomputed and used everywhere below.")
    df["Price_per_SqFt_Calc"] = calc_pps

    # Floor_No cannot exceed Total_Floors
    if {"Floor_No", "Total_Floors"} <= set(df.columns):
        bad_floor = df["Floor_No"] > df["Total_Floors"]
        log(f"Rows where Floor_No > Total_Floors: {int(bad_floor.sum()):,} "
            f"(Total_Floors set to Floor_No)")
        df.loc[bad_floor, "Total_Floors"] = df.loc[bad_floor, "Floor_No"]
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 5. OUTLIERS
# ----------------------------------------------------------------------------
def outliers(df, state):
    log("\n" + "=" * 70)
    log("5. OUTLIER DETECTION (IQR rule: Q1 - 1.5*IQR, Q3 + 1.5*IQR)")
    log("=" * 70)
    cols = ["Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_Calc"]
    bounds = {}
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
        flag = (df[c] < lo) | (df[c] > hi)
        df[f"{c}_Outlier"] = flag.astype(int)
        log(f"  {c:<22} outliers: {int(flag.sum()):>6,} ({flag.mean():.2%})   "
            f"bounds [{lo:,.1f}, {hi:,.1f}]")
        bounds[c] = (float(lo), float(hi))
        if OUTLIER_CAP:
            df[f"{c}_Capped"] = df[c].clip(lo, hi)
    log("  Outliers are FLAGGED (…_Outlier) and CAPPED (…_Capped), not deleted.")
    log("  Use the *_Capped columns for model training, raw columns for EDA.")
    state["outlier_bounds"] = bounds
    return df


# ----------------------------------------------------------------------------
# 6. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
TRANSPORT_MAP = {"low": 1, "medium": 2, "med": 2, "high": 3}


def minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0


def engineer(df, state):
    log("\n" + "=" * 70)
    log("6. FEATURE ENGINEERING")
    log("=" * 70)

    # --- ordinal encodings -------------------------------------------------
    t = df["Public_Transport_Accessibility"]
    df["Transport_Score"] = (
        t if is_numeric(t) else t.str.lower().map(TRANSPORT_MAP)
    ).astype(float)
    df["Transport_Score"] = df["Transport_Score"].fillna(df["Transport_Score"].median())

    f = df["Furnished_Status"].str.lower()
    df["Furnished_Level"] = np.where(f.str.contains("unfurn"), 0,
                             np.where(f.str.contains("semi"), 1, 2))

    # parking: Yes/No or a number
    p = df["Parking_Space"]
    df["Parking_Count"] = p if is_numeric(p) else p.str.lower().map(
        {"yes": 1, "no": 0, "y": 1, "n": 0}).fillna(pd.to_numeric(p, errors="coerce")).fillna(0)

    # security: Yes/No or text (Gated, CCTV...). Anything other than "no/none" counts as present
    s = df["Security"].astype("string").str.lower()
    df["Has_Security"] = (~s.isin(["no", "none", "n", "0", "false"])).astype(int)

    # amenities: comma separated text -> count + top amenity flags
    am = df["Amenities"].astype("string").fillna("")
    df["Amenity_Count"] = am.apply(lambda x: len([a for a in str(x).split(",") if a.strip()]))
    all_am = pd.Series([a.strip().title() for x in am for a in str(x).split(",") if a.strip()])
    top_am = list(all_am.value_counts().head(8).index)
    for a in top_am:
        col = "Amn_" + a.replace(" ", "_")
        df[col] = am.str.contains(a, case=False, regex=False).astype(int)
    state["top_amenities"] = top_am

    # --- numeric engineered features --------------------------------------
    df["Price_per_SqFt_Final"] = df["Price_per_SqFt_Calc"]
    df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"].replace(0, np.nan)
    df["Floor_Ratio"] = df["Floor_Ratio"].fillna(0)
    df["Rooms_per_1000SqFt"] = df["BHK"] / df["Size_in_SqFt"] * 1000
    df["Avg_Room_Size"] = df["Size_in_SqFt"] / df["BHK"].replace(0, np.nan)
    df["Avg_Room_Size"] = df["Avg_Room_Size"].fillna(df["Size_in_SqFt"])

    # School Density Score: schools per hospital-adjusted neighbourhood, scaled 0-10
    df["School_Density_Score"] = minmax(df["Nearby_Schools"]) * 10

    # Infrastructure Score (0-100): 40% schools, 30% hospitals, 30% transport
    df["Infrastructure_Score"] = 100 * (
        0.40 * minmax(df["Nearby_Schools"])
        + 0.30 * minmax(df["Nearby_Hospitals"])
        + 0.30 * minmax(df["Transport_Score"])
    )

    # City-level price benchmark and relative price position
    city_med = df.groupby("City")["Price_per_SqFt_Final"].transform("median")
    df["City_Median_PPSF"] = city_med
    df["PPSF_vs_City_Median"] = df["Price_per_SqFt_Final"] / city_med
    # lookup tables saved so the Streamlit app can rebuild these features for ONE new property
    state["city_median_ppsf"] = df.groupby("City")["Price_per_SqFt_Final"].median().to_dict()
    state["minmax"] = {
        "schools": (float(df["Nearby_Schools"].min()), float(df["Nearby_Schools"].max())),
        "hospitals": (float(df["Nearby_Hospitals"].min()), float(df["Nearby_Hospitals"].max())),
        "transport": (float(df["Transport_Score"].min()), float(df["Transport_Score"].max())),
    }
    df["Age_Bucket"] = pd.cut(df["Age_of_Property"], [-1, 5, 15, 30, 200],
                              labels=["0-5", "6-15", "16-30", "30+"]).astype(str)

    log("  Created: Transport_Score, Furnished_Level, Parking_Count, Has_Security, "
        "Amenity_Count, Amn_* flags,")
    log("           Floor_Ratio, Rooms_per_1000SqFt, Avg_Room_Size, School_Density_Score,")
    log("           Infrastructure_Score, City_Median_PPSF, PPSF_vs_City_Median, Age_Bucket")
    return df


# ----------------------------------------------------------------------------
# 7. TARGETS
# ----------------------------------------------------------------------------
def make_targets(df, state):
    log("\n" + "=" * 70)
    log("7. TARGET VARIABLES (derived using stated assumptions)")
    log("=" * 70)
    log("The dataset has NO appreciation rate and NO future price, so both targets are")
    log("created from domain rules. Document these assumptions in your report.\n")

    # ---- Regression target: price after 5 years -------------------------
    # yearly growth = base + infra/transport bonus - age penalty
    infra01 = df["Infrastructure_Score"] / 100
    age_clip = df["Age_of_Property"].clip(upper=60)
    state["minmax"]["age"] = (float(age_clip.min()), float(age_clip.max()))
    age01 = minmax(age_clip)
    df["Annual_Growth_Rate"] = (BASE_GROWTH + MAX_EXTRA_GROWTH * infra01
                                - AGE_PENALTY * age01)
    df["Future_Price_5Y_Lakhs"] = (df["Price_in_Lakhs"]
                                   * (1 + df["Annual_Growth_Rate"]) ** FORECAST_YEARS)
    df["Appreciation_Pct_5Y"] = (df["Future_Price_5Y_Lakhs"] / df["Price_in_Lakhs"] - 1) * 100
    log(f"Future_Price_5Y_Lakhs = Price * (1 + g)^{FORECAST_YEARS},  "
        f"g = {BASE_GROWTH:.0%} + up to {MAX_EXTRA_GROWTH:.0%} (infrastructure) "
        f"- up to {AGE_PENALTY:.0%} (age)")

    # ---- Classification target: Good_Investment -------------------------
    # a property passes a rule -> +1 point. Needs >= GOOD_INVEST_MIN_SCORE of 4 points.
    pts = pd.DataFrame(index=df.index)
    pts["fair_price"] = (df["PPSF_vs_City_Median"] <= 1.0).astype(int)         # not overpriced vs city
    pts["good_infra"] = (df["Infrastructure_Score"] >= df["Infrastructure_Score"].median()).astype(int)
    pts["high_growth"] = (df["Annual_Growth_Rate"] >= df["Annual_Growth_Rate"].median()).astype(int)
    pts["not_too_old"] = (df["Age_of_Property"] <= 20).astype(int)
    df["Investment_Score"] = pts.sum(axis=1)
    df["Good_Investment"] = (df["Investment_Score"] >= GOOD_INVEST_MIN_SCORE).astype(int)
    rate = df["Good_Investment"].mean()
    log(f"Good_Investment = 1 if >= {GOOD_INVEST_MIN_SCORE} of 4 rules pass: price <= city "
        f"median per sqft, infra >= median, growth >= median, age <= 20 yrs")
    log(f"Class balance: Good = {rate:.1%}, Not good = {1 - rate:.1%}")
    log("\n!! LEAKAGE WARNING !!")
    log("Good_Investment is built from PPSF_vs_City_Median, Infrastructure_Score,")
    log("Annual_Growth_Rate and Age_of_Property. If you feed those into the classifier,")
    log("accuracy will be ~100% and meaningless. They are excluded from CLASSIFICATION")
    log("features in the model-ready file. Likewise Annual_Growth_Rate/Appreciation_Pct_5Y")
    log("are excluded from REGRESSION features.")
    state["targets_note"] = "see report"
    return df


# ----------------------------------------------------------------------------
# 8. ENCODE + SCALE
# ----------------------------------------------------------------------------
def encode_and_scale(df, state):
    log("\n" + "=" * 70)
    log("8. ENCODING & SCALING")
    log("=" * 70)

    out = pd.DataFrame(index=df.index)
    if "ID" in df.columns:
        out["ID"] = df["ID"]

    # frequency encoding for high-cardinality columns (Locality has hundreds of values)
    freq_maps = {}
    for c in ["Locality", "City"]:
        fm = df[c].value_counts(normalize=True)
        freq_maps[c] = fm.to_dict()
        out[f"{c}_Freq"] = df[c].map(fm)

    # label-style for State via category codes is risky -> one-hot instead
    onehot_cols = [c for c in ["State", "Property_Type", "Facing", "Owner_Type",
                               "Availability_Status", "City"] if c in df.columns]
    oh = pd.get_dummies(df[onehot_cols], drop_first=True, dtype=int)
    state["onehot_columns"] = list(oh.columns)

    numeric_cols = [
        "BHK", "Size_in_SqFt_Capped", "Price_in_Lakhs_Capped", "Price_per_SqFt_Calc_Capped",
        "Age_of_Property", "Floor_No", "Total_Floors", "Floor_Ratio", "Nearby_Schools",
        "Nearby_Hospitals", "Transport_Score", "Furnished_Level", "Parking_Count",
        "Has_Security", "Amenity_Count", "Rooms_per_1000SqFt", "Avg_Room_Size",
        "School_Density_Score", "Infrastructure_Score", "PPSF_vs_City_Median",
        "City_Median_PPSF", "Annual_Growth_Rate",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    amn_cols = [c for c in df.columns if c.startswith("Amn_")]

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]),
                          columns=numeric_cols, index=df.index)
    log(f"  Scaled {len(numeric_cols)} numeric columns with StandardScaler (mean 0, std 1)")
    log(f"  One-hot encoded: {onehot_cols} -> {oh.shape[1]} columns")
    log("  Frequency encoded: Locality, City")
    log(f"  Amenity flags kept as 0/1: {len(amn_cols)} columns")

    out = pd.concat([out, scaled, oh, df[amn_cols],
                     df[["Good_Investment", "Future_Price_5Y_Lakhs"]]], axis=1)

    # columns that must NOT be used as inputs for each task (leakage control)
    leak_clf = ["PPSF_vs_City_Median", "Infrastructure_Score", "Annual_Growth_Rate",
                "Age_of_Property", "City_Median_PPSF", "Price_per_SqFt_Calc_Capped"]
    leak_reg = ["Annual_Growth_Rate"]
    state.update(dict(
        scaler=scaler, numeric_cols=numeric_cols, freq_maps=freq_maps,
        amenity_cols=amn_cols,
        feature_cols_classification=[c for c in out.columns
                                     if c not in ["ID", "Good_Investment", "Future_Price_5Y_Lakhs"]
                                     and c not in leak_clf],
        feature_cols_regression=[c for c in out.columns
                                 if c not in ["ID", "Good_Investment", "Future_Price_5Y_Lakhs"]
                                 and c not in leak_reg],
        leak_clf=leak_clf, leak_reg=leak_reg,
    ))
    log(f"  Model-ready shape: {out.shape}")
    return out


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="india_housing_prices.csv")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input file not found: {args.input}\n"
                 "Download india_housing_prices.csv from the project link and pass its path "
                 "with --input.")
    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    state = {}

    df = load(args.input)
    required = ["City", "Price_in_Lakhs", "Size_in_SqFt", "BHK"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        sys.exit(f"Missing required columns: {missing_cols}. Found: {list(df.columns)}")

    df = clean_basics(df)
    df = impute(df, state)
    df = validate(df)
    df = outliers(df, state)
    df = engineer(df, state)
    df = make_targets(df, state)
    model_df = encode_and_scale(df, state)

    # dropdown choices for the Streamlit app
    state["choices"] = {
        "city_state": df.groupby("City")["State"].first().to_dict(),
        "localities": sorted(df["Locality"].unique(), key=lambda x: int(x.split("_")[-1])
                             if x.split("_")[-1].isdigit() else 0),
        "property_types": sorted(df["Property_Type"].unique()),
        "furnished": sorted(df["Furnished_Status"].unique()),
        "facing": sorted(df["Facing"].unique()),
        "owner_types": sorted(df["Owner_Type"].unique()),
        "availability": sorted(df["Availability_Status"].unique()),
        "transport": ["Low", "Medium", "High"],
    }

    cleaned_path = os.path.join(args.outdir, "cleaned_real_estate.csv")
    model_path = os.path.join(args.outdir, "model_ready_real_estate.csv")
    # round decimals so the CSV files stay small (does not change any conclusions)
    df.round(4).to_csv(cleaned_path, index=False)
    model_df.round(5).to_csv(model_path, index=False)
    joblib.dump(state, os.path.join(args.outdir, "preprocessor.joblib"))

    log("\n" + "=" * 70)
    log("DONE")
    log("=" * 70)
    log(f"Saved: {cleaned_path}  {df.shape}")
    log(f"Saved: {model_path}  {model_df.shape}")
    log(f"Saved: {os.path.join(args.outdir, 'preprocessor.joblib')}")

    with open(os.path.join(args.outdir, "data_quality_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))


if __name__ == "__main__":
    main()
