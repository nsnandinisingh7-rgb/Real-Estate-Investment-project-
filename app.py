"""
Real Estate Investment Advisor - Streamlit app

Run:  streamlit run app.py
Pages: Predict | Explore Listings | Market Insights | Model Performance
"""
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

from app_utils import HERE, load_artifacts, predict

OUT = os.path.join(HERE, "outputs")
DATA = os.path.join(OUT, "app_data")
MODELS = os.path.join(OUT, "models")

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading models...")
def get_artifacts():
    return load_artifacts(OUT)


@st.cache_data
def read_csv(name, **kw):
    return pd.read_csv(os.path.join(DATA, name), **kw)


@st.cache_data
def read_models_csv(name, **kw):
    return pd.read_csv(os.path.join(MODELS, name), **kw)


@st.cache_data
def read_meta():
    with open(os.path.join(MODELS, "model_metadata.json")) as fh:
        return json.load(fh)


state, clf, reg = get_artifacts()
choices = state["choices"]
meta = read_meta()


def clean_name(c):
    """Turn a model column name into something readable for charts."""
    return (c.replace("_Capped", "").replace("_Calc", "").replace("_", " ").replace("Amn ", "Amenity: "))


def importance_chart(fname, title, top=10, color="#2a6f97"):
    imp = read_models_csv(fname, index_col=0)["importance"].head(top).reset_index()
    imp.columns = ["feature", "importance"]
    imp["feature"] = imp["feature"].map(clean_name)
    return (alt.Chart(imp, title=title).mark_bar(color=color)
            .encode(x=alt.X("importance:Q", title="Importance"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")])
            .properties(height=26 * top + 30))


def heatmap(df_long, x, y, val, title, fmt=".0f", scheme="blues"):
    return (alt.Chart(df_long, title=title).mark_rect()
            .encode(x=alt.X(f"{x}:O", title=x.replace("_", " ")),
                    y=alt.Y(f"{y}:O", title=y.replace("_", " "), sort=None),
                    color=alt.Color(f"{val}:Q", scale=alt.Scale(scheme=scheme), title=None),
                    tooltip=[x, y, alt.Tooltip(f"{val}:Q", format=fmt)]))


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🏠 Investment Advisor")
page = st.sidebar.radio("Go to", ["🔮 Predict", "🔎 Explore Listings",
                                  "📊 Market Insights", "🧠 Model Performance"])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Educational project. 'Good Investment' labels and 5-year prices come from stated "
    "assumptions applied to the dataset (it has no real future prices). Not financial advice.")


# ============================================================================ PAGE 1
if page == "🔮 Predict":
    st.title("Is this property a good investment?")
    st.write("Enter the property details to get an investment verdict and an estimated price "
             "after 5 years.")

    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Location")
            city = st.selectbox("City", sorted(choices["city_state"]), index=0)
            st.caption(f"State: **{choices['city_state'][city]}**")
            locality = st.selectbox("Locality", choices["localities"])
            property_type = st.selectbox("Property type", choices["property_types"])
            availability = st.selectbox("Availability", choices["availability"])
            owner_type = st.selectbox("Owner type", choices["owner_types"])
        with c2:
            st.subheader("Property")
            bhk = st.slider("BHK", 1, 5, 3)
            size = st.number_input("Size (sq ft)", 500, 5000, 1800, step=50)
            price = st.number_input("Asking price (Lakhs)", 10.0, 500.0, 150.0, step=5.0)
            year_built = st.slider("Year built", 1990, 2025, 2012)
            furnished = st.selectbox("Furnishing", choices["furnished"])
            facing = st.selectbox("Facing", choices["facing"])
        with c3:
            st.subheader("Surroundings & features")
            floor_no = st.slider("Floor number", 0, 30, 5)
            total_floors = st.slider("Total floors in building", 1, 30, 12)
            schools = st.slider("Nearby schools (1-10)", 1, 10, 6)
            hospitals = st.slider("Nearby hospitals (1-10)", 1, 10, 5)
            transport = st.select_slider("Public transport access", choices["transport"], "Medium")
            parking = st.radio("Parking space", ["Yes", "No"], horizontal=True)
            security = st.radio("Security", ["Yes", "No"], horizontal=True)
        amenities = st.multiselect("Amenities", state["top_amenities"],
                                   default=state["top_amenities"][:2])
        submitted = st.form_submit_button("Analyse investment", type="primary", width="stretch")

    if submitted:
        inp = dict(state=choices["city_state"][city], city=city, locality=locality,
                   property_type=property_type, bhk=bhk, size=size, price=price,
                   year_built=year_built, furnished=furnished, floor_no=floor_no,
                   total_floors=total_floors, schools=schools, hospitals=hospitals,
                   transport=transport, parking=parking, security=security,
                   amenities=amenities, facing=facing, owner_type=owner_type,
                   availability=availability)
        r = predict(inp, state, clf, reg)

        st.markdown("---")
        left, right = st.columns([1.1, 1])
        with left:
            if r["is_good"]:
                st.success("### ✅ Good Investment")
            else:
                st.warning("### ⚠️ Not a Good Investment")
            conf = max(r["p_good"], 1 - r["p_good"])
            st.metric("Model confidence in this verdict", f"{conf:.0%}")
            st.progress(r["p_good"], text=f"Probability of being a good investment: {r['p_good']:.0%}")
            if conf < 0.65:
                st.info("Confidence is low: this property is close to the boundary between "
                        "good and not good.")
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Estimated price after 5 years", f"₹{r['future_price']:,.1f} L",
                      f"{r['gain_pct']:+.1f}% vs today")
            m2.metric("Expected gain", f"₹{r['gain_lakhs']:,.1f} L",
                      f"{r['cagr_pct']:.1f}% per year")
            m3, m4 = st.columns(2)
            m3.metric("Price per sq ft", f"₹{r['ppsf']:,.0f}",
                      f"{(r['ppsf_vs_city'] - 1) * 100:+.0f}% vs {city} median", delta_color="inverse")
            m4.metric("Infrastructure score", f"{r['infrastructure_score']:.0f} / 100")

        st.markdown("#### What drives these predictions?")
        a, b = st.columns(2)
        a.altair_chart(importance_chart("feature_importance_classifier.csv",
                                        "Good-investment model: top features"), width="stretch")
        b.altair_chart(importance_chart("feature_importance_regressor.csv",
                                        "5-year price model: top features", color="#e07a5f"),
                       width="stretch")
        with st.expander("How is this calculated? (assumptions)"):
            st.markdown(
                "- **Good Investment** = the property passes at least 3 of 4 rules: price per sq ft at or "
                "below the city median, infrastructure score at or above the dataset median, expected "
                "growth at or above the median, and age 20 years or less. The classifier learns this "
                "pattern from the data.\n"
                "- **Price after 5 years** = today's price grown at 5% a year, plus up to 4% for strong "
                "infrastructure, minus up to 1% for old buildings. The regressor learns this relationship.\n"
                "- The dataset has no real future prices, so these are **modelled estimates under stated "
                "assumptions**, not market forecasts.")

# ============================================================================ PAGE 2
elif page == "🔎 Explore Listings":
    st.title("Explore listings")
    st.caption("A 25,000-property sample from the dataset, scored by the models.")
    df = read_csv("listings_sample.csv")

    f1, f2, f3 = st.columns(3)
    states = f1.multiselect("State", sorted(df["State"].unique()))
    pool = df[df["State"].isin(states)] if states else df
    cities = f2.multiselect("City", sorted(pool["City"].unique()))
    ptypes = f3.multiselect("Property type", sorted(df["Property_Type"].unique()))
    g1, g2, g3 = st.columns(3)
    price_rng = g1.slider("Price (Lakhs)", 10, 500, (10, 500))
    size_rng = g2.slider("Area (sq ft)", 500, 5000, (500, 5000), step=50)
    bhks = g3.multiselect("BHK", [1, 2, 3, 4, 5])
    h1, h2 = st.columns(2)
    avail = h1.multiselect("Availability", sorted(df["Availability_Status"].unique()))
    min_p = h2.slider("Minimum probability of Good Investment", 0.0, 1.0, 0.0, 0.05)

    q = df[df["Price_in_Lakhs"].between(*price_rng) & df["Size_in_SqFt"].between(*size_rng)
           & (df["Model_P_Good"] >= min_p)]
    if states:
        q = q[q["State"].isin(states)]
    if cities:
        q = q[q["City"].isin(cities)]
    if ptypes:
        q = q[q["Property_Type"].isin(ptypes)]
    if bhks:
        q = q[q["BHK"].isin(bhks)]
    if avail:
        q = q[q["Availability_Status"].isin(avail)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching properties", f"{len(q):,}")
    if len(q):
        k2.metric("Average price", f"₹{q['Price_in_Lakhs'].mean():,.0f} L")
        k3.metric("Predicted good investments", f"{(q['Model_P_Good'] >= .5).mean():.0%}")
        k4.metric("Avg est. price in 5 yrs", f"₹{q['Model_Price_5Y_Lakhs'].mean():,.0f} L")
        st.dataframe(
            q.head(500), width="stretch", hide_index=True,
            column_config={
                "Model_P_Good": st.column_config.ProgressColumn("P(Good)", min_value=0, max_value=1,
                                                                format="%.2f"),
                "Price_in_Lakhs": st.column_config.NumberColumn("Price (L)", format="%.1f"),
                "Model_Price_5Y_Lakhs": st.column_config.NumberColumn("Est. price 5y (L)", format="%.1f"),
                "Price_per_SqFt": st.column_config.NumberColumn("₹/sq ft", format="%d"),
            })
        st.caption("Showing the 500 highest-scoring matches. Download to get all matches.")
        st.download_button("Download matches (CSV)", q.to_csv(index=False), "matching_listings.csv")
        hist = (alt.Chart(q, title="Price distribution of matching properties")
                .mark_bar(color="#2a6f97").encode(alt.X("Price_in_Lakhs:Q", bin=alt.Bin(maxbins=30),
                                                        title="Price (Lakhs)"),
                                                  alt.Y("count()", title="Properties")))
        st.altair_chart(hist, width="stretch")
    else:
        st.info("No properties match these filters. Try widening them.")

# ============================================================================ PAGE 3
elif page == "📊 Market Insights":
    st.title("Market insights")
    st.caption("Computed on all 250,000 properties.")
    city_df = read_csv("city_stats.csv")
    t1, t2, t3, t4, t5 = st.tabs(["City ranking", "Location heatmaps", "Price trend",
                                  "Infrastructure", "Reading these charts"])

    with t1:
        metric = st.radio("Rank cities by", ["avg_price_lakhs", "avg_ppsf", "pct_good"], horizontal=True,
                          format_func=lambda m: {"avg_price_lakhs": "Average price (Lakhs)",
                                                 "avg_ppsf": "Average price per sq ft",
                                                 "pct_good": "% Good Investments"}[m])
        top = city_df.sort_values(metric, ascending=False).head(15)
        st.altair_chart(alt.Chart(top).mark_bar(color="#2a6f97")
                        .encode(x=alt.X(f"{metric}:Q", title=None), y=alt.Y("City:N", sort="-x", title=None),
                                tooltip=["City", "State", alt.Tooltip(f"{metric}:Q", format=",.1f")])
                        .properties(height=420), width="stretch")
    with t2:
        h1 = read_csv("heat_state_type_ppsf.csv").melt("State", var_name="Property_Type", value_name="ppsf")
        st.altair_chart(heatmap(h1, "Property_Type", "State", "ppsf",
                                "Average price per sq ft: state x property type", scheme="oranges"),
                        width="stretch")
        h2 = read_csv("heat_city_bhk_price.csv").melt("City", var_name="BHK", value_name="price")
        st.altair_chart(heatmap(h2, "BHK", "City", "price", "Average price (Lakhs): city x BHK",
                                fmt=".1f").properties(height=700), width="stretch")
    with t3:
        tr = read_csv("trend_year_built_ppsf.csv").melt("Year_Built", var_name="Property_Type",
                                                        value_name="ppsf")
        st.altair_chart(alt.Chart(tr, title="Average price per sq ft by year built")
                        .mark_line(point=True).encode(x=alt.X("Year_Built:O", title="Year built"),
                                                      y=alt.Y("ppsf:Q", title="₹ per sq ft",
                                                              scale=alt.Scale(zero=False)),
                                                      color="Property_Type:N",
                                                      tooltip=["Year_Built", "Property_Type", "ppsf"]),
                        width="stretch")
        st.caption("The dataset has no listing date, so year built is used as the time axis.")
    with t4:
        inf = read_csv("infra_bands.csv")
        c1, c2 = st.columns(2)
        c1.altair_chart(alt.Chart(inf, title="% Good Investment by infrastructure score")
                        .mark_bar(color="#81b29a").encode(x=alt.X("infra_band:N", title="Infrastructure score"),
                                                          y=alt.Y("pct_good:Q", title="% good")), width="stretch")
        c2.altair_chart(alt.Chart(inf, title="Average price per sq ft by infrastructure score")
                        .mark_bar(color="#e07a5f").encode(x=alt.X("infra_band:N", title="Infrastructure score"),
                                                          y=alt.Y("avg_ppsf:Q", title="₹ per sq ft",
                                                                  scale=alt.Scale(zero=False))), width="stretch")
    with t5:
        st.markdown(
            "In this dataset, prices differ very little between cities, states, property types, "
            "furnishing levels or facing directions (gaps of about 1-3%), and features such as "
            "schools, hospitals, parking and amenities show almost no link to price. This is typical of "
            "a synthetic dataset. The strong pattern between infrastructure and 'Good Investment' exists "
            "because the Good Investment rule itself rewards infrastructure.")

# ============================================================================ PAGE 4
else:
    st.title("Model performance")
    cm, rm = meta["classifier_metrics"], meta["regressor_metrics"]
    st.subheader("Classification: Good Investment")
    a, b, c, d = st.columns(4)
    a.metric("Best model", meta["best_classifier"])
    b.metric("Accuracy", f"{cm['accuracy']:.1%}")
    c.metric("F1 score", f"{cm['f1']:.3f}")
    d.metric("ROC-AUC", f"{cm['roc_auc']:.3f}")
    st.dataframe(read_models_csv("classification_results.csv"), width="stretch", hide_index=True)
    st.caption(f"3-fold cross-validated F1 of the best model: {meta['cv_f1_mean']:.3f} "
               f"(+/- {meta['cv_f1_std']:.3f}). Test set: {meta['n_test']:,} properties.")
    p1, p2 = st.columns(2)
    p1.image(os.path.join(MODELS, "plots", "clf_confusion_matrix.png"), width="stretch")
    p2.image(os.path.join(MODELS, "plots", "clf_roc_curves.png"), width="stretch")

    st.subheader("Regression: price after 5 years")
    a, b, c, d = st.columns(4)
    a.metric("Best model", meta["best_regressor"])
    b.metric("RMSE", f"₹{rm['rmse']:.2f} L")
    c.metric("MAE", f"₹{rm['mae']:.2f} L")
    d.metric("R²", f"{rm['r2']:.4f}")
    st.dataframe(read_models_csv("regression_results.csv"), width="stretch", hide_index=True)
    st.image(os.path.join(MODELS, "plots", "reg_actual_vs_pred_residuals.png"), width="stretch")
    st.warning(
        "The 5-year price is calculated from today's price with a growth rule, so the regression "
        "model can learn it almost perfectly (R² close to 1). That shows the model reproduces the "
        "rule, not that real prices can be forecast this accurately.")

    st.subheader("Feature importance")
    a, b = st.columns(2)
    a.altair_chart(importance_chart("feature_importance_classifier.csv", "Classifier", top=15),
                   width="stretch")
    b.altair_chart(importance_chart("feature_importance_regressor.csv", "Regressor", top=15,
                                    color="#e07a5f"), width="stretch")
    st.info("Experiments are tracked with MLflow. To browse every run, run "
            "`mlflow ui --backend-store-uri sqlite:///mlflow.db` in the project folder.")
