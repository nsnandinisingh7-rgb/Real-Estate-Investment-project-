"""
Step 2 - Exploratory Data Analysis
Project: Real Estate Investment Advisor

Answers all 20 EDA questions from the project brief + the 4 extra analyses mentioned in
the Approach section (price trend by city, area vs return, crime/infrastructure vs
Good_Investment, infrastructure vs resale value).

Run (after 01_preprocessing.py):
    python 02_eda.py --input outputs/cleaned_real_estate.csv --outdir outputs/eda

Outputs:
    outputs/eda/Q01_....png ... one chart per question
    outputs/eda/eda_findings.md   numbers + auto-written findings for your report
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
PPSF = "Price_per_SqFt_Final"
findings = []


def note(title, text):
    findings.append(f"### {title}\n\n{text}\n")


def save(fig, name, outdir):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt(x, d=2):
    return f"{x:,.{d}f}"


def spread_note(g):
    """Describe how big the gap between groups is (so tiny differences are not over-interpreted)."""
    gap = (g.max() - g.min()) / g.mean()
    if gap < 0.03:
        word = "negligible (groups are almost identical)"
    elif gap < 0.10:
        word = "small"
    elif gap < 0.30:
        word = "moderate"
    else:
        word = "large"
    return f"Gap between highest and lowest group = {gap:.1%} of the average, so the effect is **{word}**."


def shape_note(s):
    """Say whether a distribution is bell-shaped, skewed or flat (uniform-like)."""
    if abs(s.skew()) < 0.1 and s.kurt() < -1.0:
        return ("flat / approximately uniform (every value range is about equally common, "
                "unlike real market prices which are usually right-skewed)")
    if s.skew() > 0.5:
        return "right-skewed (a few very high values pull the mean above the median)"
    if abs(s.skew()) < 0.5:
        return "roughly symmetric"
    return "left-skewed"


# ---------------------------------------------------------------------------
# 1-5  Price & Size
# ---------------------------------------------------------------------------
def price_size(df, out):
    # Q1 distribution of price
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["Price_in_Lakhs"], bins=50, kde=True, ax=ax[0], color="#2a6f97")
    ax[0].set_title("Q1. Distribution of Property Price (Lakhs)")
    sns.histplot(np.log1p(df["Price_in_Lakhs"]), bins=50, kde=True, ax=ax[1], color="#61a5c2")
    ax[1].set_title("Q1. Log-scaled price")
    save(fig, "Q01_price_distribution.png", out)
    p = df["Price_in_Lakhs"]
    note("Q1. Distribution of property prices",
         f"Mean {fmt(p.mean())} L, median {fmt(p.median())} L, std {fmt(p.std())}, "
         f"min {fmt(p.min())}, max {fmt(p.max())}. Skewness = {fmt(p.skew())}, "
         f"kurtosis = {fmt(p.kurt())}. Shape: {shape_note(p)}.")

    # Q2 distribution of size
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Size_in_SqFt"], bins=50, kde=True, ax=ax, color="#e07a5f")
    ax.set_title("Q2. Distribution of Property Size (SqFt)")
    save(fig, "Q02_size_distribution.png", out)
    s = df["Size_in_SqFt"]
    note("Q2. Distribution of property sizes",
         f"Mean {fmt(s.mean(), 0)} sqft, median {fmt(s.median(), 0)}, min {fmt(s.min(), 0)}, "
         f"max {fmt(s.max(), 0)}, skewness {fmt(s.skew())}. Shape: {shape_note(s)}.")

    # Q3 price per sqft by property type
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = df.groupby("Property_Type")[PPSF].median().sort_values().index
    sns.boxplot(data=df, x="Property_Type", y=PPSF, order=order, ax=ax, showfliers=False)
    ax.set_title("Q3. Price per SqFt by Property Type")
    save(fig, "Q03_ppsf_by_property_type.png", out)
    g = df.groupby("Property_Type")[PPSF].agg(["mean", "median"]).sort_values("median")
    note("Q3. Price per sq ft by property type",
         g.round(0).to_markdown() + f"\n\nHighest median: {g.index[-1]}, lowest: {g.index[0]}. "
         + spread_note(g["median"]))

    # Q4 size vs price
    fig, ax = plt.subplots(figsize=(7, 5))
    samp = df.sample(min(len(df), 8000), random_state=1)
    sns.scatterplot(data=samp, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.25, s=12, ax=ax)
    ax.set_title("Q4. Size vs Price")
    save(fig, "Q04_size_vs_price.png", out)
    r = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
    note("Q4. Relationship between size and price",
         f"Pearson correlation = {fmt(r)} "
         f"({'strong' if abs(r) > 0.7 else 'moderate' if abs(r) > 0.4 else 'weak'} "
         f"{'positive' if r > 0 else 'negative'} relationship). "
         f"Spearman = {fmt(df['Size_in_SqFt'].corr(df['Price_in_Lakhs'], method='spearman'))}.")

    # Q5 outliers
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(y=df[PPSF], ax=ax[0], color="#81b29a")
    ax[0].set_title("Q5. Price per SqFt")
    sns.boxplot(y=df["Size_in_SqFt"], ax=ax[1], color="#f2cc8f")
    ax[1].set_title("Q5. Size (SqFt)")
    save(fig, "Q05_outliers.png", out)
    lines = []
    for c in [PPSF, "Size_in_SqFt"]:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        lines.append(f"- {c}: {n:,} outliers ({n / len(df):.2%}) outside "
                     f"[{fmt(max(q1 - 1.5 * iqr, 0), 0)}, {fmt(q3 + 1.5 * iqr, 0)}]")
    note("Q5. Outliers in price per sq ft and size", "\n".join(lines))


# ---------------------------------------------------------------------------
# 6-10  Location
# ---------------------------------------------------------------------------
def location(df, out):
    # Q6 state avg ppsf
    st = df.groupby("State")[PPSF].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(st))))
    sns.barplot(x=st.values, y=st.index, ax=ax, color="#3d405b")
    ax.set_title("Q6. Average Price per SqFt by State")
    save(fig, "Q06_ppsf_by_state.png", out)
    note("Q6. Average price per sq ft by state",
         f"Highest: {st.index[0]} ({fmt(st.iloc[0], 0)}), lowest: {st.index[-1]} "
         f"({fmt(st.iloc[-1], 0)}). " + spread_note(st))

    # Q7 city avg price
    ct = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(ct))))
    sns.barplot(x=ct.values, y=ct.index, ax=ax, color="#2a6f97")
    ax.set_title("Q7. Average Property Price by City (Lakhs)")
    save(fig, "Q07_price_by_city.png", out)
    note("Q7. Average property price by city",
         f"Top 5 cities: " + ", ".join(f"{c} ({fmt(v, 1)} L)" for c, v in ct.head(5).items())
         + f". Cheapest: {ct.index[-1]} ({fmt(ct.iloc[-1], 1)} L). " + spread_note(ct))

    # Q8 median age by locality (top / bottom)
    # Locality names (Locality_1..Locality_500) repeat across all cities, so a City+Locality pair
    # has only ~12 listings. Pooling by locality name gives ~500 listings each = stable medians.
    key = df.groupby("Locality")["Age_of_Property"].median()
    cnt = df["Locality"].value_counts()
    top = key.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#bc4749")
    ax.set_title("Q8. Localities with the Highest Median Property Age (top 15)")
    save(fig, "Q08_median_age_by_locality.png", out)
    note("Q8. Median age of properties by locality",
         f"Across {len(key):,} localities (about {int(cnt.median())} listings each) the median age is "
         f"{fmt(key.median(), 1)} years (range {fmt(key.min(), 1)}-{fmt(key.max(), 1)}). "
         f"Oldest: {top.index[0]} ({fmt(top.iloc[0], 1)} yrs). " + spread_note(key)
         + " Locality names repeat across cities in this dataset, so they were pooled.")

    # Q9 BHK across cities
    ct_bhk = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(ct_bhk))))
    sns.heatmap(ct_bhk, annot=len(ct_bhk) <= 20, fmt=".0f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "% of city's properties"})
    ax.set_title("Q9. BHK Distribution Across Cities (%)")
    save(fig, "Q09_bhk_by_city.png", out)
    note("Q9. BHK distribution across cities",
         f"Overall most common configuration: **{int(df['BHK'].mode().iloc[0])} BHK** "
         f"({(df['BHK'] == df['BHK'].mode().iloc[0]).mean():.1%} of listings). "
         "Share by BHK is similar across cities if the heatmap looks uniform.")

    # Q10 top-5 expensive localities trend (Year_Built used as time axis)
    loc = df.groupby("Locality").agg(avg_ppsf=(PPSF, "mean"), n=(PPSF, "size"))
    loc = loc[loc["n"] >= 100].sort_values("avg_ppsf", ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(10, 5))
    lines = []
    for l, row in loc.iterrows():
        sub = df[df["Locality"] == l]
        tr = sub.groupby(pd.cut(sub["Year_Built"], bins=6), observed=True)[PPSF].mean()
        ax.plot([str(int(i.mid)) for i in tr.index], tr.values, marker="o", label=l)
        lines.append(f"- {l}: avg {fmt(row['avg_ppsf'], 0)} per sqft ({int(row['n'])} listings)")
    ax.set_title("Q10. Price per SqFt vs Year Built - Top 5 Most Expensive Localities")
    ax.set_xlabel("Year built (bucket midpoint)")
    ax.legend(fontsize=8)
    save(fig, "Q10_top5_locality_trend.png", out)
    note("Q10. Price trends for the top 5 most expensive localities",
         "\n".join(lines) + "\n\n*The dataset has no listing date, so Year_Built is used as the time "
         "axis. This shows how price per sqft differs by construction year, not a true "
         "historical price series. State this limitation in your report.*")


# ---------------------------------------------------------------------------
# 11-15  Relationships
# ---------------------------------------------------------------------------
def relationships(df, out):
    # Q11 correlation
    num = df.select_dtypes(include="number")
    keep = [c for c in ["BHK", "Size_in_SqFt", "Price_in_Lakhs", PPSF, "Age_of_Property",
                        "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
                        "Transport_Score", "Parking_Count", "Amenity_Count",
                        "Infrastructure_Score", "Appreciation_Pct_5Y"] if c in num.columns]
    corr = num[keep].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Q11. Correlation Matrix of Numeric Features")
    save(fig, "Q11_correlation_heatmap.png", out)
    pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
    note("Q11. Correlation between numeric features",
         "Strongest pairs:\n" + "\n".join(f"- {a} & {b}: {fmt(v)}" for (a, b), v in pairs.head(6).items()))

    # Q12, Q13 schools / hospitals
    for q, col, label in [(12, "Nearby_Schools", "schools"), (13, "Nearby_Hospitals", "hospitals")]:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        g = df.groupby(col)[PPSF].mean()
        ax[0].plot(g.index, g.values, marker="o")
        ax[0].set_title(f"Q{q}. Avg Price per SqFt by Nearby {label}")
        ax[0].set_xlabel(col)
        sns.boxplot(data=df, x=col, y=PPSF, ax=ax[1], showfliers=False)
        ax[1].set_title("Distribution per value")
        save(fig, f"Q{q}_ppsf_vs_{label}.png", out)
        r = df[col].corr(df[PPSF])
        note(f"Q{q}. Nearby {label} vs price per sq ft",
             f"Correlation = {fmt(r, 3)} "
             f"({'practically no linear relationship' if abs(r) < 0.1 else 'a noticeable relationship'}).")

    # Q14 furnished status
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", ax=ax[0], showfliers=False)
    sns.boxplot(data=df, x="Furnished_Status", y=PPSF, ax=ax[1], showfliers=False)
    ax[0].set_title("Q14. Price (Lakhs) by Furnished Status")
    ax[1].set_title("Q14. Price per SqFt by Furnished Status")
    save(fig, "Q14_price_by_furnished.png", out)
    g = df.groupby("Furnished_Status")[["Price_in_Lakhs", PPSF]].mean().round(1)
    note("Q14. Price by furnished status",
         g.to_markdown() + "\n\n" + spread_note(df.groupby("Furnished_Status")[PPSF].mean()))

    # Q15 facing
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = df.groupby("Facing")[PPSF].mean().sort_values().index
    sns.barplot(data=df, x="Facing", y=PPSF, order=order, ax=ax, color="#6d597a", errorbar=None)
    ax.set_title("Q15. Average Price per SqFt by Facing")
    save(fig, "Q15_ppsf_by_facing.png", out)
    g = df.groupby("Facing")[PPSF].mean().sort_values(ascending=False)
    note("Q15. Price per sq ft by facing direction",
         f"Highest: {g.index[0]} ({fmt(g.iloc[0], 0)}), lowest: {g.index[-1]} ({fmt(g.iloc[-1], 0)}). "
         + spread_note(g))


# ---------------------------------------------------------------------------
# 16-20  Investment / amenities / ownership
# ---------------------------------------------------------------------------
def investment(df, out):
    # Q16 owner type
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df["Owner_Type"].value_counts()
    sns.barplot(x=vc.index, y=vc.values, ax=ax, color="#2a6f97")
    ax.set_title("Q16. Properties by Owner Type")
    save(fig, "Q16_owner_type.png", out)
    note("Q16. Properties per owner type", vc.to_frame("count").to_markdown())

    # Q17 availability
    fig, ax = plt.subplots(figsize=(6, 4))
    vc2 = df["Availability_Status"].value_counts()
    sns.barplot(x=vc2.index, y=vc2.values, ax=ax, color="#e07a5f")
    ax.set_title("Q17. Properties by Availability Status")
    save(fig, "Q17_availability.png", out)
    note("Q17. Properties per availability status", vc2.to_frame("count").to_markdown())

    # Q18 parking
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=df, x="Parking_Count", y="Price_in_Lakhs", ax=ax[0], showfliers=False)
    ax[0].set_title("Q18. Price by Parking Spaces")
    sns.barplot(data=df, x="Parking_Count", y=PPSF, ax=ax[1], errorbar=None, color="#81b29a")
    ax[1].set_title("Avg Price per SqFt by Parking Spaces")
    save(fig, "Q18_parking_vs_price.png", out)
    r = df["Parking_Count"].corr(df["Price_in_Lakhs"])
    note("Q18. Effect of parking space on price",
         f"Correlation between parking count and price = {fmt(r, 3)} "
         f"({'little to no effect' if abs(r) < 0.1 else 'some effect'}).")

    # Q19 amenities
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=df, x="Amenity_Count", y=PPSF, ax=ax[0], errorbar=None, color="#3d405b")
    ax[0].set_title("Q19. Avg Price per SqFt by Number of Amenities")
    amn = [c for c in df.columns if c.startswith("Amn_")]
    if amn:
        eff = pd.Series({c[4:].replace("_", " "): df.loc[df[c] == 1, PPSF].mean()
                         - df.loc[df[c] == 0, PPSF].mean() for c in amn}).sort_values()
        ax[1].barh(eff.index, eff.values, color="#f2cc8f")
        ax[1].set_title("Price/sqft difference: has amenity vs not")
    save(fig, "Q19_amenities_vs_ppsf.png", out)
    r = df["Amenity_Count"].corr(df[PPSF])
    note("Q19. Amenities and price per sq ft", f"Correlation between amenity count and price/sqft = {fmt(r, 3)}.")

    # Q20 transport
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=df, x="Public_Transport_Accessibility", y=PPSF, ax=ax[0], errorbar=None, color="#6d597a")
    ax[0].set_title("Q20. Avg Price per SqFt by Transport Access")
    g = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean() * 100
    ax[1].bar(g.index, g.values, color="#81b29a")
    ax[1].set_title("% Good Investment by Transport Access")
    save(fig, "Q20_transport_vs_ppsf_investment.png", out)
    note("Q20. Public transport accessibility vs price/investment",
         g.round(1).to_frame("% Good Investment").to_markdown()
         + f"\n\nCorrelation of transport score with price/sqft = "
         f"{fmt(df['Transport_Score'].corr(df[PPSF]), 3)} (no effect on price). The large gap in "
         "% Good Investment exists because the Good_Investment rule itself rewards infrastructure "
         "(which includes transport), so this pattern is built in by the label definition, "
         "not discovered from the market.")


# ---------------------------------------------------------------------------
# Extra analyses from the "Approach" section
# ---------------------------------------------------------------------------
def extras(df, out):
    # city price trend
    fig, ax = plt.subplots(figsize=(11, 5))
    top_cities = df["City"].value_counts().head(6).index
    for c in top_cities:
        sub = df[df["City"] == c]
        tr = sub.groupby(pd.cut(sub["Year_Built"], bins=8))[PPSF].mean()
        ax.plot([int(i.mid) for i in tr.index], tr.values, marker="o", label=c)
    ax.set_title("X1. Price per SqFt by Year Built - 6 Largest Cities")
    ax.legend(fontsize=8)
    save(fig, "X1_city_price_trend.png", out)

    # area vs investment return
    fig, ax = plt.subplots(figsize=(7, 5))
    samp = df.sample(min(len(df), 8000), random_state=2)
    sns.scatterplot(data=samp, x="Size_in_SqFt", y="Appreciation_Pct_5Y", alpha=.25, s=12, ax=ax)
    ax.set_title("X2. Area vs 5-Year Appreciation (%)")
    save(fig, "X2_area_vs_return.png", out)
    r = df["Size_in_SqFt"].corr(df["Appreciation_Pct_5Y"])
    note("X2. Area vs investment return", f"Correlation = {fmt(r, 3)}. "
         "Appreciation here is the assumption-based 5-year return created in preprocessing.")

    # infrastructure vs Good_Investment and vs price
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = pd.cut(df["Infrastructure_Score"], 5)
    g = df.groupby(bins, observed=True)["Good_Investment"].mean() * 100
    ax[0].bar([str(i) for i in g.index], g.values, color="#2a6f97")
    ax[0].set_title("X3. % Good Investment by Infrastructure Score")
    ax[0].tick_params(axis="x", rotation=30)
    g2 = df.groupby(bins, observed=True)[PPSF].mean()
    ax[1].bar([str(i) for i in g2.index], g2.values, color="#e07a5f")
    ax[1].set_title("X4. Avg Price per SqFt by Infrastructure Score")
    ax[1].tick_params(axis="x", rotation=30)
    save(fig, "X3_X4_infrastructure.png", out)
    note("X3/X4. Infrastructure score vs investment and resale value",
         f"Correlation of Infrastructure_Score with price/sqft = "
         f"{fmt(df['Infrastructure_Score'].corr(df[PPSF]), 3)}.\n\n"
         "**Crime rate:** the brief mentions crime rate, but the dataset has no crime column, "
         "so Infrastructure_Score and Security are used instead. Mention this in your report.")

    # good investment by city
    gi = df.groupby("City")["Good_Investment"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(9, max(4, .28 * len(gi))))
    sns.barplot(x=gi.values, y=gi.index, ax=ax, color="#81b29a")
    ax.set_title("X5. % Good Investment by City")
    save(fig, "X5_good_investment_by_city.png", out)
    note("X5. Good investments by city",
         f"Overall {df['Good_Investment'].mean():.1%} of properties are labelled Good Investment. "
         f"Best city: {gi.index[0]} ({gi.iloc[0]:.1f}%), lowest: {gi.index[-1]} ({gi.iloc[-1]:.1f}%).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/cleaned_real_estate.csv")
    ap.add_argument("--outdir", default="outputs/eda")
    args = ap.parse_args()
    if not os.path.exists(args.input):
        sys.exit(f"{args.input} not found. Run 01_preprocessing.py first.")
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {df.shape[0]:,} rows")

    price_size(df, args.outdir)
    location(df, args.outdir)
    relationships(df, args.outdir)
    investment(df, args.outdir)
    extras(df, args.outdir)

    # overall insight: how strongly does ANY raw feature relate to price?
    raw = ["BHK", "Size_in_SqFt", "Year_Built", "Floor_No", "Total_Floors", "Nearby_Schools",
           "Nearby_Hospitals", "Transport_Score", "Parking_Count", "Has_Security",
           "Amenity_Count", "Furnished_Level"]
    raw = [c for c in raw if c in df.columns]
    cp = df[raw].corrwith(df["Price_in_Lakhs"]).abs().sort_values(ascending=False)
    summary = (f"**Overall insight:** the strongest correlation between any raw feature and "
               f"Price_in_Lakhs is {cp.iloc[0]:.3f} ({cp.index[0]}). "
               + ("Almost no feature explains price, and prices/sizes are evenly spread rather than "
                  "bell-shaped. This is typical of a synthetic (computer-generated) dataset, so most "
                  "EDA relationships are weak. Report this honestly instead of inventing patterns."
                  if cp.iloc[0] < 0.05 else
                  "Some features show a measurable relationship with price."))
    with open(os.path.join(args.outdir, "eda_findings.md"), "w", encoding="utf-8") as fh:
        fh.write("# EDA Findings (auto-generated)\n\n")
        fh.write(f"Dataset: {df.shape[0]:,} properties after cleaning.\n\n{summary}\n\n")
        fh.write("\n".join(findings))
    print(f"Done. Charts + eda_findings.md saved in {args.outdir}")


if __name__ == "__main__":
    main()
