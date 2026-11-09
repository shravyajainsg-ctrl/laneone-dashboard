"""
LaneOne - Mumbai multi-brand fashion hub: Data-Analytics Feasibility Dashboard
Run:  streamlit run app.py
(First run the three pipeline scripts, or just run the app - it regenerates data if missing.)
"""
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from config import *
import analytics as A
import charts as C
import eda

st.set_page_config(page_title=f"{BUSINESS_NAME} | Feasibility Analytics", page_icon="🛍️", layout="wide")

# ------------------------------------------------------------------ data bootstrap
if not os.path.exists("data/clean_survey.csv"):
    with st.spinner("First run: generating synthetic survey and cleaning it..."):
        subprocess.run([sys.executable, "01_generate_synthetic_data.py"], check=True)
        subprocess.run([sys.executable, "02_clean_transform.py"], check=True)


@st.cache_data
def load():
    raw = pd.read_csv("data/raw_survey_responses.csv")
    df = A.load_clean()
    truth = pd.read_csv("data/_ground_truth_hidden.csv")
    return raw, df, truth, pd.read_csv("data/cleaning_log.csv"), pd.read_csv("data/noise_injection_log.csv")


@st.cache_resource
def models(_df):
    cls = A.run_classification(_df)
    reg = A.run_regression(_df)
    cl = A.run_clustering(_df)
    return cls, reg, cl


@st.cache_data(show_spinner="Simulating 1,000 futures per scenario...")
def cached_compare(aov, spend, n):
    return A.compare_scenarios(aov, spend, n=n)


@st.cache_data(show_spinner="Running Monte-Carlo...")
def cached_mc(name, aov, spend, n):
    return A.monte_carlo(name, aov, spend, n=n)


@st.cache_data(show_spinner="Analysing comments...")
def cached_text(_df, _truth):
    return A.run_text_analytics(_df, _truth)


raw, df, truth, clean_log, noise_log = load()
cls, reg, cl = models(df)
df = df.copy()
df["p_interest"] = cls["oof_prob"]
df["persona"] = cl["persona"]
df["pred_spend"] = np.expm1(reg["model"].predict(df[A.REG_NUM + A.CAT_FEATURES]))
SI = A.survey_inputs(df)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def show_eda(items, data, open_first=False):
    """Render eda.py analyses: each in an expander with table(s), chart and data-driven explanation."""
    for i, (title, fn) in enumerate(items):
        with st.expander(title, expanded=open_first and i == 0):
            try:
                r = fn(data)
            except Exception as e:  # e.g. a filter leaves too few rows for a test
                st.warning(f"Not enough data for this analysis with the current filters ({e}).")
                continue
            for k in ("table", "table2"):
                if r.get(k) is not None:
                    st.dataframe(r[k], width="stretch")
            if r.get("fig") is not None:
                show(r["fig"])
            st.markdown(r["text"])


def explain(what, why, insight):
    st.markdown(f"**What it shows:** {what}  \n**Why this chart:** {why}  \n**Insight →** {insight}")


def money(x):
    return f"₹{x/1e7:,.2f} Cr" if abs(x) >= 1e7 else (f"₹{x/1e5:,.1f} L" if abs(x) >= 1e5 else f"₹{x:,.0f}")


# ------------------------------------------------------------------ sidebar
PAGES = ["🏠 Overview", "📝 Survey & data quality", "📊 Descriptive", "🔍 Diagnostic (hypotheses)",
         "🤖 Classification (will they buy?)", "💰 Regression (how much?)", "👥 Clustering (personas)",
         "🛍️ Association (which brands first?)", "💬 Text analytics", "📈 Break-even & risk",
         "🌱 Sustainability", "🎯 Score a new customer"]
st.sidebar.title(f"🛍️ {BUSINESS_NAME}")
st.sidebar.caption("Multi-brand fashion & lifestyle hub · Mumbai · feasibility analytics")
page = st.sidebar.radio("Go to", PAGES)
st.sidebar.markdown("---")
st.sidebar.markdown("**Analytics ladder**  \n📊 Descriptive – what is happening  \n🔍 Diagnostic – why  \n"
                    "🤖 Predictive – what will happen  \n📈 Prescriptive – what we should do")
st.sidebar.caption("All data is SYNTHETIC (simulated survey, n≈2,000) for academic validation.")

# =================================================================== OVERVIEW
if page == PAGES[0]:
    st.title(f"{BUSINESS_NAME} — a 6th Street–style multi-brand fashion hub for Mumbai")
    st.markdown("Accessible-premium, authentic multi-brand retail: **one app + experience store**, powered by "
                "segmentation, recommendations, demand forecasting and data-driven pricing. "
                "This dashboard tests whether Mumbai shoppers want it, who they are, what they'd spend, "
                "which brands to launch with, and whether the numbers work.")
    k = A.kpis(df)
    cols = st.columns(len(k))
    for c, (lab, v) in zip(cols, k.items()):
        c.metric(lab, v)
    st.markdown("---")
    comp = cached_compare(SI["aov"], SI["stated_monthly_spend"], 400) if st.checkbox(
        "Compute risk headline (runs a quick Monte-Carlo)", value=True) else None
    seg = A.conversion_segment(df.p_interest)
    rules, single, _ = A.association_rules(df)
    opt = A.launch_brand_optimizer(df)
    n80 = int(opt[opt["Cumulative coverage of interested customers"] >= 0.8].Step.min())
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Key findings")
        st.markdown(f"""
- **Demand exists but is concentrated.** {df.interested.mean():.0%} say they'd probably/definitely shop with us;
  interest climbs from {df.groupby('Q5_household_income').interested.mean().reindex(INCOME_BANDS).iloc[0]:.0%} in < ₹50K households
  to {df.groupby('Q5_household_income').interested.mean().reindex(INCOME_BANDS).iloc[-1]:.0%} in ₹8L+ households.
- **Who converts:** the {cls['best']} model (CV AUC {cls['table']['ROC-AUC (5-fold CV)'].iloc[0]:.2f}) flags
  {(seg == 'High').mean():.0%} of respondents as *High* probability — {df[seg == 'High'].interested.mean():.0%} of them really are interested.
- **Where:** South Mumbai & Western Suburbs convert best → first store in the western suburbs.
- **Baby steps works:** just **{n80} brands** reach 80% of interested customers (vs 15 in the full model).
- **What they'd spend:** median ₹{SI['stated_monthly_spend']:,.0f}/month stated by the interested (we plan on only 35% of it).
""")
    with c2:
        st.subheader("Strategy verdict")
        if comp is not None:
            b, c = comp.iloc[1], comp.iloc[2]
            st.markdown(f"""
- A **full flagship from day 1** has the best expected payoff but a **{b['P(loss > ₹5 Cr)']:.0%}** chance of losing more than ₹5 Cr.
- The **staged plan** (lean pilot → flagship only if month-12 targets are hit) cuts that to **{c['P(loss > ₹5 Cr)']:.0%}**.
- Recommendation: **launch lean with {n80} brands in the western suburbs + app, prove realised spend & margin, then scale.**
""")
        st.info("Navigate with the sidebar: every page explains each chart (what / why / insight).")

# =================================================================== SURVEY & DATA
elif page == PAGES[1]:
    st.title("Survey design, synthetic data & cleaning")
    t1, t2, t3, t4 = st.tabs(["30-question survey", "Noise we injected", "Cleaning log", "Before vs after"])
    with t1:
        st.markdown("Each question is mapped to the algorithm that uses it. **Q30 = classification target**, "
                    "**Q28 = regression target**, **Q26 = association basket**, **Q29 = text**.")
        st.dataframe(pd.DataFrame(SURVEY, columns=["Code", "Question", "Type", "Options", "Used by"]), width="stretch", height=600)
    with t2:
        st.markdown("Synthetic respondents are generated from hidden traits (income, fashion affinity, price sensitivity, "
                    "digital savviness, style persona), so answers are realistically correlated. Then we **deliberately** "
                    "corrupt the data the way real Google-Form exports are corrupted:")
        st.dataframe(noise_log, width="stretch", height=450)
        st.markdown("**Raw sample (what the analyst receives):**")
        st.dataframe(raw.head(25), width="stretch")
        st.download_button("Download raw CSV", raw.to_csv(index=False), "raw_survey_responses.csv")
    with t3:
        st.markdown(f"**{len(raw):,} raw rows → {len(df):,} clean respondents.** Every step, with its rationale:")
        st.dataframe(clean_log, width="stretch", height=600)
        st.download_button("Download clean CSV", df.to_csv(index=False), "clean_survey.csv")
    with t4:
        show(C.missing_before_after(raw, df, ["Q1_age", "Q5_household_income", "Q9_monthly_fashion_spend", "Q13_avg_order_value",
                                              "Q14_social_hrs_week", "Q28_expected_monthly_spend", "Q29_feedback_text"]))
        explain("Share of missing answers per question, raw vs cleaned.",
                "Shows imputation worked and makes visible where we chose NOT to impute.",
                "Most gaps are filled with group medians/modes. Expected spend (Q28) and comments (Q29) stay partly missing "
                "on purpose: we never invent a model target, and junk text ('asdf', 'na') was converted to missing.")
        rs = raw.Q9_monthly_fashion_spend.astype(str).str.replace(r"[₹,]|INR|Rs\.", "", regex=True)
        show(C.outlier_box(rs, df.Q9_monthly_fashion_spend, "Monthly fashion spend - outliers before vs after"))
        explain("Distribution of monthly fashion spend (log scale) before and after outlier treatment.",
                "Box plots expose extreme values instantly; a log axis is needed because spend is right-skewed.",
                "'Extra-zero' typos (₹5,00,000 instead of ₹5,000) were detected with a domain rule (spend > 50% of household "
                "income) and divided by 100. Genuine luxury shoppers (₹60K–1.5L with ₹8L+ income) were deliberately kept.")

# =================================================================== DESCRIPTIVE
elif page == PAGES[2]:
    st.title("Descriptive analytics — who answered and what they do today")
    with st.expander("Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        zones = f1.multiselect("Zone", AREAS, default=AREAS)
        genders = f2.multiselect("Gender", GENDERS, default=GENDERS)
        ages = f3.slider("Age", 16, 80, (16, 80))
    d = df[df.Q3_area.isin(zones) & df.Q2_gender.isin(genders) & df.Q1_age.between(*ages)]
    st.caption(f"{len(d):,} respondents in view")
    if len(d) < 30:
        st.warning("Too few respondents for reliable charts - widen the filters."); st.stop()
    c1, c2 = st.columns(2)
    with c1:
        show(C.intent_distribution(d))
        explain("Answers to the 5-point purchase-intent question (our classification target).",
                "An ordered bar chart preserves the natural order from 'Definitely not' to 'Definitely will'.",
                f"{d.interested.mean():.0%} are positive (Probably/Definitely will). The polarised shape (big ends, small middle) "
                "means most people already have an opinion - marketing should convert the 'Probably' group, not convince sceptics.")
    with c2:
        show(C.rate_by(d, "Q5_household_income", INCOME_BANDS, "Interest rate by monthly household income (line = average)"))
        explain("Share interested within each income band.", "Income is ordinal, so bars stay in income order.",
                "Interest rises steeply with income - the accessible-premium positioning fits ₹1L+ households; "
                "below ₹50K the proposition doesn't land.")
    c1, c2 = st.columns(2)
    with c1:
        show(C.rate_by(d, "Q3_area", None, "Interest rate by Mumbai zone (line = average)"))
        explain("Interest rate per zone, ranked.", "Ranked bars make the location decision obvious.",
                "South Mumbai and the Western Suburbs lead. The Western Suburbs combine high interest with the largest "
                "population - the natural first-store catchment (Andheri–Bandra–Juhu belt).")
    with c2:
        show(C.age_hist(d))
        explain("Age distribution of interested vs not-interested respondents.", "Two outline histograms on one axis compare shapes without hiding each other.",
                "Interested respondents skew younger (peak late 20s–early 30s); interest drops after ~40. "
                "Brand voice, channels and assortment should be millennial/Gen-Z first.")
    show(C.likert_dumbbell(d))
    explain("Average agreement with each attitude statement, interested vs not.",
            "A dumbbell shows both levels AND the gap; the gap is what discriminates customers.",
            "Biggest gaps: multi-brand appeal, brand preference, phygital and AI-recommendation interest - exactly the "
            "pillars of the concept. Sustainability barely differs, and discount-seeking is HIGHER among non-buyers.")
    c1, c2 = st.columns(2)
    with c1:
        show(C.share_bars(d, "plat_", PLATFORMS, "Q11 - Where respondents shop today"))
        explain("Share of respondents who bought from each platform in 6 months.", "Ranked bars for a multi-select question.",
                "Myntra, Amazon and Ajio dominate - these are the real competitors. We must beat them on authenticity, "
                "curation and experience, not on catalogue size or price.")
    with c2:
        show(C.share_bars(d, "cat_", CATEGORIES, "Q12 - Categories bought most often"))
        explain("Share buying each category.", "Ranked bars.", "Apparel + footwear + athleisure form the core; "
                "bags/accessories and beauty are strong add-ons (good basket builders).")
    show(C.driver_split(d))
    explain("What would most make each group try us.", "Grouped bars compare the two groups on the same scale.",
            "Interested shoppers are pulled by **authentic international brands** and **fast delivery**; non-interested ones "
            "want **discounts**. Lead marketing with authenticity + 90-minute delivery, not price cuts.")

    st.markdown("---")
    st.header("Extended descriptive analysis (D1–D7)")
    st.caption("Summary statistics, distributions, categorical profile, Likert attitudes, multi-select answers and the target. "
               "Same analyses as notebook 04_descriptive_diagnostic_EDA.ipynb; they respect the filters above.")
    show_eda(eda.DESCRIPTIVE, d)

# =================================================================== DIAGNOSTIC
elif page == PAGES[3]:
    st.title("Diagnostic analytics — why do some people want this and others don't?")
    corr_cols = ["Q1_age", "income_ord", "log_monthly_spend", "Q8_shop_freq_month", "Q14_social_hrs_week", *LIKERT_COLS,
                 "n_brand", "intent_score", "Q28_expected_monthly_spend"]
    labels = [c.split("_", 1)[1].replace("_", " ") if c.startswith("Q") else c.replace("_", " ") for c in corr_cols]
    fig, corr = C.corr_heatmap(df, corr_cols, labels)
    show(fig)
    top = corr["intent_score"].drop(["intent_score", "Q28_expected_monthly_spend"]).sort_values(ascending=False)
    explain("Spearman rank correlations between every numeric/ordinal variable.",
            "Spearman (not Pearson) because Likert items are ordinal and spend is skewed. Blue = positive, red = negative.",
            f"Purchase intent correlates most with **{top.index[0]}** (ρ={top.iloc[0]:.2f}), **{top.index[1]}** "
            f"(ρ={top.iloc[1]:.2f}) and **{top.index[2]}** (ρ={top.iloc[2]:.2f}); discount-seeking is negative "
            f"(ρ={corr.loc['Q18_discount_seeker', 'intent_score']:.2f}). Income and current spend are strongly linked (ρ≈0.7) - "
            "we keep both but check multicollinearity in the linear models.")
    c1, c2 = st.columns(2)
    with c1:
        show(C.spend_scatter(df))
        explain("Current monthly fashion spend vs what they'd spend with us (log–log).",
                "Scatter reveals the relationship and the two populations at once.",
                "A tight upward band: today's heavy spenders are tomorrow's best customers (H8). "
                "The flat grey floor are uninterested people who'd spend almost nothing. A few high-spend dots with low "
                "expected spend are luxury shoppers who won't move their spend to an accessible-premium store.")
    with c2:
        show(C.rate_by(df, "Q10_channel", None, "Interest rate by preferred channel"))
        explain("Interest by current shopping channel.", "Ranked bars.",
                "Online-first shoppers are twice as interested as offline-first. The app is the acquisition engine; "
                "the store is the trust & experience layer.")
    st.subheader("Hypothesis testing (α = 0.05)")
    st.markdown("Each business hypothesis is turned into a statistical test. Non-parametric tests are used for "
                "ordinal/skewed data, chi-square for category-vs-category.")
    st.dataframe(A.hypothesis_tests(df), width="stretch", height=480)
    st.markdown("---")
    st.header("Extended diagnostic analysis (G1–G8)")
    st.caption("Pearson vs Spearman, drivers of interest, Cramér's V, zone × income drill-down, spend relationships, "
               "dose-response, effect sizes, outliers and multicollinearity.")
    show_eda(eda.DIAGNOSTIC, df)

# =================================================================== CLASSIFICATION
elif page == PAGES[4]:
    st.title("Predictive — classification: will this person become a customer?")
    st.markdown("**Target:** `interested` = 1 if Q30 is *Probably* or *Definitely will*. **Features:** Q1–Q25 + Q27 "
                "(Q28 and Q30 excluded to avoid leakage). 75/25 stratified split + 5-fold cross-validation.")
    st.dataframe(cls["table"].style.format({c: "{:.3f}" for c in cls["table"].columns if c != "Model"}), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        show(C.roc_plot(cls["rocs"], cls["table"]))
        explain("ROC curve per model: recall vs false-alarm rate at every threshold.",
                "AUC is threshold-free, so it compares models fairly on balanced and imbalanced data alike.",
                f"All models land at roughly 0.83–0.90. **{cls['best']}** wins on cross-validated AUC - and it's also the most "
                "explainable, which matters for marketing teams.")
    with c2:
        show(C.confusion(cls["cm"], f"Confusion matrix - {cls['best']}"))
        cm = cls["cm"]
        explain("Correct vs wrong predictions on the 25% test set.", "Shows the TYPE of errors, not just accuracy.",
                f"Precision {cm[1,1]/(cm[1,1]+cm[0,1]):.0%}: when we target someone, they're usually interested - "
                f"low wasted ad spend. {cm[1,0]} interested people were missed (false negatives) - acceptable at launch.")
    c1, c2 = st.columns(2)
    with c1:
        show(C.importance(cls["importance"], "What predicts interest? (Random Forest importance)"))
        explain("Which survey answers drive the prediction.", "Tree importance captures non-linear effects.",
                "Current fashion spend, multi-brand appeal, phygital interest, brand preference and AI-reco interest dominate. "
                "These five become the short qualifying questions on the sign-up / lead form.")
    with c2:
        odds = cls["odds"]
        st.markdown("**Logistic-regression odds ratios** (per 1 SD / vs base category)")
        st.dataframe(pd.DataFrame({"Odds ratio": odds}).head(10).style.format("{:.2f}"), width="stretch")
        st.dataframe(pd.DataFrame({"Odds ratio": odds}).tail(6).style.format("{:.2f}"), width="stretch")
        st.caption("Odds ratio > 1 raises the chance of interest; < 1 lowers it.")
    st.subheader("High / Medium / Low conversion segments")
    hi = st.slider("High-probability cut-off", 0.5, 0.9, 0.65, 0.05)
    lo = st.slider("Low-probability cut-off", 0.1, 0.5, 0.35, 0.05)
    seg = np.where(df.p_interest >= hi, "High", np.where(df.p_interest >= lo, "Medium", "Low"))
    show(C.segment_chart(seg, df.interested.values))
    explain("Each respondent's out-of-fold probability, bucketed into three action tiers.",
            "Probabilities are more useful than yes/no: they let marketing size budgets by tier.",
            f"High tier = {(seg=='High').mean():.0%} of people with {df[seg=='High'].interested.mean():.0%} real interest. "
            "Spend paid acquisition on High, nurture Medium with low-cost offers, and don't pay CAC for Low.")
    st.dataframe(df.assign(segment=seg).groupby("segment").agg(people=("resp_id", "count"),
                 avg_probability=("p_interest", "mean"), actual_interest=("interested", "mean"),
                 median_pred_spend=("pred_spend", "median")).round(2), width="stretch")

# =================================================================== REGRESSION
elif page == PAGES[5]:
    st.title("Predictive — regression: how much would they spend?")
    st.markdown("**Target:** Q28 expected monthly spend at us (log-transformed to tame skew). Rows with a missing / "
                "invalid target are excluded (never imputed).")
    st.dataframe(reg["table"].style.format({c: "{:.3f}" for c in reg["table"].columns if c.startswith("R")} |
                                           {"MAE (₹)": "₹{:,.0f}", "RMSE (₹)": "₹{:,.0f}"}), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        a, p = reg["preds"][reg["best"]]
        show(C.actual_vs_pred(a, p, reg["best"]))
        explain("Each test respondent: actual vs predicted spend (log axes).",
                "Points on the diagonal = perfect prediction; spread shows error.",
                f"{reg['best']} explains ~{reg['table']['R² (5-fold CV)'].iloc[0]:.0%} of variance (log scale). "
                "Error is largest at the very low end (people who'd spend ~₹0–200), which barely matters for revenue.")
    with c2:
        show(C.coef_chart(reg["coef_pct"]))
        explain("Ridge coefficients converted to % change in spend.", "Linear model = explainable 'levers'.",
                "Intent and current spend dominate; brand preference and authenticity add a premium; "
                "discount-seekers and outer-MMR residents spend less. Price for brand lovers, not bargain hunters.")
    st.subheader("Predicted spend by segment")
    seg = A.conversion_segment(df.p_interest)
    t = df.assign(segment=seg).groupby(["segment", "persona"]).agg(people=("resp_id", "count"),
            p_interest=("p_interest", "mean"), median_pred_spend=("pred_spend", "median")).reset_index()
    t["expected_value_per_person"] = t.p_interest * t.median_pred_spend
    st.dataframe(t.sort_values("expected_value_per_person", ascending=False).round(2), width="stretch")
    st.caption("Expected value = probability of converting × predicted monthly spend → where each marketing rupee should go.")

# =================================================================== CLUSTERING
elif page == PAGES[6]:
    st.title("Clustering — customer personas (K-Means)")
    st.markdown("Features: age, income, spend, frequency, digital index, brand-affinity index, deal-seeker index, sustainability "
                "(standardised).")
    show(C.elbow_silhouette(cl["ks"], cl["inertia"], cl["silhouette"], cl["k"]))
    explain("Elbow (within-cluster error) and silhouette score for k = 2–8.",
            "Two separate charts (never a dual axis) - both tests are needed to pick k.",
            "The elbow bends around k = 4. k = 2 has a higher silhouette but is too coarse to act on (rich vs not); "
            "k = 4 gives four distinct, marketable personas. Low silhouettes are normal for attitude data - segments overlap.")
    show(C.pca_small_multiples(cl["pca"], cl["persona"]))
    explain("Respondents projected onto 2 principal components, one panel per persona.",
            "Small multiples avoid colour clutter and show each persona's territory.",
            "Personas occupy distinct regions, confirming the clusters are real structure rather than noise.")
    show(C.persona_heatmap(cl["profile"]))
    explain("Average profile of each persona (blue = above average, red = below).", "A heat-table reads like a persona card.",
            "Premium Trendsetters (young, frequent, high spend, ~90% interested) are the launch core; Mainstream Pragmatists "
            "are the scale market; Deal Hunters are young but price-led; Conservative Traditionalists are a poor fit.")
    actions = {"Premium Trendsetters": "VIP early access, exclusive drops, personal styling, Instagram creators.",
               "Mainstream Pragmatists": "Authenticity guarantee, easy returns, loyalty points, family occasions (Diwali, weddings).",
               "Deal Hunters": "Member-only (not public) sales, bundles, student pricing on entry brands - protect margin.",
               "Conservative Traditionalists": "Don't chase. Reach only via store walk-ins and ethnic-fusion ranges."}
    st.dataframe(cl["profile"].assign(action=cl["profile"].persona.map(actions)).round(2), width="stretch")

# =================================================================== ASSOCIATION
elif page == PAGES[7]:
    st.title("Association rule mining — which brands, and which together?")
    st.markdown("Baskets = brands each **interested** respondent would buy (Q26). Apriori finds brand combinations that occur "
                "together more than chance (**lift > 1**).")
    c1, c2 = st.columns(2)
    ms = c1.slider("Minimum support", 0.02, 0.15, 0.05, 0.01)
    mc = c2.slider("Minimum confidence", 0.2, 0.8, 0.4, 0.05)
    rules, single, nb = A.association_rules(df, min_support=ms, min_conf=mc)
    st.caption(f"{nb} interested baskets · {len(rules)} rules")
    if len(rules):
        st.dataframe(rules.style.format({"Support": "{:.1%}", "Confidence": "{:.0%}", "Lift": "{:.2f}", "Leverage": "{:.3f}"}),
                     width="stretch", height=320)
        show(C.rules_scatter(rules))
        explain("Each rule plotted by support (how common) and confidence (how reliable); bubble = lift.",
                "Good rules are up-and-right with high lift.",
                "Brand 'families' appear clearly: Nike–Adidas–Puma (sport), Tommy Hilfiger–Calvin Klein (premium classic), "
                "Charles & Keith–ALDO (accessories), Zara–H&M–Mango (trend). Use them for bundles, store adjacency and "
                "'customers also bought' recommendations.")
    else:
        st.warning("No rules at these thresholds - lower support or confidence.")
    opt = A.launch_brand_optimizer(df)
    show(C.coverage_curve(opt))
    n80 = int(opt[opt["Cumulative coverage of interested customers"] >= 0.8].Step.min())
    explain("Greedy optimisation: add the brand that reaches the most NOT-yet-covered interested customers, one at a time.",
            "Answers the 'baby steps' question directly - diminishing returns become visible.",
            f"**{n80} brands reach 80%** of interested customers. Launch with one anchor per brand family "
            f"({', '.join(opt['Add brand'].head(n80))}), then add family partners (e.g. Adidas after Nike) using the rules above.")
    st.dataframe(opt.style.format({"Cumulative coverage of interested customers": "{:.0%}"}), width="stretch")

# =================================================================== TEXT
elif page == PAGES[8]:
    st.title("Text analytics — what customers say (Q29)")
    ta = cached_text(df, truth)
    d = ta["data"]
    c = st.columns(4)
    c[0].metric("Comments analysed", f"{len(d):,}")
    for i, s in enumerate(["Positive", "Neutral", "Negative"]):
        c[i + 1].metric(s, f"{(d.sentiment == s).mean():.0%}")
    show(C.sentiment_by_intent(d))
    explain("Sentiment of free-text comments within each intent group.", "100% stacked bars compare mix across groups.",
            "Sentiment tracks stated intent closely - the text confirms the numbers. 'Not sure' respondents are mostly neutral: "
            "they're waiting on prices and the brand list.")
    show(C.topic_chart(ta["topics"]))
    explain("How often each pain-point topic is mentioned, and how negative it is.", "Two aligned panels: volume vs pain.",
            "**Store experience** (crowds, traffic, staff) is the most negative topic, followed by price and authenticity. "
            "→ Design the store as a calm, appointment-friendly experience; make the authenticity guarantee visible; offer "
            "90-min delivery so people can skip the mall.")
    show(C.terms_chart(ta["pos_terms"], ta["neg_terms"]))
    explain("Words/phrases with the highest TF-IDF weight in positive vs negative comments.",
            "TF-IDF surfaces distinctive words, not just frequent ones.",
            "Positive: brands, delivery, variety, 'love', 'Mumbai'. Negative: malls, fake, sellers - trust and mall fatigue.")
    st.subheader("Model validation")
    st.markdown(f"- Lexicon model (no training, handles negation & Hinglish): **{ta['lexicon_accuracy']:.0%}** accuracy vs hidden truth.  \n"
                f"- TF-IDF + logistic regression trained on 300 hand-labelled comments: **{ta['supervised_accuracy']:.0%}**.  \n"
                "The supervised score is inflated because synthetic comments reuse templates - with real comments expect 75–85%.")
    st.dataframe(ta["lexicon_cm"], width="stretch")
    st.subheader("Try it")
    txt = st.text_input("Type a customer comment", "Love the idea but returns on apps are a pain")
    sc = A.lexicon_sentiment(txt)
    st.write(f"Score {sc:+.2f} → **{'Positive' if sc > .15 else 'Negative' if sc < -.15 else 'Neutral'}** · topics: {', '.join(A.detect_topics(txt))}")
    with st.expander("Sample comments"):
        st.dataframe(d[["Q29_feedback_text", "sentiment", "sent_score", "Q30_purchase_intent"]].sample(30, random_state=1), width="stretch")

# =================================================================== BREAK-EVEN
elif page == PAGES[9]:
    st.title("Prescriptive — break-even, risk and the 'baby steps' decision")
    st.markdown(f"Survey-driven inputs: average order value **₹{SI['aov']:,.0f}** (median AOV of interested respondents) and "
                f"stated monthly spend **₹{SI['stated_monthly_spend']:,.0f}**, of which we assume only a fraction is realised.")
    name = st.selectbox("Scenario", list(A.SCENARIOS), index=2)
    base = A.SCENARIOS[name]["p"] if "p2" not in A.SCENARIOS[name] else A.SCENARIOS[name]["p2"]
    with st.expander("Adjust assumptions (apply to every stage of the scenario)", expanded=True):
        c = st.columns(4)
        ov = {
            "realisation": c[0].slider("Intent realisation", 0.15, 0.8, float(base["realisation"]), 0.05, help=A.ASSUMPTION_NOTES["realisation"]),
            "gross_margin": c[1].slider("Gross margin", 0.20, 0.45, float(base["gross_margin"]), 0.01, help=A.ASSUMPTION_NOTES["gross_margin"]),
            "cac": c[2].slider("CAC (₹)", 400, 3000, int(base["cac"]), 100, help=A.ASSUMPTION_NOTES["cac"]),
            "monthly_churn": c[3].slider("Monthly churn", 0.02, 0.15, float(base["monthly_churn"]), 0.01, help=A.ASSUMPTION_NOTES["monthly_churn"]),
        }
        c = st.columns(4)
        ov["rent_per_sqft"] = c[0].slider("Rent ₹/sq ft/month", 200, 800, int(base["rent_per_sqft"]), 25, help=A.ASSUMPTION_NOTES["rent_per_sqft"])
        ov["return_rate"] = c[1].slider("Online return rate", 0.05, 0.40, float(base["return_rate"]), 0.01)
        aov = c[2].slider("Average order value (₹)", 1000, 6000, int(SI["aov"]), 100)
        ov["online_share"] = c[3].slider("Online share of orders", 0.2, 0.9, float(base["online_share"]), 0.05, help=A.ASSUMPTION_NOTES["online_share"])
    # apply overrides to both stages
    sc = A.SCENARIOS[name]
    p = dict(sc["p"], **{k: v for k, v in ov.items() if k != "online_share"})
    p2 = dict(sc["p2"], **ov) if "p2" in sc else None
    if p2 is None: p["online_share"] = ov["online_share"]
    be = A.break_even(p, aov, SI["stated_monthly_spend"], 48, p2, sc.get("switch"))
    m = st.columns(6)
    m[0].metric("Fixed cost / month", money(be["fixed_monthly"]))
    m[1].metric("Total capex", money(be["capex"]))
    m[2].metric("Contribution / order", f"₹{be['contrib_per_order']:,.0f}", f"{be['cm_ratio']:.0%} of AOV", delta_color="off")
    m[3].metric("Break-even orders / month", f"{be['be_orders']:,.0f}")
    m[4].metric("Break-even revenue / month", money(be["be_revenue"]))
    m[5].metric("Active customers needed", f"{be['be_customers']:,.0f}")
    m = st.columns(4)
    m[0].metric("Operating break-even month", be["be_month"] or "> 48")
    m[1].metric("Capex payback month", be["payback_month"] or "> 48")
    m[2].metric("Peak cash needed", money(be["peak_cash_need"]))
    m[3].metric("LTV / CAC", f"{be['ltv_cac']:.1f}×")
    st.markdown(r"**Formulas:** Contribution/order = AOV×(GM − payment − loyalty) − online% × (fulfilment + returns×reverse cost)  ·  "
                r"Break-even orders = Fixed cost ÷ contribution/order  ·  LTV = orders/customer/month × contribution ÷ churn")
    sims = {k: A.run_scenario(k, SI["aov"], SI["stated_monthly_spend"])["sim"] for k in A.SCENARIOS}
    sims[f"Your settings: {name.split('.')[0]}"] = be["sim"]
    c1, c2 = st.columns(2)
    with c1:
        show(C.cash_curves({k: v for k, v in sims.items()}))
        explain("Cumulative cash after the initial investment, month by month.", "The lowest point = cash you must raise.",
                "The flagship digs the deepest hole fastest but climbs fastest. The staged plan invests a second time at "
                "month 13, so it only pays off if the pilot proves demand.")
    with c2:
        show(C.profit_curves({k: v for k, v in sims.items()}))
        explain("Monthly operating profit after acquisition marketing.", "Break-even = first month above zero.",
                "Customers accumulate (acquisition > churn), so profit improves every month. The lean pilot on its own "
                "is too small to cover tech + head-office costs - it's a test, not a business.")
    st.subheader("Probability of failure — Monte-Carlo (1,000 simulated futures)")
    n = st.select_slider("Simulations", [200, 500, 1000], value=500)
    mcs = {k: cached_mc(k, SI["aov"], SI["stated_monthly_spend"], n) for k in A.SCENARIOS}
    show(C.mc_hist(mcs))
    comp = pd.DataFrame([{"Scenario": k, "P(monthly profit ≤ 24m)": v["p_profitable_24m"], "P(payback ≤ 48m)": v["p_payback_48m"],
                          "Expected cash @48m": money(v["expected_cash_48m"]), "Bad case (P10) cash @48m": money(v["p10_cash_48m"]),
                          "P(loss > ₹5 Cr)": v["p_big_loss"], "P(scale-up)": v.get("p_scale_up", np.nan)} for k, v in mcs.items()])
    st.dataframe(comp.style.format({c: "{:.0%}" for c in comp.columns if c.startswith("P(")}, na_rep="–"), width="stretch")
    explain("Distribution of cash at month 48 when margin, CAC, churn, realisation, rent, returns and AOV vary together.",
            "Point estimates hide risk; Monte-Carlo turns assumptions into a probability of failure.",
            f"The staged plan has a lower expected payoff than going big, but its bad-case loss is far smaller: when the pilot "
            f"shows realisation < {A.GATE['min_realisation']:.0%} or margin < {A.GATE['min_gross_margin']:.0%}, we stop and lose "
            "~₹4 Cr instead of ₹10–20 Cr. That is the quantitative case for 'baby steps'.")
    show(C.tornado(A.sensitivity(name, SI["aov"], SI["stated_monthly_spend"]), f"Sensitivity: change in cash at month 36 - {name.split('(')[0]}"))
    explain("Change in month-36 cash when each driver moves ±20% (others fixed).", "A tornado ranks what matters most.",
            "Gross margin and intent realisation dominate - negotiate brand margins hard and measure real conversion in "
            "the pilot before scaling. Rent matters less than people think.")

# =================================================================== SUSTAINABILITY
elif page == PAGES[10]:
    st.title("Is the business sustainable?")
    rec = list(A.SCENARIOS)[2]
    be = A.run_scenario(rec, SI["aov"], SI["stated_monthly_spend"])
    mc = cached_mc(rec, SI["aov"], SI["stated_monthly_spend"], 500)
    _, single, _ = A.association_rules(df)
    card, overall = A.sustainability_scorecard(df, be, mc, single, A.conversion_segment(df.p_interest))
    st.metric("Overall sustainability score (staged plan)", f"{overall:.0f} / 100")
    show(C.scorecard_chart(card))
    st.dataframe(card, width="stretch")
    explain("Nine indicators across financial, market, operational and environmental/social pillars, each scaled 0–100.",
            "A scorecard combines very different evidence into one view; colour + label show status.",
            "Market demand is strong; financial sustainability is the weak pillar (thin retail margins, cash-hungry ramp). "
            "Sustainable growth therefore depends on margin (own-label + outright buys later), retention (loyalty) and "
            "staged capital, not on growth at any cost.")
    st.subheader("Making it sustainable")
    st.markdown("""
- **Financial:** sale-or-return stock in year 1 (no dead inventory), demand forecasting before outright buys, loyalty to cut churn, no blanket discounting.
- **Environmental:** a 'Conscious Edit' of certified brands, recycled packaging, consolidated 90-min deliveries by EV / cycle couriers in dense zones, a take-back & resale corner.
- **Social:** local and Indian designer labels beside global brands, fair wages and training for store staff, accessible sizing.
- **Data ethics:** clear consent for personalisation (DPDP Act 2023), let users opt out of tracking without losing service.
""")

# =================================================================== SCORER
elif page == PAGES[11]:
    st.title("🎯 Score a new customer — the algorithms, connected")
    st.markdown("Fill the survey for a new prospect (or upload a CSV). The trained classifier, regressor and association rules "
                "return conversion probability, segment, predicted spend, brand recommendations and the next-best action.")
    rules, _, _ = A.association_rules(df)
    with st.form("score"):
        c = st.columns(3)
        ans = {
            "Q1_age": c[0].number_input("Q1 Age", 16, 80, 27),
            "Q2_gender": c[1].selectbox("Q2 Gender", GENDERS),
            "Q3_area": c[2].selectbox("Q3 Area", AREAS, index=1),
            "Q4_occupation": c[0].selectbox("Q4 Occupation", OCCUPATIONS, index=1),
            "Q5_household_income": c[1].selectbox("Q5 Household income", INCOME_BANDS, index=3),
            "Q6_household_type": c[2].selectbox("Q6 Household", HOUSEHOLD),
            "Q7_education": c[0].selectbox("Q7 Education", EDUCATION, index=1),
            "Q8_shop_freq_month": c[1].number_input("Q8 Shopping trips / month", 0, 31, 3),
            "Q9_monthly_fashion_spend": c[2].number_input("Q9 Monthly fashion spend (₹)", 0, 200000, 6000, 500),
            "Q10_channel": c[0].selectbox("Q10 Channel", CHANNELS),
            "Q13_avg_order_value": c[1].number_input("Q13 Typical bill (₹)", 0, 60000, 2500, 100),
            "Q14_social_hrs_week": c[2].number_input("Q14 Social media hrs / week", 0, 112, 8),
            "Q15_bought_intl_brand": c[0].selectbox("Q15 Bought intl brand (6 m)?", ["Yes", "No"]),
            "Q27_attraction_driver": c[1].selectbox("Q27 Main attraction", DRIVERS, index=1),
        }
        plats = c[2].multiselect("Q11 Platforms used", PLATFORMS, ["Myntra", "Amazon Fashion"])
        cats = c[0].multiselect("Q12 Categories", CATEGORIES, ["Apparel", "Footwear"])
        brands = c[1].multiselect("Q26 Brands wanted", BRANDS, ["Nike", "Puma"])
        st.markdown("**Attitudes (1 = strongly disagree … 5 = strongly agree)**")
        lc = st.columns(5)
        for i, (k, q) in enumerate(LIKERT_COLS.items()):
            ans[k] = lc[i % 5].slider(k.split("_", 1)[1].replace("_", " "), 1, 5, 4 if i in (0, 1, 3, 4, 5) else 3, help=q)
        go = st.form_submit_button("Score this customer")
    if go:
        ans.update({"income_ord": INCOME_BANDS.index(ans["Q5_household_income"]), "edu_ord": EDUCATION.index(ans["Q7_education"]),
                    "log_monthly_spend": np.log1p(ans["Q9_monthly_fashion_spend"]), "intl_buyer": int(ans["Q15_bought_intl_brand"] == "Yes"),
                    "n_plat": len(plats), "n_cat": len(cats), "n_brand": len(brands),
                    "core_zone": int(ans["Q3_area"] in ("South Mumbai", "Western Suburbs")), "liked_brands": brands})
        r = A.score_new_customer(ans, cls["model"], reg["model"], rules)
        m = st.columns(3)
        m[0].metric("Probability of becoming a customer", f"{r['probability']:.0%}")
        m[1].metric("Conversion segment", r["segment"])
        m[2].metric("Predicted monthly spend (stated)", f"₹{r['predicted_spend']:,.0f}")
        st.success(f"**Next-best action:** {r['action']}")
        st.info("**Recommend next (association rules):** " + (", ".join(r["recommended_brands"]) or "no strong rule - show bestsellers"))
    st.markdown("---")
    st.subheader("Batch scoring")
    up = st.file_uploader("Upload a CSV with the same columns as data/clean_survey.csv", type="csv")
    if up is not None:
        new = pd.read_csv(up)
        missing = [c for c in A.NUM_FEATURES + A.CAT_FEATURES if c not in new.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            new["p_interest"] = cls["model"].predict_proba(new[A.NUM_FEATURES + A.CAT_FEATURES])[:, 1]
            new["segment"] = A.conversion_segment(new.p_interest)
            st.dataframe(new[["p_interest", "segment"] + A.CAT_FEATURES[:3]].head(200), width="stretch")
            st.download_button("Download scored file", new.to_csv(index=False), "scored.csv")
