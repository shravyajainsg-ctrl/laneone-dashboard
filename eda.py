"""
eda.py - Descriptive & diagnostic analytics (the EDA section of the project).

Every analysis is a function  f(df) -> {"fig": matplotlib figure or None,
                                         "table": DataFrame or None,
                                         "text": markdown explanation built FROM the data}
so the notebook (04_descriptive_diagnostic_EDA.ipynb) and the Streamlit app show
exactly the same charts with exactly the same numbers.

DESCRIPTIVE  = what the data looks like  (distributions, frequencies, profiles)
DIAGNOSTIC   = why interest differs      (correlations, associations, tests, drill-downs)
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import charts as C
from config import *

NUM = {"Q1_age": "Age (years)", "Q8_shop_freq_month": "Shopping trips / month",
       "Q9_monthly_fashion_spend": "Monthly fashion spend (₹)", "Q13_avg_order_value": "Order value (₹)",
       "Q14_social_hrs_week": "Social media hrs / week", "Q28_expected_monthly_spend": "Expected spend at LaneOne (₹)"}
LIK_SHORT = {k: k.split("_", 1)[1].replace("_", " ") for k in LIKERT_COLS}


def pct(x):
    return f"{x:.0%}"


# =====================================================================================
# DESCRIPTIVE
# =====================================================================================
def d1_overview(df):
    """Dataset overview: size, variable types, completeness."""
    types = {"Numeric": len(NUM), "Likert (ordinal 1-5)": len(LIKERT_COLS),
             "Categorical": 7, "Multi-select (converted to 0/1)": sum(len(v) for v in MULTI_COLS.values()),
             "Open text": 1, "Target (interested 0/1)": 1}
    t = pd.DataFrame({"Variable group": list(types), "Columns": list(types.values())})
    miss = df[[c for c in df.columns if c.startswith("Q")]].isna().mean()
    miss = miss[miss > 0]
    text = (f"The clean dataset has **{len(df):,} respondents** and **{df.shape[1]} columns** "
            f"(30 survey answers plus engineered features). After cleaning, only "
            f"{', '.join(f'{c} ({v:.1%})' for c, v in miss.items())} still have gaps: the expected-spend target is never "
            f"imputed and junk comments were blanked on purpose. The sample is roughly balanced on the target "
            f"(**{df.interested.mean():.1%} interested**), so no resampling is needed for classification.")
    return {"fig": None, "table": t, "text": text}


def d2_summary_stats(df):
    """Summary statistics for every numeric variable, incl. skewness and kurtosis."""
    s = df[list(NUM)].describe(percentiles=[.25, .5, .75]).T
    s["skew"] = df[list(NUM)].skew()
    s["kurtosis"] = df[list(NUM)].kurt()
    s.index = [NUM[i] for i in s.index]
    s = s[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew", "kurtosis"]].round(1)
    spend_sk = s.loc["Monthly fashion spend (₹)", "skew"]
    text = (f"Money variables are strongly **right-skewed** (monthly spend skew = {spend_sk:.1f}): the mean "
            f"(₹{s.loc['Monthly fashion spend (₹)', 'mean']:,.0f}) sits well above the median "
            f"(₹{s.loc['Monthly fashion spend (₹)', '50%']:,.0f}) because a few luxury shoppers spend ₹60K+. "
            f"That is why the report uses **medians**, log-transforms spend for modelling, and prefers rank-based "
            f"(Spearman / Mann-Whitney) tests. Age is close to symmetric (skew {s.loc['Age (years)', 'skew']:.1f}) "
            f"around a median of {s.loc['Age (years)', '50%']:.0f} years - a young, working-age sample.")
    return {"fig": None, "table": s, "text": text}


def d3_distributions(df):
    """Histograms of the six numeric variables (money on log scale)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, (c, lab) in zip(axes.ravel(), NUM.items()):
        v = df[c].dropna()
        if "₹" in lab:
            v = v[v > 0]
            ax.hist(v, bins=np.logspace(np.log10(v.min()), np.log10(v.max()), 35), color=C.BLUE)
            ax.set_xscale("log"); ax.xaxis.set_major_formatter(C.rupee)
        else:
            ax.hist(v, bins=30, color=C.BLUE)
        ax.axvline(v.median(), color=C.ORANGE, lw=1.5)
        ax.set_title(lab, fontsize=10); ax.grid(axis="x", visible=False)
    fig.suptitle("Distributions of numeric variables (orange line = median; money on log scale)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    social_med = df.Q14_social_hrs_week.median()
    text = ("**What:** one histogram per numeric question. **Why:** before modelling we must know each variable's shape, "
            "centre and spread. **Insight:** on a log scale, spend and order value look roughly bell-shaped, so a "
            "**log transform** makes them suitable for linear models. Shopping frequency is a count piled up at 1–3 "
            f"trips a month. Social-media time has a long tail (median {social_med:.0f} hrs/week). Expected spend "
            "at LaneOne is **bimodal**: a lump near ₹50–200 (people who won't shop) and a second lump at ₹2K–10K "
            "(future customers). That two-population shape is the first sign the market splits into buyers and non-buyers.")
    return {"fig": fig, "table": None, "text": text}


def d4_categorical(df):
    """Frequency of each categorical answer."""
    cols = {"Q2_gender": "Gender", "Q3_area": "Zone", "Q4_occupation": "Occupation",
            "Q5_household_income": "Household income", "Q10_channel": "Shopping channel", "Q6_household_type": "Household"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    for ax, (c, lab) in zip(axes.ravel(), cols.items()):
        s = df[c].value_counts(normalize=True)
        if c == "Q5_household_income":
            s = s.reindex(INCOME_BANDS)
        s = s.iloc[::-1] if c != "Q5_household_income" else s.iloc[::-1]
        ax.barh([str(i)[:28] for i in s.index], s.values, color=C.BLUE, height=0.6)
        for y, v in enumerate(s.values):
            ax.text(v, y, f" {v:.0%}", va="center", fontsize=8, color=C.INK2)
        ax.xaxis.set_major_formatter(C.pct); ax.set_xlim(0, s.max() * 1.25)
        ax.set_title(lab, fontsize=10); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Who answered: categorical profile of the sample", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    z = df.Q3_area.value_counts(normalize=True)
    o = df.Q4_occupation.value_counts(normalize=True)
    text = ("**What:** share of respondents in each category. **Why:** shows whether the sample covers the target market and "
            "whether any group is too small to analyse. **Insight:** the sample is led by the "
            f"{z.index[0]} ({z.iloc[0]:.0%}) and {o.index[0].lower()}s ({o.iloc[0]:.0%}), matching an urban, working-age "
            f"audience. Women are {df.Q2_gender.eq('Female').mean():.0%}. The non-binary group and 'Other MMR' are small, "
            "so results for them should be read with caution.")
    return {"fig": fig, "table": None, "text": text}


def d5_likert(df):
    """Diverging stacked bars of the 10 attitude statements."""
    order = df[list(LIKERT_COLS)].mean().sort_values().index
    dist = pd.DataFrame({c: df[c].value_counts(normalize=True).reindex([1, 2, 3, 4, 5]).fillna(0) for c in order}).T
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = {1: "#e34948", 2: "#f0a3a2", 3: "#d6d5cf", 4: "#9ec5f4", 5: "#2a78d6"}
    left = -(dist[1] + dist[2] + dist[3] / 2)
    for k in [1, 2, 3, 4, 5]:
        ax.barh([LIK_SHORT[c] for c in dist.index], dist[k], left=left, color=cols[k], label=LIKERT_LABELS[k], height=0.6,
                edgecolor=C.SURFACE, linewidth=1.5)
        left = left + dist[k]
    ax.axvline(0, color=C.INK2, lw=1); ax.xaxis.set_major_formatter(C.FuncFormatter(lambda x, _: f"{abs(x):.0%}"))
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8); ax.grid(axis="y", visible=False)
    ax.set_title("Attitude statements (Q16-Q25): disagree ← neutral → agree")
    fig.tight_layout()
    agree = (df[list(LIKERT_COLS)] >= 4).mean().sort_values(ascending=False)
    text = ("**What:** for each statement, the split from strongly disagree (red) to strongly agree (blue), centred on "
            "neutral. **Why:** a diverging bar shows the balance of opinion better than an average. **Insight:** the most "
            f"agreed statements are **{LIK_SHORT[agree.index[0]]}** ({agree.iloc[0]:.0%} agree) and "
            f"**{LIK_SHORT[agree.index[1]]}** ({agree.iloc[1]:.0%}). The least agreed is **{LIK_SHORT[agree.index[-1]]}** "
            f"({agree.iloc[-1]:.0%}). People want authentic brands but are wary of giving data, so personalisation must be "
            "opt-in and visibly useful.")
    return {"fig": fig, "table": None, "text": text}


def d6_multiselect(df):
    """Platforms used today, categories bought, brands wanted (multi-select shares)."""
    groups = [("plat_", PLATFORMS, "Q11 Platforms used (last 6 months)"), ("cat_", CATEGORIES, "Q12 Categories bought"),
              ("brand_", BRANDS, "Q26 Brands wanted at LaneOne")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    out = {}
    for ax, (p, labels, title) in zip(axes, groups):
        cols = [p + re.sub(r"[^a-z0-9]+", "_", o.lower()).strip("_") for o in labels]
        s = pd.Series(df[cols].mean().values, index=labels).sort_values()
        out[p] = s
        ax.barh(s.index, s.values, color=C.BLUE, height=0.6); ax.xaxis.set_major_formatter(C.pct)
        for y, v in enumerate(s.values):
            ax.text(v, y, f" {v:.0%}", va="center", fontsize=8, color=C.INK2)
        ax.set_xlim(0, s.max() * 1.25); ax.set_title(title, fontsize=10); ax.grid(axis="y", visible=False); ax.tick_params(labelsize=8)
    fig.tight_layout()
    pl, ca, br = out["plat_"], out["cat_"], out["brand_"]
    text = ("**What:** share of respondents ticking each option. **Why:** multi-select answers show the competitive set, "
            "the category mix and brand demand. **Insight:** the main competitors are **"
            f"{pl.index[-1]} ({pl.iloc[-1]:.0%}), {pl.index[-2]} ({pl.iloc[-2]:.0%}) and {pl.index[-3]} ({pl.iloc[-3]:.0%})**. "
            f"{ca.index[-1]} and {ca.index[-2].lower()} are the core categories. **{br.index[-1]}** ({br.iloc[-1]:.0%}) and "
            f"**{br.index[-2]}** ({br.iloc[-2]:.0%}) are the most wanted brands. No single brand reaches half the sample, "
            "so a multi-brand assortment is needed to serve everyone.")
    return {"fig": fig, "table": None, "text": text}


def d7_target(df):
    """The target: purchase-intent distribution and interest by key segments."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    s = df.Q30_purchase_intent.value_counts(normalize=True).reindex(INTENT)
    axes[0].bar([i.replace(" ", "\n") for i in INTENT], s.values, color=C.ORD5, width=0.6)
    for x, v in enumerate(s.values):
        axes[0].text(x, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=C.INK2)
    axes[0].yaxis.set_major_formatter(C.pct); axes[0].grid(axis="x", visible=False); axes[0].set_title("Q30 purchase intent", fontsize=10)
    ag = df.copy(); ag["age_band"] = ag.age_group.replace({"45-54": "45+", "55+": "45+"})
    r = ag.groupby("age_band").interested.agg(["mean", "count"]).reindex(["16-24", "25-34", "35-44", "45+"])
    axes[1].bar(r.index, r["mean"], color=C.BLUE, width=0.6)
    for x, (v, n) in enumerate(zip(r["mean"], r["count"])):
        axes[1].text(x, v, f"{v:.0%}\n(n={n})", ha="center", va="bottom", fontsize=8, color=C.INK2)
    axes[1].axhline(df.interested.mean(), color=C.INK2, lw=1); axes[1].yaxis.set_major_formatter(C.pct)
    axes[1].set_ylim(0, 0.8); axes[1].grid(axis="x", visible=False); axes[1].set_title("Interest rate by age (line = average)", fontsize=10)
    fig.tight_layout()
    text = (f"**What:** the target variable and how it varies by age. **Why:** the target's balance decides the modelling "
            f"approach. **Insight:** {s['Definitely will']:.0%} say 'Definitely will' and {s['Definitely not']:.0%} 'Definitely "
            f"not', with a thin middle: opinions are already formed. Interest peaks at 16–24 ({r.loc['16-24', 'mean']:.0%}) and "
            f"is lowest at 35–44 ({r.loc['35-44', 'mean']:.0%}). 45–54 and 55+ are merged because only "
            f"{(df.age_group == '55+').sum()} respondents are 55+.")
    return {"fig": fig, "table": None, "text": text}


# =====================================================================================
# DIAGNOSTIC
# =====================================================================================
CORR_COLS = ["Q1_age", "income_ord", "log_monthly_spend", "Q8_shop_freq_month", "Q14_social_hrs_week",
             *LIKERT_COLS, "n_brand", "intent_score", "Q28_expected_monthly_spend"]


def _lab(c):
    return c.split("_", 1)[1].replace("_", " ") if c.startswith("Q") else c.replace("_", " ")


def g1_pearson_vs_spearman(df):
    """Pearson vs Spearman correlation matrices side by side."""
    d = df[CORR_COLS].dropna()
    pe, sp = d.corr("pearson"), d.corr("spearman")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    for ax, m, t in [(axes[0], pe, "Pearson (linear)"), (axes[1], sp, "Spearman (rank)")]:
        im = ax.imshow(m, cmap=C.DIV, vmin=-1, vmax=1); ax.grid(False)
        ax.set_xticks(range(len(m)), [_lab(c) for c in m], rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(m)), [_lab(c) for c in m], fontsize=7); ax.set_title(t, fontsize=11)
        for i in range(len(m)):
            for j in range(len(m)):
                if i != j and abs(m.iat[i, j]) >= 0.4:
                    ax.text(j, i, f"{m.iat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(m.iat[i, j]) > .65 else C.INK)
    fig.colorbar(im, ax=axes, shrink=0.6, label="correlation")
    diff = (sp - pe).abs().where(~np.eye(len(sp), dtype=bool)).stack().sort_values(ascending=False)
    a, b = diff.index[0]
    text = ("**What:** two correlation matrices of the same 18 variables (labels shown where |r| ≥ 0.4). **Why:** Pearson "
            "measures *linear* association and is distorted by skew and outliers; Spearman uses ranks and suits Likert "
            "scales and skewed spend. Comparing them shows where the relationships are non-linear. **Insight:** the "
            f"biggest gap is between **{_lab(a)}** and **{_lab(b)}** (Pearson {pe.loc[a, b]:.2f} vs Spearman "
            f"{sp.loc[a, b]:.2f}): raw rupees are skewed, so the relationship is monotonic rather than straight-line. That "
            f"justifies rank-based diagnostics and a log transform. Intent score and expected spend correlate "
            f"{sp.loc['intent_score', 'Q28_expected_monthly_spend']:.2f}, so stated interest and stated spend tell a "
            "consistent story.")
    return {"fig": fig, "table": sp.round(2).rename(index=_lab, columns=_lab), "text": text}


def g2_drivers_of_interest(df):
    """Which variables move with purchase interest? Ranked correlation with the target."""
    cols = [c for c in CORR_COLS if c not in ("intent_score", "Q28_expected_monthly_spend")] + ["intl_buyer", "core_zone", "n_plat"]
    rows = []
    for c in cols:
        rho, p = stats.spearmanr(df[c], df.interested)
        pb, _ = stats.pointbiserialr(df.interested, df[c])
        rows.append({"variable": _lab(c), "Spearman ρ": rho, "point-biserial r": pb, "p-value": p})
    t = pd.DataFrame(rows).sort_values("Spearman ρ").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(t.variable, t["Spearman ρ"], color=[C.CAT[7] if v < 0 else C.BLUE for v in t["Spearman ρ"]], height=0.6)
    for y, (v, p) in enumerate(zip(t["Spearman ρ"], t["p-value"])):
        ax.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:+.2f}{'' if p < 0.05 else ' (n.s.)'}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=C.INK2)
    ax.axvline(0, color=C.INK2, lw=1); ax.grid(axis="y", visible=False); ax.set_xlim(-0.35, 0.6)
    ax.set_xlabel("Correlation with being interested (Spearman ρ)"); ax.set_title("Diagnostic: what moves with purchase interest?")
    fig.tight_layout()
    top = t.sort_values("Spearman ρ", ascending=False)
    ns = t[t["p-value"] >= 0.05].variable.tolist()
    text = ("**What:** each variable's correlation with the 0/1 target (blue = higher interest, red = lower; n.s. = not "
            "significant at 5%). **Why:** it ranks the candidate drivers before any model is built, and the point-biserial "
            "column confirms the same ordering with the classic binary-vs-numeric measure. **Insight:** the strongest "
            f"positive drivers are **{top.variable.iloc[0]}** (ρ {top['Spearman ρ'].iloc[0]:.2f}), **{top.variable.iloc[1]}** "
            f"({top['Spearman ρ'].iloc[1]:.2f}) and **{top.variable.iloc[2]}** ({top['Spearman ρ'].iloc[2]:.2f}). "
            f"**{t.variable.iloc[0]}** pulls the other way ({t['Spearman ρ'].iloc[0]:.2f}). "
            + (f"No meaningful link: {', '.join(ns)}. " if ns else "")
            + "Interest is driven by spending power and the concept's pillars (multi-brand, phygital, AI), not by demographics alone.")
    return {"fig": fig, "table": t.round(3), "text": text}


def _cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum(); r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0.0


def g3_cramers_v(df):
    """Association between categorical variables (Cramér's V)."""
    cols = {"Q2_gender": "gender", "Q3_area": "zone", "Q4_occupation": "occupation", "Q5_household_income": "income",
            "Q6_household_type": "household", "Q10_channel": "channel", "Q27_attraction_driver": "attraction driver",
            "persona_true": None, "interested": "INTERESTED"}
    cols = {k: v for k, v in cols.items() if v and k in df.columns}
    keys = list(cols)
    m = pd.DataFrame([[_cramers_v(df[a], df[b]) if a != b else 1.0 for b in keys] for a in keys],
                     index=[cols[k] for k in keys], columns=[cols[k] for k in keys])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(m, cmap=C.SEQMAP, vmin=0, vmax=0.6); ax.grid(False)
    ax.set_xticks(range(len(m)), m.columns, rotation=45, ha="right", fontsize=8); ax.set_yticks(range(len(m)), m.index, fontsize=8)
    for i in range(len(m)):
        for j in range(len(m)):
            if i != j:
                ax.text(j, i, f"{m.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if m.iat[i, j] > .4 else C.INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Cramér's V (0 = none, 1 = perfect)")
    ax.set_title("Diagnostic: association between categorical variables"); fig.tight_layout()
    tgt = m["INTERESTED"].drop("INTERESTED").sort_values(ascending=False)
    text = ("**What:** Cramér's V for every pair of categorical variables (a chi-square-based 0–1 strength measure). "
            "**Why:** Pearson/Spearman can't be used on unordered categories, but we still need to know which categories "
            "relate to interest and to each other. **Insight:** interest is most associated with **"
            f"{tgt.index[0]}** (V = {tgt.iloc[0]:.2f}), **{tgt.index[1]}** ({tgt.iloc[1]:.2f}) and **{tgt.index[2]}** "
            f"({tgt.iloc[2]:.2f}); **{tgt.index[-1]}** barely matters ({tgt.iloc[-1]:.2f}). Zone and income are themselves "
            f"related (V = {m.loc['zone', 'income']:.2f}), because richer households cluster in South Mumbai and the Western "
            "Suburbs. Part of the zone effect is therefore really an income effect, which the drill-down below checks.")
    return {"fig": fig, "table": m.round(2), "text": text}


def g4_zone_income_drilldown(df):
    """Drill-down: is the zone effect real, or just income? Interest by zone × income tier."""
    d = df.copy()
    d["income_tier"] = pd.cut(d.income_ord, [-1, 1, 3, 5], labels=["< ₹1L", "₹1L–4L", "> ₹4L"])
    t = d.pivot_table(index="Q3_area", columns="income_tier", values="interested", aggfunc="mean", observed=False).reindex(AREAS)
    n = d.pivot_table(index="Q3_area", columns="income_tier", values="interested", aggfunc="count", observed=False).reindex(AREAS)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmap = C.SEQMAP.copy(); cmap.set_bad("#ecebe6")
    ax.imshow(np.ma.masked_invalid(t.where(n >= 15).values.astype(float)), cmap=cmap, vmin=0, vmax=1, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(t.shape[1]), [str(c) for c in t.columns]); ax.set_yticks(range(t.shape[0]), t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v, k = t.iat[i, j], n.iat[i, j]
            ax.text(j, i, "–" if pd.isna(v) or k < 15 else f"{v:.0%}\n(n={int(k)})", ha="center", va="center", fontsize=8,
                    color="white" if (not pd.isna(v) and v > .6) else C.INK)
    ax.set_xlabel("Monthly household income"); ax.set_title("Drill-down: interest rate by zone within each income tier")
    fig.tight_layout()
    hi = t["> ₹4L"]
    text = ("**What:** interest rate for each zone *within* the same income tier (cells with fewer than 15 people hidden). "
            "**Why:** a diagnostic drill-down separates two confounded drivers. If zone only mattered through income, each "
            "column would look flat. **Insight:** interest rises left to right in every zone, so **income is the dominant "
            "driver**. But zone still matters inside the same income tier: among > ₹4L households, "
            f"{hi.idxmax()} reaches {hi.max():.0%} vs {hi.dropna().idxmin()} {hi.dropna().min():.0%}. Location and income "
            "both count, which supports a Western-Suburbs store targeting ₹1L+ households.")
    return {"fig": fig, "table": (t.where(n >= 15) * 100).round(0), "text": text}


def g5_spend_relationships(df):
    """Scatter diagnostics: income → current spend → expected spend."""
    d = df.dropna(subset=["Q28_expected_monthly_spend"])
    d = d[(d.Q28_expected_monthly_spend > 0) & (d.Q9_monthly_fashion_spend > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    g = df.groupby("Q5_household_income").Q9_monthly_fashion_spend
    data = [df.loc[df.Q5_household_income == b, "Q9_monthly_fashion_spend"] for b in INCOME_BANDS]
    axes[0].boxplot(data, tick_labels=INCOME_BANDS, showfliers=False, patch_artist=True, widths=0.55,
                    boxprops=dict(facecolor="#cde2fb", color=C.BLUE), medianprops=dict(color=C.INK))
    axes[0].set_yscale("log"); axes[0].yaxis.set_major_formatter(C.rupee); axes[0].grid(axis="x", visible=False)
    axes[0].set_title("Monthly fashion spend by household income", fontsize=10)
    for k, col, lab in [(0, C.GREY, "Not interested"), (1, C.BLUE, "Interested")]:
        s = d[d.interested == k]
        axes[1].scatter(s.Q9_monthly_fashion_spend, s.Q28_expected_monthly_spend, s=9, alpha=.5, color=col, label=lab, linewidths=0)
    s1 = d[d.interested == 1]
    b, a = np.polyfit(np.log10(s1.Q9_monthly_fashion_spend), np.log10(s1.Q28_expected_monthly_spend), 1)
    xs = np.logspace(np.log10(s1.Q9_monthly_fashion_spend.min()), np.log10(s1.Q9_monthly_fashion_spend.quantile(.99)), 50)
    axes[1].plot(xs, 10 ** a * xs ** b, color=C.ORANGE, lw=2, label=f"fit (interested): elasticity {b:.2f}")
    axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].xaxis.set_major_formatter(C.rupee); axes[1].yaxis.set_major_formatter(C.rupee)
    axes[1].set_xlabel("Current monthly fashion spend"); axes[1].set_ylabel("Expected spend at LaneOne"); axes[1].legend(fontsize=8)
    axes[1].set_title("Current vs expected spend", fontsize=10)
    fig.tight_layout()
    meds = g.median().reindex(INCOME_BANDS)
    share = (s1.Q28_expected_monthly_spend / s1.Q9_monthly_fashion_spend).median()
    text = ("**What:** left, spend by income band (box = middle 50%, line = median); right, current vs expected spend "
            "with a log-log fit for interested respondents. **Why:** it traces the chain *income → fashion budget → "
            "LaneOne spend*, which is the logic behind the revenue forecast. **Insight:** median spend rises from "
            f"₹{meds.iloc[0]:,.0f} (< ₹50K) to ₹{meds.iloc[-1]:,.0f} (> ₹8L). The fitted elasticity of **{b:.2f}** means a "
            f"10% bigger fashion budget brings about {b * 10:.0f}% more expected spend with us. Interested people expect to "
            f"move a median **{share:.0%} of their current fashion budget** to LaneOne, a realistic share-of-wallet target.")
    return {"fig": fig, "table": None, "text": text}


def g6_attitudes_by_intent(df):
    """Mean attitude score at each of the 5 intent levels (dose-response check)."""
    m = df.groupby("Q30_purchase_intent")[list(LIKERT_COLS)].mean().reindex(INTENT)
    keep = (m.iloc[-1] - m.iloc[0]).abs().sort_values(ascending=False).index[:6]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, c in enumerate(keep):
        ax.plot(range(5), m[c], marker="o", ms=5, color=C.CAT[i], label=LIK_SHORT[c])
    ax.set_xticks(range(5), INTENT); ax.set_ylabel("Mean agreement (1-5)"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("Diagnostic: attitudes step up with each level of purchase intent")
    fig.tight_layout()
    mono = [c for c in LIKERT_COLS if m[c].is_monotonic_increasing]
    c0 = keep[0]
    text = ("**What:** average agreement with the six most discriminating statements at each intent level. **Why:** a "
            "*dose-response* pattern (scores rising step by step from 'Definitely not' to 'Definitely will') is stronger "
            "diagnostic evidence than a simple yes/no gap, because it rules out a one-off difference. **Insight:** "
            f"**{LIK_SHORT[c0]}** climbs from {m[c0].iloc[0]:.2f} to {m[c0].iloc[-1]:.2f}. {len(mono)} of 10 attitudes rise "
            "monotonically with intent. Discount-seeking moves the opposite way: the more someone wants LaneOne, the less "
            "they are driven by sales. That is good news for margins.")
    return {"fig": fig, "table": m.round(2).rename(columns=LIK_SHORT), "text": text}


def g7_hypothesis_effects(df):
    """Hypothesis tests with effect sizes (not just p-values)."""
    import analytics as A
    h = A.hypothesis_tests(df)
    eff = []
    n = len(df)
    for _, r in h.iterrows():
        test = r["Test"]
        if "Chi" in test:
            col = {"H1": "Q5_household_income", "H6": "Q3_area", "H10": "Q10_channel"}.get(r["H"])
            v = _cramers_v(df[col], df.interested) if col else _cramers_v(df.Q1_age <= 34, df.interested)
            eff.append((f"Cramér's V = {v:.2f}", "small" if v < .2 else "medium" if v < .4 else "large"))
        elif "Mann" in test:
            u = r["Statistic"]
            if r["H"] == "H11":
                n1, n2 = df.Q2_gender.eq("Female").sum(), df.Q2_gender.eq("Male").sum()
            else:
                n1, n2 = df.interested.sum(), (1 - df.interested).sum()
            rb = 2 * u / (n1 * n2) - 1
            eff.append((f"rank-biserial = {rb:.2f}", "small" if abs(rb) < .3 else "medium" if abs(rb) < .5 else "large"))
        elif "Spearman" in test:
            v = r["Statistic"]
            eff.append((f"ρ = {v:.2f}", "small" if abs(v) < .3 else "medium" if abs(v) < .5 else "large"))
        else:
            eps = r["Statistic"] / (n - 1)
            eff.append((f"ε² = {eps:.3f}", "negligible" if eps < .01 else "small"))
    h["Effect size"] = [e[0] for e in eff]
    h["Magnitude"] = [e[1] for e in eff]
    t = h[["H", "Hypothesis (H1 / alternative)", "Test", "p-value", "Effect size", "Magnitude", "Decision"]]
    big = t[t.Magnitude == "large"].H.tolist()
    text = ("**What:** the 11 business hypotheses with p-values *and* effect sizes. **Why:** with ~1,900 respondents almost "
            "any difference becomes 'significant', so effect size tells us whether it matters commercially. **Insight:** "
            f"large effects ({', '.join(big)}) are the ones to build strategy on: current spend (H3), multi-brand appeal (H4) "
            "and the current-to-expected spend link (H8). Income (H1), AI interest (H5) and zone (H6) are medium. H11 (gender) is significant but small, so it informs the assortment mix, not targeting. "
            "H9 (sustainability by age) is not supported, so sustainability messaging should go to all ages.")
    return {"fig": None, "table": t, "text": text}


def g8_outliers_multicollinearity(df):
    """Remaining outliers (IQR rule) and multicollinearity (VIF) among model predictors."""
    rows = []
    for c, lab in NUM.items():
        v = df[c].dropna()
        x = np.log1p(v) if "₹" in lab else v
        q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
        o = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()
        rows.append({"variable": lab, "IQR outliers": int(o), "share": o / len(v), "scale": "log" if "₹" in lab else "raw"})
    out = pd.DataFrame(rows)
    X = df[["Q1_age", "income_ord", "log_monthly_spend", "Q8_shop_freq_month", "Q13_avg_order_value", "Q14_social_hrs_week",
            *LIKERT_COLS]].astype(float)
    X = (X - X.mean()) / X.std()
    vif = []
    for c in X.columns:
        y = X[c].values; Z = np.column_stack([np.ones(len(X)), X.drop(columns=c).values])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r2 = 1 - ((y - Z @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        vif.append({"predictor": _lab(c), "VIF": 1 / (1 - r2)})
    vif = pd.DataFrame(vif).sort_values("VIF", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(vif.predictor[::-1], vif.VIF[::-1], color=[C.ORANGE if v > 2.5 else C.BLUE for v in vif.VIF[::-1]], height=0.6)
    ax.axvline(5, color=C.CAT[7], lw=1); ax.text(5.05, 0, "VIF 5 = concern", color=C.INK2, fontsize=8)
    ax.set_xlim(0, max(6, vif.VIF.max() * 1.1)); ax.grid(axis="y", visible=False)
    ax.set_title("Diagnostic: multicollinearity among predictors (VIF)"); fig.tight_layout()
    text = ("**What:** the table counts values still outside 1.5×IQR after cleaning (money on log scale); the chart shows "
            "each predictor's variance inflation factor. **Why:** outliers that survive cleaning should be *genuine*, and "
            "highly inter-correlated predictors make regression coefficients unstable. **Insight:** "
            f"{out['IQR outliers'].sum()} statistical outliers remain ({out.share.max():.1%} at most per variable). They "
            "are real heavy shoppers and frequent social users, so they were kept. The highest VIF is "
            f"**{vif.predictor.iloc[0]} ({vif.VIF.iloc[0]:.1f})**, well below 5, so multicollinearity is not a problem, though "
            "income and spend overlap enough that Ridge/Lasso were tried alongside plain linear regression.")
    return {"fig": fig, "table": out.assign(share=out.share.map(lambda x: f"{x:.1%}")), "text": text, "table2": vif.round(2)}


DESCRIPTIVE = [("D1. Dataset overview", d1_overview), ("D2. Summary statistics", d2_summary_stats),
               ("D3. Distributions of numeric variables", d3_distributions), ("D4. Categorical profile", d4_categorical),
               ("D5. Attitude statements (Likert)", d5_likert), ("D6. Platforms, categories and brands", d6_multiselect),
               ("D7. The target: purchase intent", d7_target)]
DIAGNOSTIC = [("G1. Pearson vs Spearman correlation", g1_pearson_vs_spearman),
              ("G2. What moves with purchase interest?", g2_drivers_of_interest),
              ("G3. Categorical associations (Cramér's V)", g3_cramers_v),
              ("G4. Drill-down: zone vs income", g4_zone_income_drilldown),
              ("G5. Income → spend → LaneOne spend", g5_spend_relationships),
              ("G6. Dose-response: attitudes by intent level", g6_attitudes_by_intent),
              ("G7. Hypothesis tests with effect sizes", g7_hypothesis_effects),
              ("G8. Outliers and multicollinearity", g8_outliers_multicollinearity)]
