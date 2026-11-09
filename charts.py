"""
charts.py - every figure in the project (matplotlib, so it runs anywhere).
Colour rules: categorical slots in a fixed order (colour follows the entity),
one-hue blue ramp for magnitude, blue<->red with grey midpoint for polarity.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, PercentFormatter
from config import INTENT, INCOME_BANDS, LIKERT_COLS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, GREY = CAT[0], CAT[1], "#b5b4ad"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORD5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
SEQMAP = LinearSegmentedColormap.from_list("seq", SEQ)
STATUS = {"🟢": "#0ca30c", "🟡": "#fab219", "🔴": "#d03b3b"}
SENT = {"Negative": "#e34948", "Neutral": "#b5b4ad", "Positive": "#2a78d6"}
INTEREST = {0: GREY, 1: BLUE}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
    "font.family": "DejaVu Sans", "lines.linewidth": 2, "figure.dpi": 110,
})
rupee = FuncFormatter(lambda x, _: f"₹{x/1e7:.1f} Cr" if abs(x) >= 1e7 else (f"₹{x/1e5:.1f} L" if abs(x) >= 1e5 else f"₹{x:,.0f}"))
pct = PercentFormatter(1.0, decimals=0)


def _fig(w=7.5, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def _hbar(ax, s, color=BLUE, fmt="{:.0%}", xfmt=None):
    s = s.sort_values()
    ax.barh(s.index.astype(str), s.values, color=color, height=0.62)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(s.values):
        ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8.5, color=INK2)
    if xfmt: ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlim(0, s.max() * 1.18)

# ------------------------------------------------------------------ data quality
def missing_before_after(raw, clean, cols):
    fig, ax = _fig(7.5, 4.6)
    b = raw[cols].isna().mean(); a = clean[cols].isna().mean()
    y = np.arange(len(cols))
    ax.barh(y + 0.2, b.values, 0.38, color=ORANGE, label="Raw survey export")
    ax.barh(y - 0.2, a.values, 0.38, color=BLUE, label="After cleaning")
    ax.set_yticks(y, [c.split("_", 1)[1].replace("_", " ") for c in cols]); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(pct); ax.set_title("Missing values per question - before vs after cleaning")
    ax.legend(loc="lower right"); fig.tight_layout(); return fig


def outlier_box(raw_series, clean_series, title):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, s, lab, c in [(axes[0], raw_series, "Raw (parsed)", ORANGE), (axes[1], clean_series, "Cleaned", BLUE)]:
        s = pd.to_numeric(s, errors="coerce").dropna()
        ax.boxplot(s[s > 0], vert=False, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=c, alpha=.35, color=c), medianprops=dict(color=INK),
                   flierprops=dict(marker="o", markersize=3, markerfacecolor=c, markeredgecolor=c, alpha=.5))
        ax.set_xscale("log"); ax.set_yticks([]); ax.set_title(lab, fontsize=10)
        ax.xaxis.set_major_formatter(rupee)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", fontsize=12); fig.tight_layout(); return fig

# ------------------------------------------------------------------ descriptive
def intent_distribution(df):
    fig, ax = _fig(7.5, 3.6)
    s = df.Q30_purchase_intent.value_counts(normalize=True).reindex(INTENT)
    ax.bar(INTENT, s.values, color=ORD5, width=0.62)
    for x, v in enumerate(s.values): ax.text(x, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.yaxis.set_major_formatter(pct); ax.grid(axis="x", visible=False)
    ax.set_title("Q30 - Would you shop at us in the first 3 months?"); fig.tight_layout(); return fig


def rate_by(df, col, order=None, title=None, sort=True):
    fig, ax = _fig(7.5, 3.8)
    s = df.groupby(col, observed=True).interested.mean()
    if order is not None:
        s = s.reindex(order)
        ax.bar([str(i) for i in s.index], s.values, color=ORD5[:len(s)] if len(s) <= 5 else SEQ[1:len(s) + 1], width=0.62)
        for x, v in enumerate(s.values): ax.text(x, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.yaxis.set_major_formatter(pct); ax.grid(axis="x", visible=False)
        plt.setp(ax.get_xticklabels(), rotation=0)
    else:
        _hbar(ax, s)
        ax.xaxis.set_major_formatter(pct)
    ax.axhline if order is not None else None
    avg = df.interested.mean()
    (ax.axhline(avg, color=INK2, lw=1) if order is not None else ax.axvline(avg, color=INK2, lw=1))
    ax.set_title(title or f"Share interested by {col}"); fig.tight_layout(); return fig


def age_hist(df):
    fig, ax = _fig(7.5, 3.6)
    bins = np.arange(16, 70, 3)
    for k, lab in [(0, "Not interested"), (1, "Interested")]:
        ax.hist(df.loc[df.interested == k, "Q1_age"], bins=bins, histtype="step", lw=2, color=INTEREST[k], label=lab)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Respondents"); ax.legend()
    ax.set_title("Age profile - interested vs not interested"); fig.tight_layout(); return fig


def likert_dumbbell(df):
    fig, ax = _fig(8, 4.8)
    m = df.groupby("interested")[list(LIKERT_COLS)].mean().T
    m = m.assign(gap=m[1] - m[0]).sort_values("gap")
    y = np.arange(len(m))
    ax.hlines(y, m[0], m[1], color=GRID, lw=3)
    ax.scatter(m[0], y, s=60, color=GREY, label="Not interested", zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.scatter(m[1], y, s=60, color=BLUE, label="Interested", zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, [c.split("_", 1)[1].replace("_", " ") for c in m.index]); ax.grid(axis="y", visible=False)
    ax.set_xlim(1, 5); ax.set_xlabel("Mean agreement (1 = strongly disagree, 5 = strongly agree)")
    ax.legend(loc="lower right"); ax.set_title("Attitudes that separate future customers from non-customers")
    fig.tight_layout(); return fig


def share_bars(df, prefix, labels, title):
    fig, ax = _fig(7.5, 3.8)
    cols = [c for c in df.columns if c.startswith(prefix) and not c.startswith("n_")]
    s = pd.Series(df[cols].mean().values, index=labels)
    _hbar(ax, s); ax.xaxis.set_major_formatter(pct); ax.set_title(title); fig.tight_layout(); return fig


def driver_split(df):
    fig, ax = _fig(8, 3.8)
    t = pd.crosstab(df.Q27_attraction_driver, df.interested, normalize="columns")
    t = t.sort_values(1)
    y = np.arange(len(t))
    ax.barh(y + 0.2, t[1], 0.38, color=BLUE, label="Interested"); ax.barh(y - 0.2, t[0], 0.38, color=GREY, label="Not interested")
    ax.set_yticks(y, t.index); ax.grid(axis="y", visible=False); ax.xaxis.set_major_formatter(pct)
    ax.legend(loc="lower right"); ax.set_title("Q27 - What would most make you try us?"); fig.tight_layout(); return fig

# ------------------------------------------------------------------ diagnostic
def corr_heatmap(df, cols, labels):
    c = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(c, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1))
    ax.set_xticks(range(len(cols)), labels, rotation=60, ha="right"); ax.set_yticks(range(len(cols)), labels)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = c.iat[i, j]
            if abs(v) >= 0.3 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ")
    ax.set_title("Correlation matrix (labels shown where |ρ| ≥ 0.3)"); fig.tight_layout(); return fig, c


def spend_scatter(df):
    fig, ax = _fig(7.5, 4.6)
    d = df.dropna(subset=["Q28_expected_monthly_spend"])
    d = d[d.Q28_expected_monthly_spend > 0]
    for k, lab in [(0, "Not interested"), (1, "Interested")]:
        s = d[d.interested == k]
        ax.scatter(s.Q9_monthly_fashion_spend, s.Q28_expected_monthly_spend, s=12, alpha=.55, color=INTEREST[k], label=lab, linewidths=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.xaxis.set_major_formatter(rupee); ax.yaxis.set_major_formatter(rupee)
    ax.set_xlabel("Current monthly fashion spend (Q9)"); ax.set_ylabel("Expected monthly spend at us (Q28)")
    ax.legend(); ax.set_title("Current spend vs expected spend at us"); fig.tight_layout(); return fig

# ------------------------------------------------------------------ predictive: classification
def model_bars(table, metric, title, fmt="{:.3f}"):
    fig, ax = _fig(7.5, 3.4)
    s = table.set_index("Model")[metric]
    _hbar(ax, s, fmt=fmt); ax.set_xlim(max(0, s.min() - 0.1), min(1.0, s.max() + 0.06) if s.max() <= 1 else s.max() * 1.15)
    ax.set_title(title); fig.tight_layout(); return fig


def roc_plot(rocs, table):
    fig, ax = _fig(6.5, 5)
    auc = table.set_index("Model")["ROC-AUC (test)"]
    for i, (name, (fpr, tpr)) in enumerate(sorted(rocs.items(), key=lambda kv: -auc[kv[0]])):
        ax.plot(fpr, tpr, color=CAT[i], lw=2 if i == 0 else 1.4, label=f"{name} (AUC {auc[name]:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, lw=1)
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate (recall)")
    ax.legend(loc="lower right"); ax.set_title("ROC curves - test set"); fig.tight_layout(); return fig


def confusion(cm, title):
    fig, ax = _fig(4.6, 3.9)
    ax.imshow(cm, cmap=SEQMAP); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm.sum():.0%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else INK, fontsize=11)
    ax.set_xticks([0, 1], ["Pred: not", "Pred: interested"]); ax.set_yticks([0, 1], ["Actual: not", "Actual: interested"])
    ax.set_title(title); fig.tight_layout(); return fig


def importance(imp, title, n=15):
    fig, ax = _fig(7.5, 4.8)
    s = imp.head(n)
    s.index = [i.replace("Q", "").replace("_", " ") for i in s.index]
    _hbar(ax, s, fmt="{:.3f}"); ax.set_title(title); fig.tight_layout(); return fig


def segment_chart(seg, interested):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    order = ["High", "Medium", "Low"]
    share = pd.Series(seg).value_counts(normalize=True).reindex(order)
    actual = pd.Series(interested).groupby(seg).mean().reindex(order)
    cols = [ORD5[4], ORD5[2], ORD5[0]]
    for ax, s, t in [(axes[0], share, "Share of respondents"), (axes[1], actual, "Actual interest rate inside segment")]:
        ax.bar(order, s.values, color=cols, width=0.6); ax.yaxis.set_major_formatter(pct); ax.grid(axis="x", visible=False)
        for x, v in enumerate(s.values): ax.text(x, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.set_title(t, fontsize=10.5)
    fig.suptitle("Conversion-probability segments (High ≥ 0.65, Medium 0.35–0.65, Low < 0.35)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(); return fig

# ------------------------------------------------------------------ regression
def actual_vs_pred(actual, pred, name):
    fig, ax = _fig(6, 5)
    ax.scatter(actual, pred, s=12, alpha=.5, color=BLUE, linewidths=0)
    lim = [30, max(actual.max(), pred.max()) * 1.3]
    ax.plot(lim, lim, color=INK2, lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.xaxis.set_major_formatter(rupee); ax.yaxis.set_major_formatter(rupee)
    ax.set_xlabel("Actual expected spend (Q28)"); ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs predicted spend - {name}"); fig.tight_layout(); return fig


def coef_chart(coef, n=8):
    s = pd.concat([coef.head(n), coef.tail(n)]).drop_duplicates()
    s.index = [i.replace("Q", "").replace("_", " ") for i in s.index]
    fig, ax = _fig(8, 5)
    ax.barh(s.index, s.values, color=[CAT[7] if v < 0 else BLUE for v in s.values], height=0.6)
    ax.axvline(0, color=INK2, lw=1); ax.grid(axis="y", visible=False)
    ax.set_xlabel("% change in expected spend per +1 SD (or vs base category)")
    ax.set_title("What drives willingness to spend? (Ridge, log target)"); fig.tight_layout(); return fig

# ------------------------------------------------------------------ clustering
def elbow_silhouette(ks, inertia, sil, k):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(ks, inertia, color=BLUE, marker="o", ms=6); axes[0].set_title("Elbow - within-cluster SS", fontsize=10.5)
    axes[1].plot(ks, sil, color=BLUE, marker="o", ms=6); axes[1].set_title("Silhouette score (higher = better separated)", fontsize=10.5)
    for ax in axes:
        ax.axvline(k, color=ORANGE, lw=1); ax.set_xlabel("k (number of clusters)")
    fig.tight_layout(); return fig


def pca_small_multiples(pcs, persona):
    names = sorted(set(persona))
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3.2), sharex=True, sharey=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        m = persona == n
        ax.scatter(pcs[~m, 0], pcs[~m, 1], s=4, color="#e6e5e0", linewidths=0)
        ax.scatter(pcs[m, 0], pcs[m, 1], s=6, color=BLUE, alpha=.7, linewidths=0)
        ax.set_title(f"{n}\n(n={m.sum()})", fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Customer personas on the first two principal components", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(); return fig


def persona_heatmap(prof):
    cols = ["age", "income_band", "monthly_spend", "freq", "digital", "brand_affinity", "deal_seeker", "sustainability", "interest_rate", "exp_spend"]
    z = (prof[cols] - prof[cols].mean()) / prof[cols].std()
    fig, ax = _fig(9, 3.4)
    ax.imshow(z.values, cmap=DIV, norm=TwoSlopeNorm(0, -2, 2), aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(cols)), [c.replace("_", " ") for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(prof)), prof.persona)
    for i in range(len(prof)):
        for j, c in enumerate(cols):
            v = prof[c].iloc[i]
            txt = f"{v:.0%}" if c == "interest_rate" else (f"₹{v:,.0f}" if c in ("monthly_spend", "exp_spend") else f"{v:.1f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=INK)
    ax.set_title("Persona profiles (colour = above / below average)"); fig.tight_layout(); return fig

# ------------------------------------------------------------------ association
def rules_scatter(rules, n_label=6):
    fig, ax = _fig(7.5, 4.6)
    sc = ax.scatter(rules.Support, rules.Confidence, s=rules.Lift * 45, c=rules.Lift, cmap=SEQMAP, edgecolor=SURFACE, linewidths=1.5)
    seen = set(); shown = 0
    for _, r in rules.iterrows():
        key = frozenset((r["If customer wants"] + " + " + r["→ they also want"]).split(" + "))
        if key in seen or shown >= n_label: continue
        seen.add(key); shown += 1
        ax.annotate(f"{r['If customer wants']} → {r['→ they also want']}", (r.Support, r.Confidence), fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points", color=INK2)
    fig.colorbar(sc, ax=ax, label="Lift"); ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Support (share of interested baskets)"); ax.set_ylabel("Confidence")
    ax.set_title("Brand association rules (bubble size & colour = lift)"); fig.tight_layout(); return fig


def coverage_curve(opt):
    fig, ax = _fig(7.5, 3.8)
    ax.plot(opt.Step, opt["Cumulative coverage of interested customers"], color=BLUE, marker="o", ms=7)
    for _, r in opt.iterrows():
        ax.annotate(f"+{r['Add brand']}", (r.Step, r["Cumulative coverage of interested customers"]),
                    xytext=(0, -16), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.axhline(0.8, color=ORANGE, lw=1); ax.text(1, 0.81, "80% coverage", color=INK2, fontsize=8, va="bottom")
    ax.yaxis.set_major_formatter(pct); ax.set_ylim(0, 1.02); ax.set_xlabel("Number of launch brands")
    ax.set_title("'Baby steps': how many brands reach most interested customers?"); fig.tight_layout(); return fig

# ------------------------------------------------------------------ text
def sentiment_by_intent(d):
    t = pd.crosstab(d.Q30_purchase_intent, d.sentiment, normalize="index").reindex(INTENT)[["Negative", "Neutral", "Positive"]]
    fig, ax = _fig(8, 3.6)
    left = np.zeros(len(t))
    for c in t.columns:
        ax.barh(t.index, t[c], left=left, color=SENT[c], label=c, height=0.6, edgecolor=SURFACE, linewidth=2)
        left += t[c].values
    ax.xaxis.set_major_formatter(pct); ax.grid(False); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Comment sentiment by stated purchase intent"); fig.tight_layout(); return fig


def topic_chart(topics):
    t = topics.drop("Other", errors="ignore").sort_values("mentions")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    axes[0].barh(t.index, t.mentions, color=BLUE, height=0.6); axes[0].set_title("Mentions", fontsize=10.5)
    axes[1].barh(t.index, t.pct_negative, color=CAT[7], height=0.6); axes[1].set_title("% of mentions that are negative", fontsize=10.5)
    axes[1].xaxis.set_major_formatter(pct)
    for ax in axes: ax.grid(axis="y", visible=False)
    fig.suptitle("Topics customers talk about (Q29)", x=0.02, ha="left", fontweight="bold"); fig.tight_layout(); return fig


def terms_chart(pos, neg):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, s, c, t in [(axes[0], pos.head(10), BLUE, "Most characteristic of positive comments"),
                        (axes[1], neg.head(10), CAT[7], "Most characteristic of negative comments")]:
        s = s.sort_values(); ax.barh(s.index, s.values, color=c, height=0.6); ax.grid(axis="y", visible=False)
        ax.set_title(t, fontsize=10.5); ax.set_xlabel("Mean TF-IDF weight")
    fig.tight_layout(); return fig

# ------------------------------------------------------------------ prescriptive
def cash_curves(sims, title="Cumulative cash position (after capex)"):
    fig, ax = _fig(8.5, 4.4)
    for i, (name, sim) in enumerate(sims.items()):
        ax.plot(sim.month, sim.cumulative_cash, color=CAT[i], label=name.split(" (")[0])
    ax.axhline(0, color=INK2, lw=1); ax.yaxis.set_major_formatter(rupee); ax.set_xlabel("Month")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3); ax.set_title(title); fig.tight_layout(); return fig


def profit_curves(sims):
    fig, ax = _fig(8.5, 4)
    for i, (name, sim) in enumerate(sims.items()):
        ax.plot(sim.month, sim.monthly_profit, color=CAT[i], label=name.split(" (")[0])
    ax.axhline(0, color=INK2, lw=1); ax.yaxis.set_major_formatter(rupee); ax.set_xlabel("Month")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3); ax.set_title("Monthly operating profit (break-even = crossing zero)"); fig.tight_layout(); return fig


def mc_hist(mcs):
    fig, axes = plt.subplots(1, len(mcs), figsize=(4.2 * len(mcs), 3.6), sharey=True, sharex=True)
    allv = np.concatenate([m["runs"].final_cash.values for m in mcs.values()])
    bins = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 45)
    for ax, (name, mc) in zip(np.atleast_1d(axes), mcs.items()):
        v = mc["runs"].final_cash.clip(bins[0], bins[-1])
        ax.hist(v[v < 0], bins=bins, color=CAT[7], alpha=.85); ax.hist(v[v >= 0], bins=bins, color=BLUE, alpha=.85)
        ax.axvline(0, color=INK2, lw=1); ax.xaxis.set_major_formatter(rupee)
        ax.set_title(f"{name.split(' (')[0]}\nP(loss) {(v < 0).mean():.0%}", fontsize=9.5)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Monte-Carlo: cash position at month 48 (1,000 simulated futures)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(); return fig


def tornado(sens, title):
    fig, ax = _fig(8, 3.8)
    y = np.arange(len(sens))
    ax.barh(y, sens["Low (-20%)"], color=CAT[7], height=0.55, label="Driver −20%")
    ax.barh(y, sens["High (+20%)"], color=BLUE, height=0.55, label="Driver +20%")
    ax.set_yticks(y, sens["Driver (±20%)"]); ax.axvline(0, color=INK2, lw=1); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(rupee); ax.legend(loc="lower right"); ax.set_title(title)
    fig.tight_layout(); return fig


def scorecard_chart(card):
    fig, ax = _fig(8.5, 4.2)
    c = card.iloc[::-1]
    ax.barh(c.Metric, c["Score (0-100)"], color=[STATUS[r] for r in c.RAG], height=0.6)
    for y, (v, r) in enumerate(zip(c["Score (0-100)"], c.RAG)):
        lab = {"🟢": "Strong", "🟡": "Watch", "🔴": "Weak"}[r]
        ax.text(v + 1, y, f"{v:.0f} · {lab}", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 115); ax.grid(axis="y", visible=False); ax.axvline(65, color=GRID, lw=1); ax.axvline(40, color=GRID, lw=1)
    ax.set_title("Sustainability scorecard (0-100)"); fig.tight_layout(); return fig
