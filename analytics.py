"""
analytics.py - every algorithm used in the project, in one importable module.

Descriptive  : kpis(), profile tables (in the dashboard)
Diagnostic   : hypothesis_tests(), correlation matrix
Predictive   : classification (6 models), regression (6 models), clustering, sentiment
Prescriptive : association rules + launch-brand optimiser, segment actions,
               break-even & Monte-Carlo, sustainability scorecard, new-customer scorer
"""
import re
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error,
                             silhouette_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from config import *

SEED = RANDOM_SEED

# =========================================================== data
def load_clean(path="data/clean_survey.csv"):
    return pd.read_csv(path, parse_dates=["submitted_at"])

NUM_FEATURES = ["Q1_age", "income_ord", "edu_ord", "Q8_shop_freq_month", "log_monthly_spend",
                "Q13_avg_order_value", "Q14_social_hrs_week", "intl_buyer", *LIKERT_COLS.keys(),
                "n_plat", "n_cat", "n_brand", "core_zone"]
CAT_FEATURES = ["Q2_gender", "Q3_area", "Q4_occupation", "Q6_household_type", "Q10_channel", "Q27_attraction_driver"]


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)])


def kpis(df):
    return {
        "Respondents (clean)": len(df),
        "Interested (Probably/Definitely will)": f"{df.interested.mean():.1%}",
        "Median monthly fashion spend": f"₹{df.Q9_monthly_fashion_spend.median():,.0f}",
        "Median order value": f"₹{df.Q13_avg_order_value.median():,.0f}",
        "Median expected spend at us (interested)": f"₹{df.loc[df.interested == 1, 'Q28_expected_monthly_spend'].median():,.0f}",
        "Want multi-brand under one roof (Agree+)": f"{(df.Q19_multibrand_appeal >= 4).mean():.1%}",
    }

# =========================================================== DIAGNOSTIC: hypothesis tests
def hypothesis_tests(df, alpha=0.05):
    R = []
    def add(h, claim, test, stat, p, effect, implication):
        R.append({"H": h, "Hypothesis (H1 / alternative)": claim, "Test": test, "Statistic": round(stat, 3),
                  "p-value": p, "Effect / evidence": effect,
                  "Decision": "Reject H0 ✔ (supported)" if p < alpha else "Fail to reject H0 ✘",
                  "Business implication": implication})

    ct = pd.crosstab(df.Q5_household_income, df.interested)
    chi, p, _, _ = stats.chi2_contingency(ct)
    rates = df.groupby("Q5_household_income").interested.mean().reindex(INCOME_BANDS)
    add("H1", "Interest differs across household-income bands", "Chi-square", chi, p,
        f"Interest {rates.iloc[0]:.0%} (lowest band) → {rates.iloc[-1]:.0%} (highest band)",
        "Price the core range for ₹1L+ households; don't build the launch around the < ₹50K segment.")

    young = df.Q1_age <= 34
    ct = pd.crosstab(young, df.interested); chi, p, _, _ = stats.chi2_contingency(ct)
    add("H2", "Under-35s are more interested than 35+", "Chi-square (2×2)", chi, p,
        f"≤34: {df[young].interested.mean():.0%} vs 35+: {df[~young].interested.mean():.0%}",
        "Gen-Z / millennial-first brand voice, Instagram & creator marketing.")

    a, b = df.loc[df.interested == 1, "Q9_monthly_fashion_spend"], df.loc[df.interested == 0, "Q9_monthly_fashion_spend"]
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    add("H3", "Interested respondents already spend more on fashion", "Mann-Whitney U (one-sided)", u, p,
        f"Median ₹{a.median():,.0f} vs ₹{b.median():,.0f}",
        "Target existing heavy fashion spenders - we win share-of-wallet, not create new demand.")

    a, b = df.loc[df.interested == 1, "Q19_multibrand_appeal"], df.loc[df.interested == 0, "Q19_multibrand_appeal"]
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    add("H4", "'Many brands under one roof' appeal is higher among the interested", "Mann-Whitney U (one-sided)", u, p,
        f"Mean {a.mean():.2f} vs {b.mean():.2f} (1–5)", "The core multi-brand proposition is validated - lead with curation & breadth.")

    rho, p = stats.spearmanr(df.Q20_ai_reco_interest, df.intent_score)
    add("H5", "Interest in AI recommendations correlates with purchase intent", "Spearman ρ", rho, p,
        f"ρ = {rho:.2f}", "Personalisation is a conversion lever - invest in the recommender early.")

    ct = pd.crosstab(df.Q3_area, df.interested); chi, p, _, _ = stats.chi2_contingency(ct)
    r = df.groupby("Q3_area").interested.mean().sort_values(ascending=False)
    add("H6", "Interest differs by Mumbai zone", "Chi-square", chi, p,
        f"Highest {r.index[0]} {r.iloc[0]:.0%}, lowest {r.index[-1]} {r.iloc[-1]:.0%}",
        "First store location should sit in the top-converting zone.")

    rho, p = stats.spearmanr(df.Q18_discount_seeker, df.intent_score)
    add("H7", "Discount-seekers show lower purchase intent", "Spearman ρ", rho, p, f"ρ = {rho:.2f}",
        "Don't compete on discounts; use loyalty & exclusives instead of blanket sales.")

    sub = df.dropna(subset=["Q28_expected_monthly_spend"])
    rho, p = stats.spearmanr(sub.Q9_monthly_fashion_spend, sub.Q28_expected_monthly_spend)
    add("H8", "Expected spend at us rises with current fashion spend", "Spearman ρ", rho, p, f"ρ = {rho:.2f}",
        "Current spend is the strongest revenue predictor - collect it at sign-up.")

    groups = [g.Q23_sustainability_values.values for _, g in df.groupby("age_group", observed=True)]
    h, p = stats.kruskal(*groups)
    m = df.groupby("age_group", observed=True).Q23_sustainability_values.mean()
    add("H9", "Sustainability values differ by age group", "Kruskal-Wallis H", h, p,
        f"Means: " + ", ".join(f"{k} {v:.2f}" for k, v in m.items()),
        "A 'conscious edit' appeals to younger shoppers specifically." if p < alpha else
        "Sustainability interest is similar at every age (~40% agree) - position the 'conscious edit' for all ages, not as a youth niche.")

    ct = pd.crosstab(df.Q10_channel, df.interested); chi, p, _, _ = stats.chi2_contingency(ct)
    r = df.groupby("Q10_channel").interested.mean()
    add("H10", "Interest differs between online-first and offline-first shoppers", "Chi-square", chi, p,
        ", ".join(f"{k}: {v:.0%}" for k, v in r.items()),
        "Omnichannel matters - app-first acquisition, store for experience.")

    a = df.loc[df.Q2_gender == "Female", "Q28_expected_monthly_spend"].dropna()
    b = df.loc[df.Q2_gender == "Male", "Q28_expected_monthly_spend"].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    add("H11", "Expected spend differs between women and men", "Mann-Whitney U (two-sided)", u, p,
        f"Median ₹{a.median():,.0f} (F) vs ₹{b.median():,.0f} (M)",
        "Women expect to spend somewhat more - lead the launch assortment with women's apparel, bags & footwear, but keep menswear (gap is modest)." if p < alpha else
        "No real gender gap - plan a gender-balanced assortment.")
    out = pd.DataFrame(R)
    out["p-value"] = out["p-value"].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
    return out

# =========================================================== PREDICTIVE: classification
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=25),
    "Naive Bayes": GaussianNB(),
}


def run_classification(df):
    X, y = df[NUM_FEATURES + CAT_FEATURES], df.interested
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted, rocs = [], {}, {}
    for name, model in CLASSIFIERS.items():
        pipe = Pipeline([("prep", preprocessor()), ("model", clone(model))]).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
        cv_auc = cross_val_score(Pipeline([("prep", preprocessor()), ("model", clone(model))]), X, y, cv=cv, scoring="roc_auc")
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
                     "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred), "ROC-AUC (test)": roc_auc_score(yte, p),
                     "ROC-AUC (5-fold CV)": cv_auc.mean(), "CV std": cv_auc.std()})
        fitted[name] = pipe; rocs[name] = roc_curve(yte, p)[:2]
    table = pd.DataFrame(rows).sort_values("ROC-AUC (5-fold CV)", ascending=False).reset_index(drop=True)
    best = table.Model.iloc[0]
    best_pipe = fitted[best]
    cm = confusion_matrix(yte, best_pipe.predict(Xte))
    # out-of-fold probabilities for EVERY respondent (honest, not in-sample)
    oof = cross_val_predict(Pipeline([("prep", preprocessor()), ("model", clone(CLASSIFIERS[best]))]), X, y, cv=cv, method="predict_proba")[:, 1]
    # final model on all data for scoring new customers
    final = Pipeline([("prep", preprocessor()), ("model", clone(CLASSIFIERS[best]))]).fit(X, y)
    # interpretability: RF importances + logistic odds ratios
    rf = fitted["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
    lr = fitted["Logistic Regression"]
    odds = pd.Series(np.exp(lr.named_steps["model"].coef_[0]), index=names).sort_values(ascending=False)
    return {"table": table, "best": best, "model": final, "cm": cm, "rocs": rocs, "oof_prob": oof,
            "importance": imp, "odds": odds, "y_test": yte}


def conversion_segment(p):
    return np.where(p >= 0.65, "High", np.where(p >= 0.35, "Medium", "Low"))

# =========================================================== PREDICTIVE: regression (willingness to spend)
REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=5),
    "Lasso": Lasso(alpha=0.005, max_iter=5000),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=15, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED),
}
REG_NUM = NUM_FEATURES + ["intent_score"]


def reg_prep():
    return ColumnTransformer([("num", StandardScaler(), REG_NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)])


def run_regression(df):
    d = df.dropna(subset=["Q28_expected_monthly_spend"])
    X, y = d[REG_NUM + CAT_FEATURES], np.log1p(d.Q28_expected_monthly_spend)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    rows, fitted, preds = [], {}, {}
    for name, model in REGRESSORS.items():
        pipe = Pipeline([("prep", reg_prep()), ("model", clone(model))]).fit(Xtr, ytr)
        pl = pipe.predict(Xte); pr, ar = np.expm1(pl), np.expm1(yte)
        cv = cross_val_score(Pipeline([("prep", reg_prep()), ("model", clone(model))]), X, y, cv=5, scoring="r2")
        rows.append({"Model": name, "R² (log scale, test)": r2_score(yte, pl), "R² (5-fold CV)": cv.mean(),
                     "R² (₹ scale)": r2_score(ar, pr), "MAE (₹)": mean_absolute_error(ar, pr),
                     "RMSE (₹)": np.sqrt(mean_squared_error(ar, pr))})
        fitted[name] = pipe; preds[name] = (ar.values, pr)
    table = pd.DataFrame(rows).sort_values("R² (5-fold CV)", ascending=False).reset_index(drop=True)
    best = table.Model.iloc[0]
    final = Pipeline([("prep", reg_prep()), ("model", clone(REGRESSORS[best]))]).fit(X, y)
    lin = fitted["Ridge"]
    names = [n.split("__", 1)[1] for n in lin.named_steps["prep"].get_feature_names_out()]
    # coefficient on log target ≈ % change in spend per 1 SD of feature
    coef = pd.Series((np.exp(lin.named_steps["model"].coef_) - 1) * 100, index=names).sort_values()
    return {"table": table, "best": best, "model": final, "preds": preds, "coef_pct": coef}

# =========================================================== PREDICTIVE: clustering (personas)
CLUSTER_FEATURES = ["Q1_age", "income_ord", "log_monthly_spend", "Q8_shop_freq_month", "digital_index",
                    "brand_affinity_index", "deal_seeker_index", "Q23_sustainability_values"]


def run_clustering(df, k=None):
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    ks = range(2, 9)
    inertia, sil = [], []
    for kk in ks:
        km = KMeans(kk, n_init=10, random_state=SEED).fit(X)
        inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_, sample_size=1500, random_state=SEED))
    if k is None:
        k = 4  # elbow region; also the most actionable number of personas for marketing
    km = KMeans(k, n_init=20, random_state=SEED).fit(X)
    labels = km.labels_
    prof = df.assign(cluster=labels).groupby("cluster").agg(
        size=("resp_id", "count"), age=("Q1_age", "mean"), income_band=("income_ord", "mean"),
        monthly_spend=("Q9_monthly_fashion_spend", "median"), freq=("Q8_shop_freq_month", "mean"),
        digital=("digital_index", "mean"), brand_affinity=("brand_affinity_index", "mean"),
        deal_seeker=("deal_seeker_index", "mean"), sustainability=("Q23_sustainability_values", "mean"),
        interest_rate=("interested", "mean"), exp_spend=("Q28_expected_monthly_spend", "median"))
    names = name_clusters(prof)
    prof.insert(0, "persona", prof.index.map(names))
    pcs = PCA(2, random_state=SEED).fit_transform(X)
    return {"labels": labels, "persona": pd.Series(labels).map(names).values, "profile": prof,
            "ks": list(ks), "inertia": inertia, "silhouette": sil, "pca": pcs, "k": k}


def name_clusters(prof):
    """Rule-based naming so personas stay readable even if numbers shift."""
    names, used = {}, set()
    order = prof.sort_values("monthly_spend", ascending=False).index
    for c in order:
        r = prof.loc[c]
        if r.monthly_spend == prof.monthly_spend.max() and "Premium Trendsetters" not in used:
            n = "Premium Trendsetters"
        elif r.deal_seeker == prof.deal_seeker.max() and "Deal Hunters" not in used:
            n = "Deal Hunters"
        elif r.age == prof.age.max() and "Conservative Traditionalists" not in used:
            n = "Conservative Traditionalists"
        elif r.digital == prof.digital.max() and "Digital Explorers" not in used:
            n = "Digital Explorers"
        else:
            n = "Mainstream Pragmatists" if "Mainstream Pragmatists" not in used else f"Segment {c}"
        used.add(n); names[c] = n
    return names

# =========================================================== PRESCRIPTIVE: association rules (Apriori)
def apriori(baskets, min_support=0.05, max_len=3):
    """Pure-python Apriori (mlxtend not required). baskets: list[set]."""
    n = len(baskets)
    items = sorted({i for b in baskets for i in b})
    support = {}
    current = [frozenset([i]) for i in items]
    for k in range(1, max_len + 1):
        counts = {c: sum(1 for b in baskets if c <= b) for c in current}
        freq = {c: v / n for c, v in counts.items() if v / n >= min_support}
        support.update(freq)
        keys = list(freq)
        current = list({a | b for a in keys for b in keys if len(a | b) == k + 1})
        if not current:
            break
    return support


def association_rules(df, col="Q26_brands_wanted", min_support=0.05, min_conf=0.4, only_interested=True):
    d = df[df.interested == 1] if only_interested else df
    baskets = [set(s.split(";")) for s in d[col].fillna("") if s]
    sup = apriori(baskets, min_support)
    rules = []
    for itemset, s in sup.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for ante in combinations(itemset, r):
                ante = frozenset(ante); cons = itemset - ante
                conf = s / sup[ante]; lift = conf / sup[cons]
                if conf >= min_conf:
                    rules.append({"If customer wants": " + ".join(sorted(ante)), "→ they also want": " + ".join(sorted(cons)),
                                  "Support": s, "Confidence": conf, "Lift": lift,
                                  "Leverage": s - sup[ante] * sup[cons]})
    rules = pd.DataFrame(rules)
    if len(rules):
        rules = rules.sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)
    single = pd.Series({next(iter(k)): v for k, v in sup.items() if len(k) == 1}).sort_values(ascending=False)
    return rules, single, len(baskets)


def launch_brand_optimizer(df, max_brands=8):
    """Greedy set cover: which brands, added one at a time, reach the most INTERESTED customers?
    This is the 'baby steps' answer - how many brands do we really need at launch?"""
    d = df[df.interested == 1]
    baskets = [set(s.split(";")) for s in d.Q26_brands_wanted.fillna("") if s]
    chosen, covered, out = [], set(), []
    for step in range(max_brands):
        best, gain = None, -1
        for b in BRANDS:
            if b in chosen:
                continue
            g = sum(1 for i, bk in enumerate(baskets) if i not in covered and b in bk)
            if g > gain:
                best, gain = b, g
        chosen.append(best)
        covered |= {i for i, bk in enumerate(baskets) if best in bk}
        out.append({"Step": step + 1, "Add brand": best, "New customers reached": gain,
                    "Cumulative coverage of interested customers": len(covered) / len(baskets)})
    return pd.DataFrame(out)

# =========================================================== text analytics
POS = {"love", "great", "amazing", "awesome", "brilliant", "excited", "fair", "genuine", "authentic", "exactly",
       "finally", "game", "changer", "save", "easy", "fun", "quick", "variety", "personalised", "rewards",
       "mast", "accha", "support", "original", "help", "discover", "compare", "need"}
NEG = {"high", "fake", "inflated", "double", "scared", "never", "pain", "weeks", "same", "out", "tiring",
       "rude", "crowded", "messy", "don't", "trust", "bakwaas", "waste", "nahi", "copy", "wait", "takes", "late",
       "expensive", "nothing"}
NEGATORS = {"not", "no", "never", "don't", "dont", "nahi"}
TOPIC_WORDS = {
    "Price & discounts": ["price", "discount", "mrp", "cost", "expensive", "sale", "offer", "pricing", "fair"],
    "Authenticity & trust": ["fake", "authentic", "original", "copy", "trust", "genuine", "guarantee", "card"],
    "Delivery & returns": ["delivery", "return", "refund", "pickup", "minute", "quick", "days"],
    "Variety & sizes": ["brand", "variety", "size", "stock", "new", "discover", "same"],
    "Store experience": ["mall", "store", "trial", "staff", "traffic", "event", "styling", "fun", "try"],
    "Personalisation & loyalty": ["personalised", "suggestion", "loyalty", "reward", "app", "myntra"],
}


def lexicon_sentiment(text):
    if not isinstance(text, str):
        return np.nan
    toks = re.findall(r"[a-z']+", text.lower())
    score = 0
    for i, t in enumerate(toks):
        s = (t in POS) - (t in NEG)
        if s and i > 0 and toks[i - 1] in NEGATORS and t not in NEGATORS:
            s = -s
        score += s
    # last clause carries the customer's view of OUR idea - give it more weight
    last = re.split(r"[.\-!]", text.lower())
    last = [x for x in last if x.strip()]
    if len(last) > 1:
        lt = re.findall(r"[a-z']+", last[-1])
        score += 0.5 * sum((t in POS) - (t in NEG) for t in lt)
    return score / np.sqrt(max(len(toks), 1))


def detect_topics(text):
    if not isinstance(text, str):
        return []
    t = text.lower()
    return [k for k, ws in TOPIC_WORDS.items() if any(w in t for w in ws)] or ["Other"]


def run_text_analytics(df, truth=None):
    d = df[df.Q29_feedback_text.notna()].copy()
    d["sent_score"] = d.Q29_feedback_text.apply(lexicon_sentiment)
    d["sentiment"] = np.where(d.sent_score > 0.15, "Positive", np.where(d.sent_score < -0.15, "Negative", "Neutral"))
    d["topics"] = d.Q29_feedback_text.apply(detect_topics)
    out = {"data": d}
    topic_rows = d.explode("topics").groupby("topics").agg(mentions=("resp_id", "count"),
                                                             avg_sentiment=("sent_score", "mean"),
                                                             pct_negative=("sentiment", lambda s: (s == "Negative").mean()))
    out["topics"] = topic_rows.sort_values("mentions", ascending=False)
    # top distinguishing words: TF-IDF on positive vs negative
    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5)
    M = tf.fit_transform(d.Q29_feedback_text)
    vocab = np.array(tf.get_feature_names_out())
    pos_mean = np.asarray(M[(d.sentiment == "Positive").values].mean(axis=0)).ravel()
    neg_mean = np.asarray(M[(d.sentiment == "Negative").values].mean(axis=0)).ravel()
    out["pos_terms"] = pd.Series(pos_mean, vocab).sort_values(ascending=False).head(15)
    out["neg_terms"] = pd.Series(neg_mean, vocab).sort_values(ascending=False).head(15)
    if truth is not None:
        m = d.merge(truth[["resp_id", "true_sentiment"]], on="resp_id")
        out["lexicon_accuracy"] = (m.sentiment == m.true_sentiment).mean()
        out["lexicon_cm"] = pd.crosstab(m.true_sentiment, m.sentiment)
        # supervised alternative: pretend we hand-labelled 300 comments, train TF-IDF + logistic on them
        lab = m.sample(300, random_state=SEED); rest = m.drop(lab.index)
        tf2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        clf = LogisticRegression(max_iter=2000, C=3).fit(tf2.fit_transform(lab.Q29_feedback_text), lab.true_sentiment)
        pred = clf.predict(tf2.transform(rest.Q29_feedback_text))
        out["supervised_accuracy"] = (pred == rest.true_sentiment).mean()
    return out

# =========================================================== break-even & Monte Carlo
_LEAN = dict(
    store_sqft=1500, rent_per_sqft=350, cam_per_sqft=50, staff_count=7, staff_cost=30000, utilities=80000,
    tech_monthly=400000, brand_marketing=400000, admin=400000, dark_store=150000, misc=80000,
    fitout_per_sqft=3000, tech_build=5000000, deposit_months=6, opening_inventory=4000000, launch_marketing=3000000,
    gross_margin=0.30, payment_fee=0.018, loyalty_cost=0.02, online_share=0.65, fulfil_cost=150,
    return_rate=0.20, reverse_cost=130, cac=1200, acq_budget=600000, organic_new=600,
    monthly_churn=0.07, realisation=0.35, n_brands=8)
_FLAG = dict(
    store_sqft=4000, rent_per_sqft=350, cam_per_sqft=50, staff_count=16, staff_cost=32000, utilities=180000,
    tech_monthly=500000, brand_marketing=800000, admin=600000, dark_store=250000, misc=150000,
    fitout_per_sqft=3000, tech_build=8000000, deposit_months=6, opening_inventory=15000000, launch_marketing=6000000,
    gross_margin=0.34, payment_fee=0.018, loyalty_cost=0.02, online_share=0.5, fulfil_cost=150,
    return_rate=0.20, reverse_cost=130, cac=1200, acq_budget=1200000, organic_new=1500,
    monthly_churn=0.07, realisation=0.35, n_brands=15)
SCENARIOS = {
    "A. Lean pilot only (1,500 sq ft, 8 brands)": {"p": _LEAN},
    "B. Full flagship from day 1 (4,000 sq ft, 15 brands)": {"p": _FLAG},
    "C. Staged: lean pilot → flagship in month 13 (recommended)": {"p": _LEAN, "p2": _FLAG, "switch": 13},
}
ASSUMPTION_NOTES = {
    "rent_per_sqft": "₹/sq ft/month. Mumbai mall average ≈ ₹438 (2026); premium malls up to ₹777. Mid-tier suburban mall assumed.",
    "gross_margin": "Retailer margin after brand cost. Sale-or-return (pilot) ≈ 30%; partly outright buying (flagship) ≈ 34%.",
    "realisation": "Share of STATED spend that actually happens. Surveys overstate - 35% is deliberately conservative.",
    "cac": "Paid cost to acquire one new buying customer (ads + first-order offer).",
    "monthly_churn": "Share of active customers lost each month (7% ≈ 14-month average life).",
    "organic_new": "New customers / month from mall walk-ins, word of mouth, SEO (ramps over year 1).",
    "online_share": "Share of orders delivered (pay fulfilment + returns cost).",
}


def _econ(p, aov):
    rent = p["store_sqft"] * (p["rent_per_sqft"] + p["cam_per_sqft"])
    fixed = (rent + p["staff_count"] * p["staff_cost"] + p["utilities"] + p["tech_monthly"] + p["brand_marketing"]
             + p["admin"] + p["dark_store"] + p["misc"])
    capex = (p["store_sqft"] * p["fitout_per_sqft"] + p["tech_build"] + p["deposit_months"] * rent
             + p["opening_inventory"] + p["launch_marketing"])
    contrib = aov * (p["gross_margin"] - p["payment_fee"] - p["loyalty_cost"]) \
        - p["online_share"] * (p["fulfil_cost"] + p["return_rate"] * p["reverse_cost"])
    return rent, fixed, capex, contrib


def survey_inputs(df):
    i = df[df.interested == 1]
    return {"aov": float(i.Q13_avg_order_value.median()),
            "stated_monthly_spend": float(i.Q28_expected_monthly_spend.median()),
            "interest_rate": float(df.interested.mean())}


def break_even(p, aov, stated_spend, months=48, p2=None, switch=None):
    """Static break-even (for the final-stage set-up) + month-by-month cash simulation.
    If p2/switch are given, the business runs on p until `switch`, then scales up to p2."""
    final = p2 if p2 is not None else p
    rent, fixed, capex, contrib = _econ(final, aov)
    _, fixed1, capex1, _ = _econ(p, aov)
    opc = stated_spend * final["realisation"] / aov           # realised orders / active customer / month
    be_orders = fixed / contrib if contrib > 0 else np.inf
    active, cum, rows, be_month, payback = 0.0, -capex1, [], None, None
    total_capex = capex1
    for m in range(1, months + 1):
        q = p2 if (p2 is not None and m >= switch) else p
        _, f, _, c = _econ(q, aov)
        if p2 is not None and m == switch:                    # scale-up investment (reuse pilot tech & part of stock)
            extra = (p2["store_sqft"] * p2["fitout_per_sqft"] + p2["deposit_months"] * _econ(p2, aov)[0]
                     + (p2["tech_build"] - p["tech_build"]) + (p2["opening_inventory"] - p["opening_inventory"])
                     + 0.5 * p2["launch_marketing"])
            cum -= extra; total_capex += extra
        ramp = min(1.0, 0.4 + 0.6 * m / 12)
        new = q["acq_budget"] / q["cac"] + q["organic_new"] * ramp
        active = active * (1 - q["monthly_churn"]) + new
        orders = active * stated_spend * q["realisation"] / aov
        profit = orders * c - f - q["acq_budget"]
        cum += profit
        if be_month is None and profit >= 0 and (p2 is None or m >= switch): be_month = m
        if payback is None and cum >= 0: payback = m
        stage = ("Flagship" if m >= switch else "Pilot") if p2 is not None else ("Pilot" if p["store_sqft"] < 3000 else "Flagship")
        rows.append({"month": m, "stage": stage,
                     "new_customers": new, "active_customers": active, "orders": orders,
                     "revenue": orders * aov, "monthly_profit": profit, "cumulative_cash": cum})
    sim = pd.DataFrame(rows)
    ltv = opc * contrib / final["monthly_churn"]
    return {"fixed_monthly": fixed, "capex": total_capex, "contrib_per_order": contrib, "cm_ratio": contrib / aov,
            "orders_per_customer": opc, "be_orders": be_orders, "be_revenue": be_orders * aov,
            "be_customers": be_orders / opc if opc > 0 else np.inf, "be_month": be_month, "payback_month": payback,
            "peak_cash_need": -sim.cumulative_cash.min(), "ltv": ltv, "ltv_cac": ltv / final["cac"], "sim": sim}


def run_scenario(name, aov, stated_spend, overrides=None, months=48):
    sc = SCENARIOS[name]
    p = dict(sc["p"], **(overrides or {}))
    p2 = dict(sc["p2"], **(overrides or {})) if "p2" in sc else None
    return break_even(p, aov, stated_spend, months, p2, sc.get("switch"))


GATE = {"min_realisation": 0.30, "min_gross_margin": 0.28}   # month-12 go / no-go rule for the staged plan


def monte_carlo(name, aov, stated_spend, n=1000, months=48, seed=SEED):
    """Shock the uncertain assumptions together -> PROBABILITY OF FAILURE.
    For the staged plan the pilot MEASURES realisation and margin; if they miss the GATE at month 12
    we do not build the flagship and exit (deposits recovered) - this is the value of 'baby steps'."""
    r = np.random.default_rng(seed)
    sc = SCENARIOS[name]
    res = []
    for _ in range(n):
        gm_shift = r.normal(0, 0.03); cac_m = r.lognormal(0, 0.3); churn = r.normal(0, 0.02)
        real = r.normal(0, 0.12); rent_m = r.uniform(0.85, 1.3); rr = r.normal(0, 0.05); a = aov * r.normal(1, 0.1)
        def shock(p):
            q = dict(p)
            q["gross_margin"] = p["gross_margin"] + gm_shift; q["cac"] = p["cac"] * cac_m
            q["monthly_churn"] = float(np.clip(p["monthly_churn"] + churn, 0.02, 0.2))
            q["realisation"] = float(np.clip(p["realisation"] + real, 0.1, 1.0))
            q["rent_per_sqft"] = p["rent_per_sqft"] * rent_m
            q["return_rate"] = float(np.clip(p["return_rate"] + rr, 0.05, 0.4))
            return q
        p = shock(sc["p"]); scaled = None
        if "p2" in sc:
            go = p["realisation"] >= GATE["min_realisation"] and p["gross_margin"] >= GATE["min_gross_margin"]
            if go:
                o = break_even(p, a, stated_spend, months, shock(sc["p2"]), sc["switch"]); scaled = True
            else:   # exit after the pilot: cash at month 12 + deposit refund
                o = break_even(p, a, stated_spend, 12); scaled = False
                o["sim"].loc[o["sim"].index[-1], "cumulative_cash"] += p["deposit_months"] * _econ(p, a)[0]
        else:
            o = break_even(p, a, stated_spend, months)
        res.append({"realisation": p["realisation"], "scaled_up": scaled,
                    "be_month": o["be_month"] or np.nan, "payback": o["payback_month"] or np.nan,
                    "final_cash": o["sim"].cumulative_cash.iloc[-1], "peak_cash": o["peak_cash_need"]})
    res = pd.DataFrame(res)
    out = {"runs": res, "p_profitable_24m": (res.be_month <= 24).mean(),
           "p_payback_48m": res.payback.notna().mean(), "median_be_month": res.be_month.median(),
           "expected_cash_48m": res.final_cash.mean(), "p10_cash_48m": res.final_cash.quantile(0.10),
           "p_loss_making_48m": (res.final_cash < 0).mean(), "peak_cash_p90": res.peak_cash.quantile(0.90),
           "p_big_loss": (res.final_cash < -50_000_000).mean()}
    if "p2" in sc:
        out["p_scale_up"] = res.scaled_up.mean()
    return out


def compare_scenarios(aov, stated_spend, n=1000):
    rows = []
    for k in SCENARIOS:
        be = run_scenario(k, aov, stated_spend); mc = monte_carlo(k, aov, stated_spend, n)
        rows.append({"Scenario": k, "Total capex (₹ Cr)": be["capex"] / 1e7, "Peak cash needed, base (₹ Cr)": be["peak_cash_need"] / 1e7,
                     "Break-even month (base)": be["be_month"], "Payback month (base)": be["payback_month"],
                     "P(monthly profit ≤ 24 m)": mc["p_profitable_24m"], "P(payback ≤ 48 m)": mc["p_payback_48m"],
                     "Expected cash @48m (₹ Cr)": mc["expected_cash_48m"] / 1e7,
                     "Bad case P10 cash (₹ Cr)": mc["p10_cash_48m"] / 1e7,
                     "P(loss > ₹5 Cr)": mc["p_big_loss"], "P(scale-up)": mc.get("p_scale_up", np.nan)})
    return pd.DataFrame(rows)


def sensitivity(name, aov, stated_spend, horizon=36):
    base = run_scenario(name, aov, stated_spend)["sim"].cumulative_cash.iloc[horizon - 1]
    tests = {"Average order value": "aov", "Gross margin": "gross_margin", "Rent per sq ft": "rent_per_sqft",
             "CAC": "cac", "Monthly churn": "monthly_churn", "Intent realisation": "realisation"}
    sc = SCENARIOS[name]; rows = []
    for label, k in tests.items():
        vals = []
        for f in (0.8, 1.2):
            if k == "aov":
                o = run_scenario(name, aov * f, stated_spend)
            else:
                ov = {k: sc["p"][k] * f}
                p = dict(sc["p"], **ov); p2 = dict(sc["p2"], **{k: sc["p2"][k] * f}) if "p2" in sc else None
                o = break_even(p, aov, stated_spend, 48, p2, sc.get("switch"))
            vals.append(o["sim"].cumulative_cash.iloc[horizon - 1] - base)
        rows.append({"Driver (±20%)": label, "Low (-20%)": vals[0], "High (+20%)": vals[1], "Swing": abs(vals[1] - vals[0])})
    return pd.DataFrame(rows).sort_values("Swing")

# =========================================================== sustainability scorecard
def sustainability_scorecard(df, be, mc, rules_single, cls_seg):
    interest = df.interested.mean()
    high_share = (cls_seg == "High").mean()
    deal_share = (df.loc[df.interested == 1, "Q18_discount_seeker"] >= 4).mean()
    top3 = rules_single.head(3).sum() / rules_single.sum()
    sust = (df.Q23_sustainability_values >= 4).mean()
    sust_hi = df.loc[df.Q23_sustainability_values >= 4, "Q28_expected_monthly_spend"].median()
    sust_lo = df.loc[df.Q23_sustainability_values <= 2, "Q28_expected_monthly_spend"].median()
    def score(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100)
    rows = [
        ("Financial", "LTV / CAC ratio", f"{be['ltv_cac']:.1f}×", score(be["ltv_cac"], 1, 5), "≥3× healthy; <1× destroys value"),
        ("Financial", "P(monthly profit within 24 months)", f"{mc['p_profitable_24m']:.0%}", mc["p_profitable_24m"] * 100, "Monte-Carlo, 1,000 runs"),
        ("Financial", "P(capex paid back within 48 months)", f"{mc['p_payback_48m']:.0%}", mc["p_payback_48m"] * 100, "Investor-grade if > 60%"),
        ("Market", "Stated interest rate", f"{interest:.0%}", score(interest, 0.2, 0.6), "Share answering Probably/Definitely will"),
        ("Market", "High-conversion segment size", f"{high_share:.0%}", score(high_share, 0.1, 0.4), "Model probability ≥ 0.65"),
        ("Market", "Discount dependence of interested base", f"{deal_share:.0%}", 100 - score(deal_share, 0.2, 0.6), "Lower is better for margin"),
        ("Operational", "Brand concentration (top-3 share of demand)", f"{top3:.0%}", 100 - score(top3, 0.25, 0.6), "Lower = less supplier risk"),
        ("Environmental/Social", "Customers valuing sustainability (Agree+)", f"{sust:.0%}", score(sust, 0.15, 0.5), "Demand for a 'conscious edit'"),
        ("Environmental/Social", "Spend premium of sustainability-minded", f"₹{sust_hi:,.0f} vs ₹{sust_lo:,.0f}",
         score(sust_hi / max(sust_lo, 1), 0.9, 1.3), "Do eco-minded shoppers spend more?"),
    ]
    t = pd.DataFrame(rows, columns=["Pillar", "Metric", "Value", "Score (0-100)", "Note"])
    t["Score (0-100)"] = t["Score (0-100)"].round(0)
    t["RAG"] = np.where(t["Score (0-100)"] >= 65, "🟢", np.where(t["Score (0-100)"] >= 40, "🟡", "🔴"))
    return t, t["Score (0-100)"].mean()

# =========================================================== connect the algorithm: score a NEW customer
def score_new_customer(answers, cls_model, reg_model, rules):
    row = pd.DataFrame([answers])
    p = float(cls_model.predict_proba(row[NUM_FEATURES + CAT_FEATURES])[:, 1][0])
    seg = str(conversion_segment(np.array([p]))[0])
    row["intent_score"] = 1 + 4 * p      # unknown at scoring time -> use model probability as proxy
    spend = float(np.expm1(reg_model.predict(row[REG_NUM + CAT_FEATURES]))[0])
    liked = set(answers.get("liked_brands", []))
    recs = []
    if len(rules) and liked:
        r = rules[rules["If customer wants"].apply(lambda s: set(s.split(" + ")) <= liked)]
        for _, x in r.iterrows():
            for b in x["→ they also want"].split(" + "):
                if b not in liked and b not in recs:
                    recs.append(b)
    action = {"High": "Onboard now: app-install offer, personal stylist call, early-access drop.",
              "Medium": "Nurture: first-order ₹300 credit + recommendations email/WhatsApp series.",
              "Low": "Low-cost touch only: Instagram retargeting; don't spend paid CAC."}[seg]
    return {"probability": p, "segment": seg, "predicted_spend": spend, "recommended_brands": recs[:4], "action": action}
