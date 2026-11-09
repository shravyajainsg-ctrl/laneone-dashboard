"""
STEP 2 - Data cleaning & transformation.

Every fix is logged to data/cleaning_log.csv so the dashboard can show
"what was wrong -> what we did -> how many rows" (the rationale for each step).

Output: data/clean_survey.csv   (analysis-ready, one row per genuine respondent)
"""
import re
import numpy as np
import pandas as pd
from config import *

raw = pd.read_csv("data/raw_survey_responses.csv")
df = raw.copy()
LOG = []

def log(step, column, rows, action, why):
    LOG.append({"step": step, "column": column, "rows_affected": int(rows), "action": action, "rationale": why})

n0 = len(df)

# ---------------------------------------------------------------- 1. duplicates
before = len(df)
df = df.drop_duplicates()
log("1 Duplicates", "all", before - len(df), "Dropped exact duplicate rows",
    "Same form submitted twice would double-count a customer.")
content_cols = [c for c in df.columns if c != "resp_id"]
before = len(df)
df = df.drop_duplicates(subset=content_cols, keep="first")
log("1 Duplicates", "all except resp_id", before - len(df), "Dropped identical answers with a new ID",
    "Identical 30 answers + timestamp = same person re-submitting.")

# ---------------------------------------------------------------- 2. timestamps
def parse_ts(s):
    for f in ["%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%b %d, %Y %I:%M %p"]:
        try:
            return pd.to_datetime(s, format=f)
        except (ValueError, TypeError):
            pass
    return pd.NaT
non_iso = (~df.submitted_at.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}")).sum()
df["submitted_at"] = df.submitted_at.apply(parse_ts)
log("2 Dates", "submitted_at", non_iso, "Parsed 3 date formats into one datetime",
    "Mixed DD/MM vs ISO formats break time-series and can swap day/month.")

# ---------------------------------------------------------------- 3. Likert text -> number (needed before bot check)
label_to_num = {v.lower(): k for k, v in LIKERT_LABELS.items()}
def likert_clean(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    m = re.match(r"^(\d+(\.\d+)?)", s)
    if m:
        x = float(m.group(1))
        return x if 1 <= x <= 5 else np.nan
    return label_to_num.get(s, np.nan)
for c in LIKERT_COLS:
    txt = df[c].apply(lambda v: isinstance(v, str) and not re.match(r"^\d+(\.0)?$", v.strip())).sum()
    oor = pd.to_numeric(df[c], errors="coerce").pipe(lambda s: ((s < 1) | (s > 5)).sum())
    df[c] = df[c].apply(likert_clean)
    log("3 Likert", c, txt + oor, "Text -> 1-5 code; values outside 1-5 set to missing",
        "Likert must be a 1-5 ordinal for correlations, clustering and models.")

# ---------------------------------------------------------------- 4. bots / speeders
straight = df[list(LIKERT_COLS)].std(axis=1, skipna=True) == 0
fast = df.completion_time_sec < 90
bots = fast & straight
before = len(df)
df = df[~(fast)].copy()
log("4 Bots", "completion_time_sec", before - len(df),
    f"Removed responses finished in < 90 sec ({int(bots.sum())} of them also straight-lined all 10 Likert items)",
    "A 30-question survey can't be read honestly in under 90 sec; these add random noise to every model.")

# ---------------------------------------------------------------- 5. categorical standardisation
gmap = {"female": "Female", "f": "Female", "woman": "Female", "male": "Male", "m": "Male", "man": "Male"}
bad = (~df.Q2_gender.isin(GENDERS)).sum()
df["Q2_gender"] = df.Q2_gender.apply(lambda v: v if v in GENDERS else gmap.get(str(v).strip().lower(), "Non-binary / Prefer not to say"))
log("5 Categories", "Q2_gender", bad, "Mapped F/female/Woman -> Female etc.", "One concept must have one label or group counts split.")

amap = {"western suburbs": "Western Suburbs", "w. suburbs": "Western Suburbs", "andheri": "Western Suburbs",
        "bandra": "Western Suburbs", "western suburb": "Western Suburbs", "sobo": "South Mumbai",
        "south mumbai": "South Mumbai", "colaba": "South Mumbai", "central": "Central Suburbs",
        "ghatkopar": "Central Suburbs", "chembur": "Central Suburbs", "thane west": "Thane", "thane": "Thane",
        "navi mumbai": "Navi Mumbai", "vashi": "Navi Mumbai", "navimumbai": "Navi Mumbai"}
bad = (~df.Q3_area.isin(AREAS)).sum()
df["Q3_area"] = df.Q3_area.apply(lambda v: v if v in AREAS else amap.get(str(v).strip().lower(), "Other MMR"))
log("5 Categories", "Q3_area", bad, "Mapped sub-localities (Andheri, Vashi, Colaba...) to 6 zones",
    "Store-location analysis needs consistent zones.")

imap = {"below 50k": "< ₹50K", "50k-1 lakh": "₹50K–1L", "1-2 lakh": "₹1L–2L", "2-4 lakh": "₹2L–4L",
        "4-8 lakh": "₹4L–8L", "8l+": "> ₹8L"}
bad = (df.Q5_household_income.notna() & ~df.Q5_household_income.isin(INCOME_BANDS)).sum()
df["Q5_household_income"] = df.Q5_household_income.apply(
    lambda v: v if (pd.isna(v) or v in INCOME_BANDS) else imap.get(str(v).strip().lower(), np.nan))
log("5 Categories", "Q5_household_income", bad, "Unified '1-2 Lakh' style text to standard bands", "Income must be one ordinal scale.")

def norm_channel(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    return {"online": "Mostly online", "offline": "Mostly offline", "both equally": "Both equally"}.get(
        s.replace("mostly ", ""), np.nan)
bad = (df.Q10_channel.notna() & ~df.Q10_channel.isin(CHANNELS)).sum()
df["Q10_channel"] = df.Q10_channel.apply(norm_channel)
log("5 Categories", "Q10_channel", bad, "Fixed lower-case / shortened channel labels", "Consistent labels.")

def norm_intent(v):
    s = str(v).strip().lower().replace("yes", "will")
    for opt in INTENT:
        if s == opt.lower():
            return opt
    return np.nan
bad = (~df.Q30_purchase_intent.isin(INTENT)).sum()
df["Q30_purchase_intent"] = df.Q30_purchase_intent.apply(norm_intent)
log("5 Categories", "Q30_purchase_intent", bad, "Fixed case/spacing/'yes' variants of target", "Target variable must be exact.")

# ---------------------------------------------------------------- 6. numeric parsing + outliers
def money(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float)): return float(v)
    s = str(v).lower().replace("₹", "").replace("rs.", "").replace("inr", "").replace(",", "").strip()
    mult = 1000 if s.endswith("k") else 1
    s = s.rstrip("k").strip()
    try:
        return float(s) * mult
    except ValueError:
        return np.nan

# age
txt = df.Q1_age.apply(lambda v: isinstance(v, str) and not re.match(r"^\d+(\.\d+)?$", v)).sum()
df["Q1_age"] = df.Q1_age.apply(lambda v: float(re.findall(r"\d+", str(v))[0]) if re.findall(r"\d+", str(v)) else np.nan)
impossible = ((df.Q1_age < 16) | (df.Q1_age > 80)).sum()
df.loc[(df.Q1_age < 16) | (df.Q1_age > 80), "Q1_age"] = np.nan
log("6 Outliers", "Q1_age", txt + impossible, "Extracted number from '25 yrs'; ages <16 or >80 set missing",
    "Ages like 0, 150, 999 are typing errors, not customers (domain rule, not statistics).")

for c in ["Q9_monthly_fashion_spend", "Q13_avg_order_value", "Q28_expected_monthly_spend"]:
    txt = df[c].apply(lambda v: isinstance(v, str) and not re.match(r"^-?\d+(\.\d+)?$", v)).sum()
    df[c] = df[c].apply(money)
    log("6 Money", c, txt, "Parsed '₹4,500' / '4.5k' / '4500 INR' to numbers", "Spend must be numeric for regression & break-even.")
    neg = (df[c] < 0).sum()
    df.loc[df[c] < 0, c] = np.nan
    if neg: log("6 Outliers", c, neg, "Negative amounts set missing", "Spend can't be negative.")

# spend typos: compare to income (domain-based outlier rule)
inc_mid = df.Q5_household_income.map(INCOME_MID)
cap = (inc_mid.fillna(150_000) * 0.5)
typo = df.Q9_monthly_fashion_spend > cap
fixed = typo & (df.Q9_monthly_fashion_spend / 100 <= cap)
df.loc[fixed, "Q9_monthly_fashion_spend"] = df.loc[fixed, "Q9_monthly_fashion_spend"] / 100
df.loc[typo & ~fixed, "Q9_monthly_fashion_spend"] = np.nan
log("6 Outliers", "Q9_monthly_fashion_spend", typo.sum(),
    f"Spend > 50% of household income: {int(fixed.sum())} corrected (÷100 'extra zeros'), {int((typo & ~fixed).sum())} set missing",
    "Fashion spend above half of household income is implausible. Genuine luxury shoppers (₹60K–1.5L with ₹8L+ income) are KEPT - they are real, valuable outliers.")

bad = (df.Q28_expected_monthly_spend > 150_000).sum()
df.loc[df.Q28_expected_monthly_spend > 150_000, "Q28_expected_monthly_spend"] = np.nan
log("6 Outliers", "Q28_expected_monthly_spend", bad, "Expected spend > ₹1.5L/month set missing",
    "₹4L–20L/month at one store is a typo; left missing (we never impute a model target).")

bad = (df.Q8_shop_freq_month > 31).sum()
df.loc[df.Q8_shop_freq_month > 31, "Q8_shop_freq_month"] = np.nan
log("6 Outliers", "Q8_shop_freq_month", bad, "> 31 shopping trips/month set missing", "More than daily is not credible.")
bad = (df.Q14_social_hrs_week > 112).sum()
df.loc[df.Q14_social_hrs_week > 112, "Q14_social_hrs_week"] = np.nan
log("6 Outliers", "Q14_social_hrs_week", bad, "> 112 hrs/week (16 hrs/day) set missing", "A week has 168 hours; >16/day is impossible.")

# ---------------------------------------------------------------- 7. text cleaning
junk = {"asdf", "...", "na", "-", "good", "nothing", "😂😂", "n/a"}
def clean_text(v):
    if pd.isna(v): return np.nan
    s = re.sub(r"\s+", " ", str(v)).strip().rstrip("!").strip()
    if s.lower() in junk or len(re.findall(r"[a-zA-Z]", s)) < 5:
        return np.nan
    return s[0].upper() + s[1:].lower() if s.isupper() else s
bad = df.Q29_feedback_text.notna().sum()
df["Q29_feedback_text"] = df.Q29_feedback_text.apply(clean_text)
bad = bad - df.Q29_feedback_text.notna().sum()
log("7 Text", "Q29_feedback_text", bad, "Removed junk ('asdf', 'na', emojis only); fixed ALL-CAPS",
    "Junk text adds no meaning and would distort sentiment scores.")

# ---------------------------------------------------------------- 8. missing-value imputation
n = df.Q5_household_income.isna().sum()
df["Q5_household_income"] = df.groupby("Q3_area").Q5_household_income.transform(
    lambda s: s.fillna(s.mode().iat[0]))
log("8 Impute", "Q5_household_income", n, "Mode within same area",
    "Neighbourhood is the best proxy for income in Mumbai.")
n = df.Q1_age.isna().sum()
df["Q1_age"] = df.groupby("Q4_occupation").Q1_age.transform(lambda s: s.fillna(s.median()))
log("8 Impute", "Q1_age", n, "Median within same occupation", "Students and retirees have very different ages.")
for c in ["Q9_monthly_fashion_spend", "Q13_avg_order_value"]:
    n = df[c].isna().sum()
    df[c] = df.groupby("Q5_household_income")[c].transform(lambda s: s.fillna(s.median()))
    log("8 Impute", c, n, "Median within same income band", "Median is robust to the remaining skew; band keeps it realistic.")
for c in ["Q8_shop_freq_month", "Q14_social_hrs_week"]:
    n = df[c].isna().sum(); df[c] = df[c].fillna(df[c].median())
    log("8 Impute", c, n, "Overall median", "Few missing; median is robust.")
for c in LIKERT_COLS:
    n = df[c].isna().sum(); df[c] = df[c].fillna(df[c].median()).round().astype(int)
    log("8 Impute", c, n, "Column median (neutral-ish)", "Keeps the 1-5 scale; < 5% missing.")
n = df.Q10_channel.isna().sum(); df["Q10_channel"] = df.Q10_channel.fillna(df.Q10_channel.mode().iat[0])
log("8 Impute", "Q10_channel", n, "Mode", "Few missing.")
n = df.Q30_purchase_intent.isna().sum()
df = df[df.Q30_purchase_intent.notna()]
log("8 Impute", "Q30_purchase_intent", n, "Rows without target dropped", "Can't train or evaluate on an unknown label.")

# ---------------------------------------------------------------- 9. TRANSFORMATION / feature engineering
df["interested"] = df.Q30_purchase_intent.isin(["Probably will", "Definitely will"]).astype(int)
df["intent_score"] = df.Q30_purchase_intent.map({v: i + 1 for i, v in enumerate(INTENT)})
df["income_ord"] = df.Q5_household_income.map({v: i for i, v in enumerate(INCOME_BANDS)})
df["income_mid"] = df.Q5_household_income.map(INCOME_MID)
df["edu_ord"] = df.Q7_education.map({v: i for i, v in enumerate(EDUCATION)})
df["age_group"] = pd.cut(df.Q1_age, [15, 24, 34, 44, 54, 81], labels=["16-24", "25-34", "35-44", "45-54", "55+"])
df["log_monthly_spend"] = np.log1p(df.Q9_monthly_fashion_spend)
df["spend_share_of_income_pct"] = (df.Q9_monthly_fashion_spend / df.income_mid * 100).round(2)
df["intl_buyer"] = (df.Q15_bought_intl_brand == "Yes").astype(int)
df["digital_index"] = df[["Q20_ai_reco_interest", "Q21_phygital_interest", "Q24_data_sharing_comfort", "Q25_fast_delivery_need"]].mean(axis=1).round(2)
df["brand_affinity_index"] = df[["Q16_brand_preference", "Q17_authenticity_matters", "Q19_multibrand_appeal"]].mean(axis=1).round(2)
df["deal_seeker_index"] = df[["Q18_discount_seeker", "Q22_loyalty_influence"]].mean(axis=1).round(2)
df["core_zone"] = df.Q3_area.isin(["South Mumbai", "Western Suburbs"]).astype(int)
for col, opts in MULTI_COLS.items():
    prefix = {"Q11_platforms_used": "plat_", "Q12_categories": "cat_", "Q26_brands_wanted": "brand_"}[col]
    for o in opts:
        df[prefix + re.sub(r"[^a-z0-9]+", "_", o.lower()).strip("_")] = df[col].fillna("").str.split(";").apply(lambda L: int(o in L))
    df["n_" + prefix.rstrip("_")] = df[col].fillna("").str.split(";").apply(lambda L: len([x for x in L if x]))
log("9 Transform", "derived", len(df),
    "Created target (interested 0/1), ordinal encodings, log spend, indices, one-hot for multi-selects",
    "Models need numbers; log removes spend skew; indices summarise related Likert items.")

df = df.reset_index(drop=True)
df.to_csv("data/clean_survey.csv", index=False)
pd.DataFrame(LOG).to_csv("data/cleaning_log.csv", index=False)
print(f"Raw rows {n0} -> clean rows {len(df)}; columns {df.shape[1]}")
print(f"Interested share: {df.interested.mean():.3f}")
print(pd.DataFrame(LOG)[["step", "column", "rows_affected"]].to_string())
print("Remaining NaN:", df[[c for c in df.columns if c.startswith('Q')]].isna().sum()[lambda s: s > 0].to_dict())
