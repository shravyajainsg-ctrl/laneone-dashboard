"""
STEP 1 - Synthetic survey data with intentional noise and outliers.

How it works
------------
1. Each simulated Mumbai respondent gets hidden "latent traits":
   income, fashion affinity, price sensitivity, digital savviness,
   sustainability orientation and a style persona.
2. Every survey answer is generated FROM those traits (plus random noise),
   so the data contains realistic, learnable relationships - e.g. richer,
   younger, digitally savvy fashion lovers are more likely to convert.
3. Intentional data-quality problems are then injected on purpose
   (missing values, outliers, typos, mixed formats, duplicates, bots,
   label noise) so the cleaning step has real work to do.

Output: data/raw_survey_responses.csv  (messy, what a Google Form export looks like)
        data/_ground_truth_hidden.csv  (hidden truth - used ONLY to validate text analytics)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import *

rng = np.random.default_rng(RANDOM_SEED)
N = N_RESPONDENTS


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def likert(latent, noise=0.8, shift=0.0):
    """Turn a continuous latent score into a 1-5 Likert answer."""
    z = latent + shift + rng.normal(0, noise, len(latent))
    return np.digitize(z, [-1.5, -0.5, 0.5, 1.5]) + 1


# ------------------------------------------------------------------ 1. demographics
age = np.clip(rng.normal(30, 9, N), 16, 68).round()
gender = rng.choice(GENDERS, N, p=[0.52, 0.45, 0.03])
area = rng.choice(AREAS, N, p=[0.12, 0.35, 0.20, 0.15, 0.13, 0.05])

occupation = np.empty(N, dtype=object)
for i, a in enumerate(age):
    if a < 23:
        p = [0.70, 0.15, 0.03, 0.10, 0.02, 0.00]
    elif a < 55:
        p = [0.02, 0.55, 0.20, 0.10, 0.13, 0.00]
    else:
        p = [0.00, 0.30, 0.25, 0.03, 0.17, 0.25]
    occupation[i] = rng.choice(OCCUPATIONS, p=p)

area_income = {"South Mumbai": 0.9, "Western Suburbs": 0.35, "Central Suburbs": -0.05,
               "Thane": -0.1, "Navi Mumbai": 0.0, "Other MMR": -0.5}
occ_income = {"Student": -0.2, "Salaried professional": 0.2, "Business owner / Self-employed": 0.5,
              "Freelancer / Creator": 0.0, "Homemaker": 0.0, "Retired": -0.2}
inc_z = (np.array([area_income[a] for a in area]) + np.array([occ_income[o] for o in occupation])
         + 0.015 * (np.clip(age, 22, 50) - 30) + rng.normal(0, 0.8, N))
income_band = np.array(INCOME_BANDS)[np.digitize(inc_z, [-1.0, -0.3, 0.4, 1.1, 1.8])]

household = np.where(age < 26, rng.choice(HOUSEHOLD, N, p=[0.6, 0.1, 0.05, 0.25]),
                     rng.choice(HOUSEHOLD, N, p=[0.15, 0.25, 0.42, 0.18]))
edu_idx = np.clip(np.round(1.2 + 0.4 * inc_z + rng.normal(0, 0.7, N)), 0, 3).astype(int)
education = np.array(EDUCATION)[edu_idx]

# ------------------------------------------------------------------ 2. latent traits
young = (age < 32).astype(float)
fashion = rng.normal(0, 1, N) + 0.3 * inc_z + 0.35 * young + 0.2 * (gender == "Female")
price_sens = rng.normal(0, 1, N) - 0.45 * inc_z + 0.2 * (occupation == "Student")
digital = rng.normal(0, 1, N) - 0.045 * (age - 30) + 0.15 * inc_z
sustain = rng.normal(0, 1, N) + 0.2 * inc_z + 0.25 * young

# style persona drives brand baskets (for association mining)
personas = ["Sporty", "Premium Classic", "Trend Seeker", "Accessory Glam", "Ethnic Fusion"]
pers = []
for i in range(N):
    w = np.array([1.2 + 0.6 * (gender[i] == "Male") + 0.5 * young[i],
                  0.6 + 0.6 * max(inc_z[i], 0) + 0.3 * (age[i] > 30),
                  1.0 + 0.8 * young[i] + 0.3 * (gender[i] == "Female"),
                  0.3 + 1.1 * (gender[i] == "Female"),
                  0.5 + 0.4 * (age[i] > 35) + 0.3 * (gender[i] == "Female")])
    pers.append(rng.choice(personas, p=w / w.sum()))
pers = np.array(pers)

# ------------------------------------------------------------------ 3. behaviour
freq = rng.poisson(np.exp(0.55 + 0.35 * fashion + 0.15 * digital - 0.1 * price_sens)).astype(float)
freq = np.clip(freq, 0, 20)
spend = np.exp(8.05 + 0.45 * inc_z + 0.35 * fashion - 0.1 * price_sens + rng.normal(0, 0.35, N))
aov = np.clip(spend / np.maximum(freq, 1) * rng.lognormal(0, 0.25, N), 400, 60_000)
social = np.clip(rng.gamma(2.0, 2.2, N) * np.exp(0.35 * digital) * (1.3 - 0.01 * age), 0, 60)
intl = rng.random(N) < sigmoid(-0.2 + 0.8 * inc_z + 0.6 * fashion)

channel = np.array(CHANNELS)[np.digitize(-digital + rng.normal(0, 0.6, N), [-0.4, 0.6])]

def multi(prob_matrix, options):
    picks = rng.random(prob_matrix.shape) < prob_matrix
    out = []
    for row in picks:
        chosen = [o for o, p in zip(options, row) if p]
        out.append(";".join(chosen) if chosen else options[int(rng.integers(len(options)))])
    return np.array(out, dtype=object)

plat_p = np.column_stack([
    sigmoid(0.6 + 0.8 * digital + 0.2 * young), sigmoid(0.0 + 0.6 * digital + 0.3 * price_sens),
    sigmoid(-0.9 + 0.5 * digital + 0.8 * (gender == "Female")), sigmoid(-1.4 + 0.4 * digital + 0.3 * inc_z),
    sigmoid(0.3 + 0.5 * digital + 0.3 * price_sens), sigmoid(-0.5 - 0.4 * digital + 0.2 * inc_z),
    sigmoid(-0.7 + 0.6 * inc_z + 0.4 * fashion), sigmoid(-1.3 + 0.7 * digital + 0.6 * young)])
platforms = multi(plat_p, PLATFORMS)

cat_p = np.column_stack([
    np.full(N, 0.85), sigmoid(0.2 + 0.3 * (pers == "Sporty")), sigmoid(-0.6 + 1.2 * (pers == "Accessory Glam")),
    sigmoid(-0.6 + 1.6 * (pers == "Sporty")), sigmoid(-1.5 + 0.5 * inc_z), sigmoid(-1.0 + 1.0 * (gender == "Female")),
    sigmoid(-2.2 + 2.0 * (household == "Family with kids"))])
categories = multi(cat_p, CATEGORIES)

# ------------------------------------------------------------------ 4. attitudes (Likert)
L = {
    "Q16_brand_preference": likert(0.7 * fashion + 0.4 * inc_z),
    "Q17_authenticity_matters": likert(0.5 * fashion + 0.3 * inc_z, shift=0.6),
    "Q18_discount_seeker": likert(0.9 * price_sens, shift=0.3),
    "Q19_multibrand_appeal": likert(0.55 * fashion + 0.45 * digital, shift=0.3),
    "Q20_ai_reco_interest": likert(0.75 * digital + 0.2 * fashion),
    "Q21_phygital_interest": likert(0.5 * digital + 0.4 * fashion),
    "Q22_loyalty_influence": likert(0.5 * price_sens + 0.3 * fashion, shift=0.3),
    "Q23_sustainability_values": likert(0.9 * sustain),
    "Q24_data_sharing_comfort": likert(0.6 * digital, shift=-0.3),
    "Q25_fast_delivery_need": likert(0.5 * digital + 0.2 * fashion, shift=0.2),
}

# ------------------------------------------------------------------ 5. target: purchase intent
area_bonus = {"South Mumbai": 0.35, "Western Suburbs": 0.35, "Central Suburbs": 0.0,
              "Thane": -0.05, "Navi Mumbai": -0.1, "Other MMR": -0.5}
intent_latent = (-0.85 + 0.55 * fashion + 0.30 * inc_z + 0.35 * digital - 0.30 * price_sens
                 + 0.30 * (L["Q19_multibrand_appeal"] - 3) + 0.22 * (L["Q20_ai_reco_interest"] - 3)
                 + 0.18 * (L["Q21_phygital_interest"] - 3) + 0.15 * (L["Q17_authenticity_matters"] - 3)
                 + np.array([area_bonus[a] for a in area]) - 0.6 * (age > 50)
                 + 0.25 * intl + rng.normal(0, 0.9, N))
intent = np.array(INTENT)[np.digitize(intent_latent, [-1.6, -0.6, 0.1, 1.1])]

# brands wanted - persona baskets (creates association rules)
B = {b: i for i, b in enumerate(BRANDS)}
base = np.full((N, len(BRANDS)), 0.06)
persona_brands = {
    "Sporty": {"Nike": .75, "Adidas": .65, "Puma": .5, "Skechers": .4, "Crocs": .3, "Levi's": .2},
    "Premium Classic": {"Tommy Hilfiger": .7, "Calvin Klein": .6, "Levi's": .5, "Nike": .25, "Steve Madden": .15},
    "Trend Seeker": {"Zara": .75, "H&M": .65, "Mango": .45, "Nike": .3, "Levi's": .3, "Crocs": .2},
    "Accessory Glam": {"Charles & Keith": .75, "ALDO": .6, "Steve Madden": .5, "Zara": .4, "Mango": .3},
    "Ethnic Fusion": {"FabIndia": .7, "H&M": .3, "Skechers": .3, "Mango": .2, "Crocs": .2},
}
for i, p in enumerate(pers):
    for b, pr in persona_brands[p].items():
        base[i, B[b]] = pr
engagement = np.clip(0.6 + 0.25 * fashion + 0.1 * (intent_latent > 0), 0.25, 1.4)[:, None]
brands = multi(np.clip(base * engagement, 0, 0.95), BRANDS)

# attraction driver
drv = []
for i in range(N):
    w = np.array([1 + 1.2 * max(price_sens[i], 0), 0.8 + 0.6 * max(fashion[i], 0) + 0.4 * intl[i],
                  0.5 + 0.5 * max(digital[i], 0), 0.4 + 0.5 * max(digital[i], 0),
                  0.5 + 0.4 * max(price_sens[i], 0), 0.4 + 0.3 * young[i]])
    drv.append(rng.choice(DRIVERS, p=w / w.sum()))
drv = np.array(drv)

# expected spend at the new store (regression target)
intent_mult = {"Definitely not": 0.05, "Probably not": 0.25, "Not sure": 0.5, "Probably will": 0.8, "Definitely will": 1.0}
exp_spend = (spend ** 0.85 * 1.6 * np.array([intent_mult[x] for x in intent])
             * np.exp(0.12 * (L["Q16_brand_preference"] - 3) + 0.08 * (L["Q19_multibrand_appeal"] - 3)
                      - 0.06 * (L["Q18_discount_seeker"] - 3) + rng.normal(0, 0.3, N)))
exp_spend = np.round(exp_spend / 50) * 50

# ------------------------------------------------------------------ 6. open-text feedback
topics = {
    "price":        (["Prices on brand stores are way too high, I wait for sales",
                      "Discounts on apps are fake, MRP is inflated first",
                      "International brands cost double in India compared to Dubai"],
                     ["Fair prices and genuine discounts would make me switch",
                      "Love that I could compare brands and prices in one place"]),
    "authenticity": (["Scared of fake products from Instagram sellers",
                      "Got a first copy shoe once from a marketplace, never again"],
                     ["Authentic brands with a guarantee is exactly what Mumbai needs",
                      "Original products from many brands under one roof sounds amazing"]),
    "delivery":     (["Delivery takes 5-7 days and returns pickup never comes on time",
                      "Return process is a pain, refunds take weeks"],
                     ["90 minute delivery in Mumbai would be a game changer",
                      "Quick delivery and easy returns would make me shop more"]),
    "variety":      (["Malls have the same few brands, nothing new",
                      "Sizes are always out of stock in my size"],
                     ["Great idea to have so many brands in one app, love the variety",
                      "Excited to discover new brands, finally something like Dubai"]),
    "experience":   (["Mumbai traffic makes going to malls tiring, store staff are rude",
                      "Trial rooms are crowded and stores are messy"],
                     ["A fun store with events and styling help would be awesome",
                      "Try in store and get delivery home is a brilliant idea"]),
    "trust":        (["Already happy with Myntra, don't need another app",
                      "Don't trust new apps with my card details"],
                     ["Personalised suggestions would save me a lot of time",
                      "Loyalty rewards across brands would be great"]),
}
neutral_lines = ["Depends on the prices and offers", "Will try if there are good launch offers",
                 "Not sure, maybe for gifting", "Need to see the brand list first", "Okay idea, depends on location"]
hinglish_pos = ["bahut accha idea", "ekdum mast concept", "full support!!"]
hinglish_neg = ["bakwaas pricing yaar", "time waste", "kuch naya nahi hai"]

sent_score = intent_latent + 0.4 * rng.normal(0, 1, N)
true_sent = np.where(sent_score > 0.6, "Positive", np.where(sent_score < -0.5, "Negative", "Neutral"))
true_topic, texts = [], []
for i in range(N):
    t = rng.choice(list(topics))
    true_topic.append(t)
    neg, pos = topics[t]
    if true_sent[i] == "Positive":
        s = rng.choice(pos)
        if rng.random() < 0.3: s += ". " + rng.choice(neg)  # mixed: complaint about today + praise
        if rng.random() < 0.12: s += " - " + rng.choice(hinglish_pos)
    elif true_sent[i] == "Negative":
        s = rng.choice(neg)
        if rng.random() < 0.15: s += " - " + rng.choice(hinglish_neg)
    else:
        s = rng.choice(neutral_lines)
        if rng.random() < 0.4: s = rng.choice(neg) + ". " + s
    texts.append(s)
texts = np.array(texts, dtype=object)

# ------------------------------------------------------------------ 7. assemble clean "truth" frame
start = datetime(2026, 9, 1)
ts = [start + timedelta(minutes=int(m)) for m in rng.integers(0, 60 * 24 * 40, N)]
df = pd.DataFrame({
    "resp_id": [f"R{10000 + i}" for i in range(N)],
    "submitted_at": ts,
    "completion_time_sec": np.clip(rng.normal(420, 120, N), 150, 1500).round(),
    "Q1_age": age, "Q2_gender": gender, "Q3_area": area, "Q4_occupation": occupation,
    "Q5_household_income": income_band, "Q6_household_type": household, "Q7_education": education,
    "Q8_shop_freq_month": freq, "Q9_monthly_fashion_spend": np.round(spend / 10) * 10, "Q10_channel": channel,
    "Q11_platforms_used": platforms, "Q12_categories": categories,
    "Q13_avg_order_value": np.round(aov / 10) * 10, "Q14_social_hrs_week": social.round(1),
    "Q15_bought_intl_brand": np.where(intl, "Yes", "No"),
    **L,
    "Q26_brands_wanted": brands, "Q27_attraction_driver": drv,
    "Q28_expected_monthly_spend": exp_spend, "Q29_feedback_text": texts, "Q30_purchase_intent": intent,
})
truth = pd.DataFrame({"resp_id": df.resp_id, "true_sentiment": true_sent, "true_topic": true_topic,
                      "true_persona": pers})

# ================================================================== 8. INJECT NOISE
raw = df.copy().astype(object)
log = []

def pick(frac):
    return rng.choice(N, int(N * frac), replace=False)

# 8a. missing values (MCAR)
for col, frac in {"Q1_age": .03, "Q5_household_income": .05, "Q9_monthly_fashion_spend": .06,
                  "Q13_avg_order_value": .05, "Q14_social_hrs_week": .07, "Q28_expected_monthly_spend": .04,
                  "Q29_feedback_text": .06, "Q10_channel": .02, **{c: .03 for c in LIKERT_COLS}}.items():
    idx = pick(frac); raw.loc[idx, col] = np.nan
    log.append(("Missing values", col, len(idx)))

# 8b. inconsistent category spellings
gmap = {"Female": ["female", "F", "FEMALE ", "Woman"], "Male": ["male", "M", " MALE", "Man"]}
idx = pick(.12)
for i in idx:
    g = raw.at[i, "Q2_gender"]
    if g in gmap: raw.at[i, "Q2_gender"] = rng.choice(gmap[g])
log.append(("Inconsistent labels", "Q2_gender", len(idx)))
amap = {"Western Suburbs": ["western suburbs", "W. Suburbs", "Andheri", "Bandra", "Western suburb"],
        "South Mumbai": ["SoBo", "south mumbai", "Colaba"], "Central Suburbs": ["central", "Ghatkopar", "Chembur"],
        "Thane": ["thane west", "THANE"], "Navi Mumbai": ["navi mumbai", "Vashi", "NaviMumbai"]}
idx = pick(.15)
for i in idx:
    a = raw.at[i, "Q3_area"]
    if a in amap: raw.at[i, "Q3_area"] = rng.choice(amap[a])
log.append(("Inconsistent labels / sub-localities", "Q3_area", len(idx)))
imap = {"< ₹50K": "Below 50k", "₹50K–1L": "50k-1 lakh", "₹1L–2L": "1-2 Lakh", "₹2L–4L": "2-4 lakh",
        "₹4L–8L": "4-8 Lakh", "> ₹8L": "8L+"}
idx = pick(.10)
for i in idx:
    v = raw.at[i, "Q5_household_income"]
    if isinstance(v, str): raw.at[i, "Q5_household_income"] = imap[v]
log.append(("Inconsistent income band text", "Q5_household_income", len(idx)))
idx = pick(.08)
for i in idx:
    v = raw.at[i, "Q30_purchase_intent"]
    raw.at[i, "Q30_purchase_intent"] = rng.choice([v.upper(), v.lower() + " ", v.replace("will", "yes")])
log.append(("Inconsistent case/spacing", "Q30_purchase_intent", len(idx)))
idx = pick(.05)
for i in idx:
    v = raw.at[i, "Q10_channel"]
    if isinstance(v, str): raw.at[i, "Q10_channel"] = v.lower().replace("mostly ", "")
log.append(("Inconsistent labels", "Q10_channel", len(idx)))

# 8c. money typed as text: "₹4,500", "4.5k", "4500 INR"
for col in ["Q9_monthly_fashion_spend", "Q13_avg_order_value", "Q28_expected_monthly_spend"]:
    idx = pick(.10); n = 0
    for i in idx:
        v = raw.at[i, col]
        if pd.isna(v): continue
        v = float(v); n += 1
        raw.at[i, col] = rng.choice([f"₹{v:,.0f}", f"{v/1000:.1f}k", f"{v:.0f} INR", f"Rs. {v:.0f}"])
    log.append(("Money stored as text", col, n))

# 8d. Likert typed as text / out of range
for col in LIKERT_COLS:
    idx = pick(.04)
    for i in idx:
        v = raw.at[i, col]
        if pd.isna(v): continue
        raw.at[i, col] = rng.choice([f"{int(v)} - {LIKERT_LABELS[int(v)]}", LIKERT_LABELS[int(v)]])
    idx2 = pick(.006); raw.loc[idx2, col] = rng.choice([0, 6, 7, 9], len(idx2))
    log.append(("Likert as text + out-of-range", col, len(idx) + len(idx2)))

# 8e. OUTLIERS
idx = pick(.012); raw.loc[idx, "Q1_age"] = rng.choice([0, 3, 150, 999, 210], len(idx)); log.append(("Impossible ages", "Q1_age", len(idx)))
idx = pick(.01);  raw.loc[idx, "Q1_age"] = [f"{int(a)} yrs" for a in age[idx]]; log.append(("Age typed as text", "Q1_age", len(idx)))
idx = pick(.015)
for i in idx:
    v = raw.at[i, "Q9_monthly_fashion_spend"]
    if isinstance(v, (int, float)) and not pd.isna(v): raw.at[i, "Q9_monthly_fashion_spend"] = v * 100  # extra zeros typo
log.append(("Extreme spend (x100 typo)", "Q9_monthly_fashion_spend", len(idx)))
idx = pick(.005); raw.loc[idx, "Q9_monthly_fashion_spend"] = -rng.integers(500, 5000, len(idx)); log.append(("Negative spend", "Q9_monthly_fashion_spend", len(idx)))
idx = pick(.008); raw.loc[idx, "Q28_expected_monthly_spend"] = rng.integers(400_000, 2_000_000, len(idx)); log.append(("Extreme expected spend", "Q28_expected_monthly_spend", len(idx)))
idx = pick(.01);  raw.loc[idx, "Q8_shop_freq_month"] = rng.choice([45, 60, 99, 100], len(idx)); log.append(("Impossible shopping frequency", "Q8_shop_freq_month", len(idx)))
idx = pick(.008); raw.loc[idx, "Q14_social_hrs_week"] = rng.choice([168, 200, 500], len(idx)); log.append(("Social hrs > hours in a week", "Q14_social_hrs_week", len(idx)))
# genuine (valid) high spenders - real outliers that should be KEPT
idx = pick(.01)
raw.loc[idx, "Q9_monthly_fashion_spend"] = rng.integers(60_000, 150_000, len(idx))
raw.loc[idx, "Q5_household_income"] = "> ₹8L"
log.append(("Genuine luxury shoppers (valid outliers, keep)", "Q9_monthly_fashion_spend", len(idx)))

# 8f. gibberish / junk text
idx = pick(.03); raw.loc[idx, "Q29_feedback_text"] = rng.choice(["asdf", "...", "na", "NA", "-", "good", "😂😂", "nothing"], len(idx))
log.append(("Junk text", "Q29_feedback_text", len(idx)))
idx = pick(.05)
for i in idx:
    v = raw.at[i, "Q29_feedback_text"]
    if isinstance(v, str) and len(v) > 10: raw.at[i, "Q29_feedback_text"] = v.upper() + " !!!"
log.append(("SHOUTING text", "Q29_feedback_text", len(idx)))

# 8g. bots / speeders: very fast completion + straight-lined Likert
idx = pick(.03)
raw.loc[idx, "completion_time_sec"] = rng.integers(25, 75, len(idx))
for i in idx:
    val = rng.choice([3, 5, 1])
    for c in LIKERT_COLS: raw.at[i, c] = val
    raw.at[i, "Q30_purchase_intent"] = rng.choice(INTENT)
log.append(("Speeders / straight-liners (bots)", "completion_time_sec + Likert", len(idx)))

# 8h. label noise on target
idx = pick(.04); raw.loc[idx, "Q30_purchase_intent"] = rng.choice(INTENT, len(idx)); log.append(("Random label noise", "Q30_purchase_intent", len(idx)))

# 8i. mixed timestamp formats
fmts = ["%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M", "%b %d, %Y %I:%M %p"]
raw["submitted_at"] = [t.strftime(fmts[rng.choice(3, p=[.7, .2, .1])]) for t in df.submitted_at]
log.append(("Mixed date formats", "submitted_at", N))

# 8j. duplicates: exact re-submissions + same person with new ID
dup_exact = raw.iloc[rng.choice(N, 40, replace=False)].copy()
dup_newid = raw.iloc[rng.choice(N, 20, replace=False)].copy()
dup_newid["resp_id"] = [f"R{90000 + i}" for i in range(len(dup_newid))]
raw = pd.concat([raw, dup_exact, dup_newid], ignore_index=True)
raw = raw.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
log.append(("Exact duplicate submissions", "all", 40)); log.append(("Duplicate person, new resp_id", "all", 20))

raw.to_csv("data/raw_survey_responses.csv", index=False)
truth.to_csv("data/_ground_truth_hidden.csv", index=False)
pd.DataFrame(log, columns=["noise_type", "column", "rows_affected"]).to_csv("data/noise_injection_log.csv", index=False)
pd.DataFrame(SURVEY, columns=["code", "question", "type", "options", "used_by"]).to_csv("data/survey_codebook.csv", index=False)

print(f"Raw rows: {len(raw)}  (clean truth N={N})")
print("Purchase-intent mix (truth):", pd.Series(intent).value_counts(normalize=True).round(3).to_dict())
print("Noise injections:", len(log))
