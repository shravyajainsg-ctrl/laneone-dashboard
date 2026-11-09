"""
Shared configuration: business name, survey codebook, category orders.
Every other script imports from here so the survey, data, cleaning and
dashboard always agree on column names and answer options.
"""

BUSINESS_NAME = "LaneOne"          # working name - change freely
CITY = "Mumbai"
RANDOM_SEED = 42
N_RESPONDENTS = 2000

AREAS = ["South Mumbai", "Western Suburbs", "Central Suburbs", "Thane", "Navi Mumbai", "Other MMR"]
GENDERS = ["Female", "Male", "Non-binary / Prefer not to say"]
OCCUPATIONS = ["Student", "Salaried professional", "Business owner / Self-employed",
               "Freelancer / Creator", "Homemaker", "Retired"]
INCOME_BANDS = ["< ₹50K", "₹50K–1L", "₹1L–2L", "₹2L–4L", "₹4L–8L", "> ₹8L"]   # monthly household
INCOME_MID = {"< ₹50K": 35_000, "₹50K–1L": 75_000, "₹1L–2L": 150_000,
              "₹2L–4L": 300_000, "₹4L–8L": 600_000, "> ₹8L": 1_000_000}
HOUSEHOLD = ["Living alone / with flatmates", "Couple, no kids", "Family with kids", "Joint family"]
EDUCATION = ["School / Diploma", "Graduate", "Post-graduate", "Doctorate / Professional"]
CHANNELS = ["Mostly online", "Both equally", "Mostly offline"]
PLATFORMS = ["Myntra", "Ajio", "Nykaa Fashion", "Tata CLiQ", "Amazon Fashion",
             "Shoppers Stop / Lifestyle", "Brand-owned stores", "Instagram sellers"]
CATEGORIES = ["Apparel", "Footwear", "Bags & accessories", "Athleisure / sportswear",
              "Watches & eyewear", "Beauty & fragrance", "Kidswear"]
BRANDS = ["Nike", "Adidas", "Puma", "Skechers", "Levi's", "Tommy Hilfiger", "Calvin Klein",
          "Zara", "H&M", "Mango", "Charles & Keith", "ALDO", "Steve Madden", "Crocs", "FabIndia"]
DRIVERS = ["Big discounts & sales", "Authentic international brands", "Fast delivery (90 min)",
           "Personalised recommendations", "Loyalty rewards", "In-store experience & events"]
INTENT = ["Definitely not", "Probably not", "Not sure", "Probably will", "Definitely will"]
LIKERT_LABELS = {1: "Strongly disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly agree"}

LIKERT_COLS = {
    "Q16_brand_preference":      "I prefer buying branded products over unbranded ones.",
    "Q17_authenticity_matters":  "A 100% authenticity guarantee strongly influences where I shop.",
    "Q18_discount_seeker":       "I mostly buy fashion only during sales or with discount codes.",
    "Q19_multibrand_appeal":     "I would like to discover many brands in one store / one app.",
    "Q20_ai_reco_interest":      "I would value AI-based personalised outfit and product recommendations.",
    "Q21_phygital_interest":     "I would use 'try in store, get it delivered home' or 'reserve online, try in store'.",
    "Q22_loyalty_influence":     "Loyalty points and member rewards influence where I shop.",
    "Q23_sustainability_values": "I prefer brands with sustainable or ethical practices, even at a slightly higher price.",
    "Q24_data_sharing_comfort":  "I am comfortable sharing my shopping data in return for personalised offers.",
    "Q25_fast_delivery_need":    "Same-day / 90-minute delivery would make me shop more often.",
}

# Full questionnaire (question text, type, options, which algorithm uses it)
SURVEY = [
    # ---- Section A: Profile (segmentation + classification features)
    ("Q1_age", "What is your age (in years)?", "Numeric", "", "Classification, Regression, Clustering"),
    ("Q2_gender", "What is your gender?", "Single choice", " · ".join(GENDERS), "Classification, Descriptive"),
    ("Q3_area", "Where in Mumbai do you live?", "Single choice", " · ".join(AREAS), "Classification, Store-location diagnostics"),
    ("Q4_occupation", "What is your occupation?", "Single choice", " · ".join(OCCUPATIONS), "Classification, Clustering"),
    ("Q5_household_income", "What is your monthly household income?", "Single choice (ordinal)", " · ".join(INCOME_BANDS), "Classification, Regression, Clustering"),
    ("Q6_household_type", "Which best describes your household?", "Single choice", " · ".join(HOUSEHOLD), "Classification, Descriptive"),
    ("Q7_education", "What is your highest education level?", "Single choice (ordinal)", " · ".join(EDUCATION), "Classification"),
    # ---- Section B: Current behaviour
    ("Q8_shop_freq_month", "How many times a month do you shop for fashion, footwear or accessories (online + offline)?", "Numeric", "", "Classification, Regression, Clustering"),
    ("Q9_monthly_fashion_spend", "Roughly how much do you spend on fashion, footwear & accessories per month (₹)?", "Numeric (₹)", "", "Regression (key driver), Clustering"),
    ("Q10_channel", "How do you usually shop for fashion?", "Single choice", " · ".join(CHANNELS), "Classification, Omnichannel planning"),
    ("Q11_platforms_used", "Which platforms / stores have you bought fashion from in the last 6 months?", "Multi-select", " · ".join(PLATFORMS), "Competitor diagnostics, Association"),
    ("Q12_categories", "Which categories do you buy most often?", "Multi-select", " · ".join(CATEGORIES), "Category mix, Association"),
    ("Q13_avg_order_value", "What is your typical spend per fashion order / bill (₹)?", "Numeric (₹)", "", "Regression, Break-even (AOV)"),
    ("Q14_social_hrs_week", "How many hours a week do you spend on Instagram / YouTube / fashion apps?", "Numeric", "", "Classification, Marketing-channel planning"),
    ("Q15_bought_intl_brand", "Have you bought an international fashion brand in the last 6 months?", "Yes / No", "Yes · No", "Classification"),
    # ---- Section C: Attitudes (Likert 1-5)
    *[(k, v, "Likert 1–5", "Strongly disagree … Strongly agree", "Classification, Clustering, Hypothesis tests")
      for k, v in LIKERT_COLS.items()],
    # ---- Section D: Intent
    ("Q26_brands_wanted", f"If these brands were all available at {BUSINESS_NAME}, which would you buy? (select all)", "Multi-select", " · ".join(BRANDS), "Association rule mining (Apriori)"),
    ("Q27_attraction_driver", f"What would MOST make you try {BUSINESS_NAME}?", "Single choice", " · ".join(DRIVERS), "Classification, Prescriptive marketing"),
    ("Q28_expected_monthly_spend", f"How much would you expect to spend per month at {BUSINESS_NAME} (₹)?", "Numeric (₹)", "", "Regression TARGET"),
    ("Q29_feedback_text", "In your own words: what frustrates you about fashion shopping in Mumbai today, and what would you want from a new multi-brand store?", "Open text", "", "Text analytics / sentiment"),
    ("Q30_purchase_intent", f"If {BUSINESS_NAME} (store + app) opened in Mumbai, how likely are you to shop there in the first 3 months?", "Single choice (5-pt)", " · ".join(INTENT), "Classification TARGET (Probably/Definitely will = 1)"),
]

NUMERIC_COLS = ["Q1_age", "Q8_shop_freq_month", "Q9_monthly_fashion_spend", "Q13_avg_order_value",
                "Q14_social_hrs_week", "Q28_expected_monthly_spend"]
MULTI_COLS = {"Q11_platforms_used": PLATFORMS, "Q12_categories": CATEGORIES, "Q26_brands_wanted": BRANDS}
