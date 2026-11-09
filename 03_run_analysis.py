"""
STEP 3 - Run every algorithm, save all charts (PNG) and result tables (CSV).
The Streamlit app does the same thing interactively; this script is for the report.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from analytics import *
import charts as C
from config import *

OUT = "outputs/"
raw = pd.read_csv("data/raw_survey_responses.csv")
df = load_clean()
truth = pd.read_csv("data/_ground_truth_hidden.csv")
summary = {}

def save(fig, name):
    fig.savefig(f"{OUT}charts/{name}.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---------------- data quality
save(C.missing_before_after(raw, df, ["Q1_age", "Q5_household_income", "Q9_monthly_fashion_spend", "Q13_avg_order_value",
                                      "Q14_social_hrs_week", "Q28_expected_monthly_spend", "Q29_feedback_text", "Q20_ai_reco_interest"]), "01_missing")
raw_spend = raw.Q9_monthly_fashion_spend.astype(str).str.replace(r"[₹,]|INR|Rs\.", "", regex=True)
save(C.outlier_box(raw_spend, df.Q9_monthly_fashion_spend, "Monthly fashion spend - outliers before vs after"), "02_outliers")

# ---------------- descriptive
summary["kpis"] = kpis(df)
save(C.intent_distribution(df), "03_intent")
save(C.rate_by(df, "Q5_household_income", INCOME_BANDS, "Interest rate by monthly household income (line = average)"), "04_income")
save(C.rate_by(df, "Q3_area", None, "Interest rate by Mumbai zone (line = average)"), "05_zone")
save(C.age_hist(df), "06_age")
save(C.likert_dumbbell(df), "07_likert")
save(C.share_bars(df, "plat_", PLATFORMS, "Q11 - Where respondents shop today"), "08_platforms")
save(C.share_bars(df, "cat_", CATEGORIES, "Q12 - Categories bought most often"), "09_categories")
save(C.driver_split(df), "10_drivers")

# ---------------- diagnostic
corr_cols = ["Q1_age", "income_ord", "log_monthly_spend", "Q8_shop_freq_month", "Q14_social_hrs_week", *LIKERT_COLS,
             "n_brand", "intent_score", "Q28_expected_monthly_spend"]
labels = [c.split("_", 1)[1].replace("_", " ") if c.startswith("Q") else c.replace("_", " ") for c in corr_cols]
fig, corr = C.corr_heatmap(df, corr_cols, labels); save(fig, "11_corr")
corr.to_csv(OUT + "correlation_matrix.csv")
save(C.spend_scatter(df), "12_spend_scatter")
save(C.rate_by(df, "Q10_channel", None, "Interest rate by preferred channel"), "13_channel")
ht = hypothesis_tests(df); ht.to_csv(OUT + "hypothesis_tests.csv", index=False)

# ---------------- classification
cls = run_classification(df)
cls["table"].to_csv(OUT + "classification_models.csv", index=False)
save(C.roc_plot(cls["rocs"], cls["table"]), "14_roc")
save(C.model_bars(cls["table"], "ROC-AUC (5-fold CV)", "Classifier comparison - ROC-AUC (5-fold CV)"), "15_cls_models")
save(C.confusion(cls["cm"], f"Confusion matrix - {cls['best']}"), "16_cm")
save(C.importance(cls["importance"], "What predicts interest? (Random Forest importance)"), "17_importance")
seg = conversion_segment(cls["oof_prob"])
save(C.segment_chart(seg, df.interested.values), "18_segments")
df["p_interest"] = cls["oof_prob"]; df["conv_segment"] = seg

# ---------------- regression
reg = run_regression(df)
reg["table"].to_csv(OUT + "regression_models.csv", index=False)
a, p = reg["preds"][reg["best"]]
save(C.actual_vs_pred(a, p, reg["best"]), "19_reg_fit")
save(C.model_bars(reg["table"], "R² (5-fold CV)", "Regressor comparison - R² (5-fold CV, log spend)"), "20_reg_models")
save(C.coef_chart(reg["coef_pct"]), "21_reg_coef")
df["pred_spend"] = np.expm1(reg["model"].predict(df.assign(intent_score=df.intent_score)[REG_NUM + CAT_FEATURES]))

# ---------------- clustering
cl = run_clustering(df)
df["persona"] = cl["persona"]
cl["profile"].to_csv(OUT + "persona_profiles.csv")
save(C.elbow_silhouette(cl["ks"], cl["inertia"], cl["silhouette"], cl["k"]), "22_elbow")
save(C.pca_small_multiples(cl["pca"], cl["persona"]), "23_pca")
save(C.persona_heatmap(cl["profile"]), "24_personas")

# ---------------- association
rules, single, nb = association_rules(df)
rules.to_csv(OUT + "association_rules.csv", index=False)
opt = launch_brand_optimizer(df); opt.to_csv(OUT + "launch_brand_optimizer.csv", index=False)
save(C.rules_scatter(rules), "25_rules")
save(C.coverage_curve(opt), "26_coverage")
fig, ax = plt.subplots(figsize=(7.5, 4)); C._hbar(ax, single); ax.xaxis.set_major_formatter(C.pct)
ax.set_title("Brand demand among interested respondents (support)"); fig.tight_layout(); save(fig, "27_brand_support")

# ---------------- text
ta = run_text_analytics(df, truth)
ta["topics"].to_csv(OUT + "text_topics.csv")
save(C.sentiment_by_intent(ta["data"]), "28_sent_intent")
save(C.topic_chart(ta["topics"]), "29_topics")
save(C.terms_chart(ta["pos_terms"], ta["neg_terms"]), "30_terms")

# ---------------- prescriptive: break-even, Monte Carlo, sensitivity
si = survey_inputs(df)
sims, mcs, static = {}, {}, {}
for k in SCENARIOS:
    be = run_scenario(k, si["aov"], si["stated_monthly_spend"])
    sims[k] = be["sim"]; static[k] = {x: v for x, v in be.items() if x != "sim"}
    mcs[k] = monte_carlo(k, si["aov"], si["stated_monthly_spend"], n=1000)
    be["sim"].to_csv(OUT + f"cashflow_{k[0]}.csv", index=False)
save(C.cash_curves(sims), "31_cash")
save(C.profit_curves(sims), "32_profit")
save(C.mc_hist(mcs), "33_montecarlo")
rec = list(SCENARIOS)[2]
sens = sensitivity(rec, si["aov"], si["stated_monthly_spend"])
save(C.tornado(sens, "Sensitivity: change in cash at month 36 (staged plan)"), "34_tornado")
comp = compare_scenarios(si["aov"], si["stated_monthly_spend"], n=1000); comp.to_csv(OUT + "scenario_comparison.csv", index=False)

# ---------------- sustainability
card, overall = sustainability_scorecard(df, static[rec] | {"ltv_cac": static[rec]["ltv_cac"]}, mcs[rec], single, seg)
card.to_csv(OUT + "sustainability_scorecard.csv", index=False)
save(C.scorecard_chart(card), "35_scorecard")

# ---------------- prescriptive targeting table
df["expected_value"] = df.p_interest * df.pred_spend
tgt = df.groupby(["conv_segment", "persona"]).agg(customers=("resp_id", "count"), p_interest=("p_interest", "mean"),
                                                  pred_spend=("pred_spend", "median"), expected_value=("expected_value", "mean")).reset_index()
tgt.sort_values("expected_value", ascending=False).to_csv(OUT + "targeting_matrix.csv", index=False)
df.to_csv(OUT + "scored_respondents.csv", index=False)

# ---------------- summary for the report
summary.update({
    "n_raw": len(raw), "n_clean": len(df), "interest_rate": df.interested.mean(),
    "best_classifier": cls["best"], "cls_table": cls["table"].round(3).to_dict("records"),
    "segments": pd.Series(seg).value_counts(normalize=True).round(3).to_dict(),
    "segment_actual": df.groupby("conv_segment").interested.mean().round(3).to_dict(),
    "best_regressor": reg["best"], "reg_table": reg["table"].round(3).to_dict("records"),
    "top_importance": cls["importance"].head(8).round(3).to_dict(),
    "silhouette": dict(zip(cl["ks"], np.round(cl["silhouette"], 3))),
    "personas": cl["profile"].round(2).reset_index().to_dict("records"),
    "top_rules": rules.head(10).round(3).to_dict("records"), "brand_support": single.round(3).to_dict(),
    "optimizer": opt.round(3).to_dict("records"),
    "lexicon_acc": ta["lexicon_accuracy"], "supervised_acc": ta["supervised_accuracy"],
    "sentiment_mix": ta["data"].sentiment.value_counts(normalize=True).round(3).to_dict(),
    "topics": ta["topics"].round(3).reset_index().to_dict("records"),
    "survey_inputs": si, "static": {k: {x: (float(v) if v is not None else None) for x, v in d.items()} for k, d in static.items()},
    "comparison": comp.round(3).to_dict("records"), "sensitivity": sens.round(0).to_dict("records"),
    "scorecard": card.to_dict("records"), "score_overall": overall,
    "hypotheses": ht[["H", "Hypothesis (H1 / alternative)", "Test", "p-value", "Effect / evidence", "Decision"]].to_dict("records"),
    "targeting_top": tgt.sort_values("expected_value", ascending=False).head(6).round(2).to_dict("records"),
})
json.dump(summary, open(OUT + "results_summary.json", "w"), indent=1, default=str)
print("done", len(df))
