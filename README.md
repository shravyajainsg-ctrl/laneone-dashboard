# LaneOne — Data-analytics feasibility study for a multi-brand fashion hub in Mumbai

A 6th Street–style, accessible-premium multi-brand fashion & lifestyle platform (app + experience store).
Everything here uses a **synthetic** survey (n ≈ 2,000) built to validate the idea end to end.

## Run it
```bash
pip install -r requirements.txt
python 01_generate_synthetic_data.py   # 30-question survey -> messy raw data (noise + outliers injected)
python 02_clean_transform.py           # cleaning & transformation, every fix logged
python 03_run_analysis.py              # all models, 35 charts (outputs/charts), result tables (outputs/)
python tools/build_eda_notebook.py     # rebuilds the executed EDA notebook + HTML report
streamlit run app.py                   # interactive dashboard (12 pages)
```
The app regenerates the data automatically if `data/clean_survey.csv` is missing.

## Files
| File | Purpose |
|---|---|
| `config.py` | Business name, survey codebook (30 questions → algorithm mapping), answer options |
| `01_generate_synthetic_data.py` | Latent-trait simulation + 51 deliberate noise injections |
| `02_clean_transform.py` | Duplicates, bots, label standardisation, money parsing, outlier rules, imputation, feature engineering |
| `analytics.py` | Hypothesis tests, 6 classifiers, 6 regressors, K-Means, Apriori, launch-brand optimiser, sentiment, break-even, Monte-Carlo, sustainability, new-customer scorer |
| `eda.py` | Descriptive (D1–D7) and diagnostic (G1–G8) analyses, each returning chart + table + data-driven explanation |
| `04_descriptive_diagnostic_EDA.ipynb` | Executed notebook: code, outputs and What/Why/Insight for every graph (opens in Jupyter, Colab, VS Code) |
| `outputs/descriptive_diagnostic_EDA.html` | Same EDA as a standalone web page |
| `charts.py` | All figures (colour-blind-checked palette) |
| `app.py` | Streamlit dashboard |
| `data/` | raw, clean, codebook, noise log, cleaning log, hidden ground truth (validation only) |
| `outputs/` | charts + CSV results + `results_summary.json` |

## Analytics ladder
* **Descriptive** – distributions, interest by income/zone/age/channel, platforms, categories
* **Diagnostic** – Spearman correlation matrix, 11 hypothesis tests (chi-square, Mann-Whitney, Kruskal-Wallis, Spearman)
* **Predictive** – classification (will they buy?), regression (how much?), clustering (personas), sentiment
* **Prescriptive** – launch-brand optimiser, segment actions, break-even + Monte-Carlo risk, staged go/no-go plan, new-customer scoring

Change `BUSINESS_NAME` in `config.py` to rebrand. Financial assumptions live in `analytics.SCENARIOS` and are adjustable in the dashboard.
