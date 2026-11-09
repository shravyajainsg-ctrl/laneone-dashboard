"""
Builds 04_descriptive_diagnostic_EDA.ipynb WITH outputs (tables, charts, explanations)
and a standalone HTML copy, by executing each code cell here.
(Jupyter is not required to build it; the notebook re-runs normally in Jupyter / Colab / VS Code.)
Run from the project folder:  python tools/build_eda_notebook.py
"""
import base64, contextlib, inspect, io, json, sys, types, html, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import eda

# ---- fake IPython.display that records outputs ------------------------------------------
_outputs = []
def _display(obj):
    if isinstance(obj, pd.DataFrame):
        _outputs.append({"output_type": "display_data", "metadata": {},
                         "data": {"text/html": obj.to_html(border=0, classes="dataframe"), "text/plain": obj.to_string()}})
    elif hasattr(obj, "savefig"):
        b = io.BytesIO(); obj.savefig(b, format="png", dpi=110, bbox_inches="tight")
        _outputs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}})
    elif isinstance(obj, _Markdown):
        _outputs.append({"output_type": "display_data", "metadata": {}, "data": {"text/markdown": obj.data, "text/plain": obj.data}})
class _Markdown:
    def __init__(self, data): self.data = data
ipy = types.ModuleType("IPython"); disp = types.ModuleType("IPython.display")
disp.display, disp.Markdown = _display, _Markdown
ipy.display = disp; ipy.get_ipython = lambda: None; ipy.version_info = (9, 0, 0); sys.modules["IPython"] = ipy; sys.modules["IPython.display"] = disp

# ---- notebook cells ------------------------------------------------------------------------
cells = []
def md(text): cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "source": src, "execution_count": None, "outputs": []})

md("""# LaneOne — Descriptive & Diagnostic Analytics (EDA)
**Business question:** is a 6th Street–style, accessible-premium multi-brand fashion hub viable in Mumbai, and *who* would buy?

**Dataset:** synthetic survey of 2,000 simulated Mumbai shoppers (30 questions) → **1,940 clean respondents** after the cleaning
pipeline (`02_clean_transform.py`). Target: `interested` = 1 if Q30 is *Probably* or *Definitely will*.

**Structure**
* **Descriptive analytics (D1–D7)** — *what* the data looks like: size, summary statistics, distributions, categorical profile,
  attitudes, multi-select answers and the target.
* **Diagnostic analytics (G1–G8)** — *why* interest differs: Pearson vs Spearman correlation, drivers of interest,
  Cramér's V for categories, a zone × income drill-down, spend relationships, a dose-response check, hypothesis tests with
  effect sizes, outliers and multicollinearity.

Every chart is followed by **What / Why / Insight**. All numbers in the explanations are computed from the data, not typed in.

| Technique | Why it fits this dataset |
|---|---|
| Summary statistics + skew/kurtosis | Spend is money data, so skew must be measured before choosing mean vs median |
| Histograms (log scale for ₹) | Shows shape; log reveals the bell shape hidden by skew |
| Frequency bars / diverging Likert bars | Correct display for nominal and ordinal survey answers |
| Pearson **and** Spearman | Likert = ordinal and ₹ = skewed, so rank correlation is the right default; comparing both exposes non-linearity |
| Point-biserial / Spearman vs 0/1 target | Ranks candidate drivers of interest before modelling |
| Cramér's V | Correlation can't be used for unordered categories (zone, occupation…) |
| Drill-down cross-tab | Separates two confounded drivers (zone and income) |
| Dose-response line chart | Stronger causal evidence than a single yes/no gap |
| Chi-square, Mann-Whitney, Kruskal-Wallis + effect sizes | Non-parametric tests suit ordinal/skewed data; effect sizes show commercial importance |
| IQR outliers + VIF | Confirms remaining outliers are genuine and predictors aren't redundant |
""")
code("""# Setup: load the cleaned data and the analysis helpers
import re, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display, Markdown
import charts as C                      # shared colour-blind-safe chart style
from config import *                    # survey codebook & category orders
from eda import NUM, LIK_SHORT, CORR_COLS, _lab, _cramers_v, pct
import analytics as A

df = A.load_clean()                     # data/clean_survey.csv
print(df.shape)

def show(r):
    \"\"\"Display a result: table(s), chart, then the data-driven explanation.\"\"\"
    for k in ("table", "table2"):
        if r.get(k) is not None: display(r[k])
    if r.get("fig") is not None: display(r["fig"]); plt.close(r["fig"])
    display(Markdown(r["text"]))""")

for section, items in [("Part A — Descriptive analytics", eda.DESCRIPTIVE), ("Part B — Diagnostic analytics", eda.DIAGNOSTIC)]:
    md(f"## {section}")
    for title, fn in items:
        md(f"### {title}\n{inspect.getdoc(fn)}")
        code(inspect.getsource(fn).rstrip() + f"\n\nshow({fn.__name__}(df))")

md("""## Summary of insights
**Descriptive**
* Young, urban, working-age sample (median age 30); 46% salaried; Western Suburbs largest zone.
* Spend variables are heavily right-skewed, so medians and log transforms are used throughout.
* 51% interested; answers are polarised (few 'Not sure').
* Authenticity is the most agreed attitude; data-sharing comfort the least.
* Myntra, Amazon and Ajio are the incumbents; no brand is wanted by more than ~28%, so a multi-brand assortment is needed.

**Diagnostic**
* Interest is driven by **spending power** (current spend, income) and the **concept pillars** (multi-brand appeal,
  phygital, AI recommendations, authenticity). Discount-seeking and age pull it down.
* Income is the dominant categorical driver; zone adds an effect *within* the same income tier.
* Attitudes rise step by step with intent (dose-response), so the relationship is consistent, not a one-off gap.
* The largest effect sizes: current spend, multi-brand appeal and the spend link. Gender is significant but small.
* Remaining outliers are genuine and multicollinearity is low (all VIF < 5).

**Strategic implication:** target ₹1L+ households in the Western Suburbs and South Mumbai, lead with authenticity and
multi-brand discovery (not discounts), and make personalisation opt-in.""")

# ---- execute ------------------------------------------------------------------------------
ns = {}
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    _outputs.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(c["source"], ns)
    outs = []
    if buf.getvalue():
        outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
    outs += list(_outputs)
    c["outputs"] = outs
    c["execution_count"] = sum(1 for x in cells[:i + 1] if x["cell_type"] == "code")
    plt.close("all")

def _lines(s):
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]
for c in cells:
    c["source"] = _lines(c["source"])
    for o in c.get("outputs", []):
        if "text" in o: o["text"] = _lines(o["text"])
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("04_descriptive_diagnostic_EDA.ipynb", "w"), indent=1)

# ---- HTML copy ----------------------------------------------------------------------------
def mdhtml(t):
    import re
    out, table = [], []
    merged = []
    for line in t.split("\n"):
        cont = merged and line.strip() and not line.startswith(("|", "#", "* ")) and merged[-1].strip() \
            and not merged[-1].startswith(("|", "#"))
        if cont: merged[-1] += " " + line.strip()
        else: merged.append(line)
    t = "\n".join(merged)
    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s); s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    for line in t.split("\n"):
        if line.startswith("|"):
            if set(line.replace("|", "").strip()) <= set("-: "): continue
            table.append("<tr>" + "".join(f"<td>{inline(x.strip())}</td>" for x in line.strip("|").split("|")) + "</tr>"); continue
        if table: out.append("<table>" + "".join(table) + "</table>"); table = []
        if line.startswith("### "): out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "): out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "): out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("* "): out.append(f"<li>{inline(line[2:])}</li>")
        elif line.strip(): out.append(f"<p>{inline(line)}</p>")
    if table: out.append("<table>" + "".join(table) + "</table>")
    return "\n".join(out)
body = []
for c in cells:
    src = "".join(c["source"])
    if c["cell_type"] == "markdown":
        body.append(mdhtml(src)); continue
    body.append(f"<details><summary>Show Python code</summary><pre>{html.escape(src)}</pre></details>")
    for o in c["outputs"]:
        d = o.get("data", {})
        if o["output_type"] == "stream": body.append(f"<pre class='out'>{html.escape(''.join(o['text']))}</pre>")
        elif "image/png" in d: body.append(f"<img src='data:image/png;base64,{d['image/png']}'>")
        elif "text/html" in d: body.append(f"<div class='tbl'>{d['text/html']}</div>")
        elif "text/markdown" in d: body.append(f"<div class='expl'>{mdhtml(d['text/markdown'])}</div>")
page = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>LaneOne - Descriptive & Diagnostic EDA</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#1a1a19;line-height:1.5}}
h1{{font-weight:500}} h2{{margin-top:2.2em;border-bottom:1px solid #e5e4df;padding-bottom:6px}} h3{{margin-top:1.8em}}
img{{max-width:100%;display:block;margin:12px 0}} pre{{background:#f6f6f4;padding:12px;overflow-x:auto;font-size:12px;border-radius:6px}}
.tbl{{overflow-x:auto;font-size:12px}} table{{border-collapse:collapse;margin:10px 0}} td,th{{border-bottom:1px solid #e5e4df;padding:4px 8px;text-align:left}}
.expl{{background:#f2f7fd;border-left:3px solid #2a78d6;padding:4px 14px;margin:8px 0 18px;border-radius:4px}}
summary{{cursor:pointer;color:#52514e;font-size:13px}} li{{margin-left:1.2em}}
</style></head><body>{''.join(body)}</body></html>"""
open("outputs/descriptive_diagnostic_EDA.html", "w").write(page)
print("notebook cells:", len(cells), "| code cells:", sum(c['cell_type'] == 'code' for c in cells))
