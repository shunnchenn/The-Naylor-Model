"""
make_main_notebook.py  —  Generate the master Naylor_Model.ipynb at the repo root.

The notebook walks the whole model end-to-end and is runnable top-to-bottom WITHOUT
network (it reads the cached feature CSV and the on-disk figures):
  1. Data            2. SSSI            3. Model B (tuned XGBoost)   4. GLM equation
  5. xSB quadrant    6. Benchmark/tuning recap                      7. Build the report

Run:  python3 scripts/make_main_notebook.py
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "Naylor_Model.ipynb"

def md(t):   return nbf.v4.new_markdown_cell(t)
def code(s): return nbf.v4.new_code_cell(s)

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

nb.cells = [
md("""# The Naylor Model — Master Notebook

Base-stealing intelligence for slow-but-elite stealers (the *Josh Naylor* trait: a runner who
steals better than ~99% of MLB without the wheels). This notebook runs the model end-to-end.

**Runs without network** — it reads the cached feature data (`Naylor_Model_Data.csv`) and the
tuned hyperparameters, refits Model B, and rebuilds the report. The full Statcast data pull lives in
`scripts/v7_explore.py` (network-bound) and is *not* required here.

| Section | What |
|---|---|
| 1 | Data — the runner-season master |
| 2 | SSSI — Slow-Steal Skill Index leaderboard |
| 3 | Model B — season-level Bayesian-tuned **XGBoost** (AUC by era) |
| 4 | GLM — the steal-success equation |
| 5 | xSB — speed-vs-production quadrant |
| 6 | **Model A — per-attempt XGBoost (AUC ~0.74)** |
| 7 | Benchmark & tuning recap |
| 8 | Build the v8 report |
"""),

md("## Setup"),
code("""\
import sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

ROOT = Path().resolve()
while not (ROOT / "Figures").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
SCRIPTS = ROOT / "scripts"
DATA    = ROOT / "data"
FIGS    = ROOT / "Figures"
sys.path.insert(0, str(SCRIPTS))
print("Repo root:", ROOT)
"""),

md("""## 1 — Data: the runner-season master

`Naylor_Model_Data.csv` is the curated master — one row per qualified runner-season with every
feature plus the SSSI ranking. (`data/` holds the full working set the pipeline reads.)"""),
code("""\
data = pd.read_csv(ROOT / "Naylor_Model_Data.csv")
print(f"{len(data)} runner-seasons × {data.shape[1]} columns")
data[["player_name", "season", "sprint_speed", "jump_time", "lead_gain",
      "real_sb_pct", "sb_attempts"]].head()
"""),

md("""## 2 — SSSI: Slow-Steal Skill Index

A weighted composite that surfaces the Naylor/Soto archetype — elite-performing slow runners.
Naylor and Soto were held out when the weights were fit, so their ranking is genuinely out-of-sample."""),
code("""\
sssi_col = next((c for c in data.columns if c.lower().startswith("sssi")), None)
top = data.sort_values(sssi_col, ascending=False).head(15)
top[["player_name", "season", "sprint_speed", "real_sb_pct", sssi_col]].reset_index(drop=True)
"""),

md("""## 3 — Model B: Bayesian-tuned XGBoost

Model B predicts season steal success. v8 upgraded it from a gradient-boosting classifier to a
**Bayesian-tuned XGBoost** (Optuna, 100 trials). `model_xgb.main()` refits it on the cached features
(no network), recomputes AUC by era, and refreshes the AUC + importance figures."""),
code("""\
import model_xgb
auc_rows = model_xgb.main()
pd.DataFrame(auc_rows)
"""),
code("""\
display(Image(filename=str(FIGS / "Fig_v7_AUC.png")))
"""),

md("""## 4 — GLM: the steal-success equation

The interpretable model. Each lever's weight is reported as the change in steal-success rate for a
+1-SD improvement — the basis for the report's equation figure and coaching levers."""),
code("""\
glm = pd.read_csv(DATA / "DF_v7_GLM_PlainEnglish.csv")
display(glm)
display(Image(filename=str(FIGS / "Fig_v8_Equation.png")))
"""),

md("""## 5 — xSB: speed-vs-production quadrant

A descriptive lens splitting the league into Realized Burners, Untapped Wheels, Crafty Technicians
(the Naylor/Soto archetype), and Stationary runners."""),
code("""\
xsb = pd.read_csv(DATA / "DF_v7_xSB_Outcome.csv")
print(xsb["quadrant"].value_counts())
display(Image(filename=str(FIGS / "Fig_v8_xSB_Quadrant.png")))
"""),

md("""## 6 — Per-attempt model (Model A) — the AUC jump

The season model tops out near 0.62 on 673 rows. Modeling the **~10,400 individual tracked attempts**
(Statcast leads cache) instead lifts CV AUC to **~0.74** — the per-pitch lead distances are what
decide a steal. Leakage-checked; deep learning is *not* used (gradient boosting wins on tabular data
this size)."""),
code("""\
import model_perattempt
pa = model_perattempt.main()
display(pa)
"""),

md("""## 7 — Benchmark & tuning recap

Six season-level classifiers compared on the same de-leaked data, plus the tuned XGBoost params."""),
code("""\
bench = pd.read_csv(DATA / "DF_benchmark_AUC.csv")
display(bench.sort_values("auc", ascending=False).reset_index(drop=True))
params = pd.read_csv(DATA / "DF_xgb_tuned_params.csv")
display(params.T)
"""),

md("""## 8 — Build the v8 report

Regenerates the applied main report (repo root) and the Technical Appendix (`Reports/`). Reads the
data refreshed above — no network."""),
code("""\
import importlib.util, subprocess
if importlib.util.find_spec("docx") is None:
    print("python-docx is not installed in this kernel — skipping the in-notebook build.")
    print("Run it from the project environment:  python3 scripts/build_v8_report.py")
else:
    r = subprocess.run([sys.executable, str(SCRIPTS / "build_v8_report.py")],
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:])
"""),

md("""---
### Next steps
See [`AUC_Roadmap.md`](AUC_Roadmap.md) for the prioritized plan to push AUC higher — the untapped
matchup variables (pitcher handedness, pitch type, count, catcher identity). To re-pull raw Statcast
data, run `python3 scripts/v7_explore.py` (requires network)."""),
]

nbf.write(nb, str(OUT))
print(f"Wrote {OUT}")
