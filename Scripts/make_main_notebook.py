"""
make_main_notebook.py — generate the master Naylor_Model.ipynb at the repo root (V10).

The notebook walks the model end-to-end and runs WITHOUT network: it reads the
consolidated raw data (Data/Raw_Season.csv, Data/Raw_Attempts.csv), the model results
in Output/Results, and the figures/tables in Output/.

Run:  python3 Scripts/make_main_notebook.py
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
md("""# The Naylor Model — Master Notebook (V10)

Base-stealing skill that sprint speed hides — the *Josh Naylor* trait: a runner who steals
better than ~99% of MLB without the wheels.

**Runs without network.** Reads `Data/Raw_Season.csv`, `Data/Raw_Attempts.csv`, the results in
`Output/Results/`, and the figures/tables in `Output/`. The Statcast pull lives in
`Scripts/v7_explore.py` (network-bound) and is not required here.

| § | What |
|---|---|
| 1 | Raw data — per-attempt (the primary grain) + season |
| 2 | **Model A — per-attempt XGBoost (~11k rows, AUC ~0.74) — THE model** |
| 3 | Slow-Steal Skill leaderboard (SSSI) |
| 4 | GLM — the steal-success equation (coaching levers) |
| 5 | xSB — speed-vs-production quadrant |
| 6 | Statcast-style leaderboards |
| 7 | Build the report |
"""),

md("## Setup"),
code("""\
import sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

ROOT = Path().resolve()
while not (ROOT / "Output").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
SCRIPTS = ROOT / "Scripts"
DATA    = ROOT / "Data"
RESULTS = ROOT / "Output" / "Results"
FIGS    = ROOT / "Output" / "Figures"
TABLES  = ROOT / "Output" / "Tables"
sys.path.insert(0, str(SCRIPTS))
print("Repo root:", ROOT)
"""),

md("""## 1 — Raw data

`Data/Raw_Season.csv` is the runner-season master (one row per qualified runner-season, every
feature + SSSI + team). `Data/Raw_Attempts.csv` is the per-attempt grain the AUC model uses."""),
code("""\
season = pd.read_csv(DATA / "Raw_Season.csv")
attempts = pd.read_csv(DATA / "Raw_Attempts.csv")
print(f"season:   {len(season)} runner-seasons × {season.shape[1]} cols")
print(f"attempts: {len(attempts)} attempts × {attempts.shape[1]} cols")
season[["player_name", "season", "team", "sprint_speed", "jump_time",
        "real_sb_pct", "sb_attempts", "SSSI_v7"]].head()
"""),

md("""## 2 — Model A: the per-attempt model (THE model)

**This is the primary analysis.** It models the ~11,169 individual tracked attempts (one row per
steal), not 673 season aggregates — the grain that actually decides a steal. CV AUC ~0.74, driven
by the per-pitch lead distances. `model_perattempt.main()` runs 5-fold CV (no network), writes the
AUC + importance, and refreshes `Fig_AUC.png` / `Fig_Importance.png`. Leakage-checked; gradient
boosting wins on tabular data this size, so no deep learning."""),
code("""\
import model_perattempt
pa = model_perattempt.main()
display(pa)
"""),
code("""display(Image(filename=str(FIGS / "Fig_AUC.png")))
display(Image(filename=str(FIGS / "Fig_Importance.png")))"""),

md("""## 3 — Slow-Steal Skill (SSSI)

A descriptive season-level composite that surfaces the Naylor/Soto archetype — elite-performing
slow runners. Naylor and Soto were held out when the weights were fit, so their ranking is
out-of-sample."""),
code("""\
top = season.sort_values("SSSI_v7", ascending=False).head(15)
top[["player_name", "season", "team", "sprint_speed", "real_sb_pct", "SSSI_v7"]].reset_index(drop=True)
"""),

md("""## 4 — GLM: the steal-success equation

The interpretable season-level model (not a predictor). Each lever's weight is the change in
steal-success rate for a +1-SD improvement — the basis for the report's equation figure and
coaching levers."""),
code("""\
glm = pd.read_csv(RESULTS / "DF_v7_GLM_PlainEnglish.csv")
display(glm)
display(Image(filename=str(FIGS / "Fig_Equation.png")))
"""),

md("""## 5 — xSB: speed-vs-production quadrant

Realized Burners, Untapped Wheels, Crafty Technicians (the Naylor/Soto archetype), and Stationary."""),
code("""\
xsb = pd.read_csv(RESULTS / "DF_v7_xSB_Outcome.csv")
print(xsb["quadrant"].value_counts())
display(Image(filename=str(FIGS / "Fig_xSB_Quadrant.png")))
"""),

md("""## 6 — Statcast-style leaderboards

The marquee outputs: headshot + team logo + supporting stats + a heat-colored headline column."""),
code("""\
for t in ["Slow_Steal_Skill.png", "Blueprint_Conversion.png", "Ground_Covered.png"]:
    p = TABLES / t
    if p.exists():
        display(Image(filename=str(p)))
"""),

md("""## 7 — Build the report

Regenerates the V10 main report (repo root) and the Technical Appendix (`Reports/`)."""),
code("""\
import importlib.util, subprocess
if importlib.util.find_spec("docx") is None:
    print("python-docx not installed in this kernel — run: python3 Scripts/build_report.py")
else:
    r = subprocess.run([sys.executable, str(SCRIPTS / "build_report.py")],
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:])
"""),

md("""---
### Regenerating from scratch
- Assets (one-time network): `python3 Scripts/fetch_assets.py`
- Consolidate raw: `python3 Scripts/consolidate_raw.py`
- Statcast tables: `python3 Scripts/statcast_tables.py`
- Full Statcast pull (network): `python3 Scripts/v7_explore.py`
See [`AUC_Roadmap.md`](AUC_Roadmap.md) for how to push AUC higher."""),
]

nbf.write(nb, str(OUT))
print(f"Wrote {OUT}")
