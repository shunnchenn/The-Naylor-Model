"""
make_cv_notebooks.py  —  Generate the two CV Jupyter notebooks from source .py files.
Run once: python3 "Computer Vision/make_cv_notebooks.py"
"""
import nbformat as nbf
from pathlib import Path

HERE    = Path(__file__).parent
PIPE    = HERE / "CV Detection Pipeline"
CORE    = HERE / "Statcast Analysis Core"
ARCH    = HERE / "archive" / "Archived Runner Runs"

def md(text):  return nbf.v4.new_markdown_cell(text)
def code(src): return nbf.v4.new_code_cell(src)
def read(p):   return Path(p).read_text()

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 1 — CV Detection Pipeline
# ─────────────────────────────────────────────────────────────────────────────
nb1 = nbf.v4.new_notebook()
nb1.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

nb1.cells = [

md("""# CV Detection Pipeline — The Naylor Model

Measures **pitcher delivery time** (first move → ball release) from Statcast broadcast clips
using YOLOv8-pose, then combines it with Savant per-attempt lead distances to compute a
**runner closing-velocity metric** (`avg_velocity_ftps = gain_to_release_ft / delivery_s`).

**Pipeline:**
1. Fetch per-attempt leads from Savant  (`fetch_leads.py`)
2. Download broadcast clips             (`fetch_clips.py`)
3. Extract delivery times via YOLO      (`extract_delivery.py`)
4. Build the per-attempt delivery table (`build_delivery.py`)
5. Aggregate by pitcher                 (`aggregate_delivery.py`)
6. Compute runner velocity metric       (`delivery_velocity.py`)
7. Statcast reference cross-check       (`statcast_ref.py`)
8. Evaluate accuracy vs manual labels   (`evaluate.py`)

**Tip:** Use `run_runner.py` to run steps 1–3 in one command.
**Requires:** `ultralytics` (YOLOv8), `opencv-python`, `requests`, `pybaseball`.
Steps 2–3 require network access (Savant clip downloads).
"""),

md("## Setup"),

code("""\
import sys, subprocess
from pathlib import Path

REPO = Path().resolve()
while not (REPO / "Figures").exists() and REPO.parent != REPO:
    REPO = REPO.parent

CV    = REPO / "Computer Vision"
PIPE  = CV / "CV Detection Pipeline"
CORE  = CV / "Statcast Analysis Core"
DATA  = REPO / "data"

for p in [str(PIPE), str(CORE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

print("Repo root:", REPO)
print("Pipeline: ", PIPE)
print("Core:     ", CORE)
"""),

md("""## Configuration

Set the runner and year here — all steps below use these variables."""),

code("""\
# ── Configure your runner ─────────────────────────────────────────────────────
RUNNER_ID   = 647304        # MLB player_id  (647304 = Josh Naylor)
RUNNER_NAME = "naylor"      # lowercase tag — used in filenames
YEAR        = 2025

OUT_DIR  = CV / "archive" / "Archived Runner Runs" / f"{RUNNER_NAME.capitalize()}_{YEAR}"
LEADS    = OUT_DIR / f"{RUNNER_NAME}{YEAR}_leads.csv"
TARGETS  = OUT_DIR / f"{RUNNER_NAME}{YEAR}_targets.csv"
RESULTS  = OUT_DIR / "pilot_results.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Runner: {RUNNER_NAME} ({RUNNER_ID})  Year: {YEAR}")
print(f"Output: {OUT_DIR}")
"""),

md("## Step 1 — Fetch Per-Attempt Leads from Savant"),

code(f"""\
# fetch_leads.py — pulls gain_to_release_ft per attempt from Baseball Savant
# Source:
{read(CORE / "fetch_leads.py")}
"""),

code("""\
sys.path.insert(0, str(CORE))   # fetch_leads lives in Statcast Analysis Core
from fetch_leads import fetch_leads
import pandas as pd

rows = fetch_leads(RUNNER_ID, YEAR, runner_name=RUNNER_NAME,
                   out=str(LEADS), targets_out=str(TARGETS))
df_leads = pd.DataFrame(rows)
print(f"Fetched {len(df_leads)} attempts for {RUNNER_NAME} {YEAR}")
df_leads.head()
"""),

md("## Step 2 — Download Broadcast Clips from Savant"),

code(f"""\
# fetch_clips.py — downloads mp4 clips via Baseball Savant's Film Room
# Source:
{read(PIPE / "fetch_clips.py")}
"""),

code("""\
# Download clips for the targets we just fetched
# NOTE: requires network access; produces <OUT_DIR>/clips/*.mp4
result = subprocess.run([
    sys.executable, str(PIPE / "fetch_clips.py"),
    "--targets", str(TARGETS),
    "--out-dir", str(OUT_DIR),
    "--download",
], capture_output=True, text=True)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
"""),

md("""## Step 3 — Extract Delivery Times (YOLOv8-pose)

The detector reads each clip and measures:
- **first_movement_time** — when the pitcher initiates from the stretch
- **ball_release_time** — when the ball leaves the hand
- **delivery_s** = `ball_release_time − first_movement_time`

⚠️ This step is compute-intensive (~10–30 s per clip on MPS/CPU).
Pass `--no-qa` to skip the annotated QA videos and run ~2× faster."""),

code(f"""\
# extract_delivery.py — YOLOv8-pose delivery-time detector (910 lines)
# Source:
{read(PIPE / "extract_delivery.py")}
"""),

code("""\
clips_dir = OUT_DIR / "clips"
result = subprocess.run([
    sys.executable, str(PIPE / "extract_delivery.py"),
    "--clips-dir", str(clips_dir),
    "--meta", str(OUT_DIR / "clips_meta.csv"),
    "--out", str(RESULTS),
    "--no-qa",               # remove to generate annotated QA videos
    "--first-move", "heel",  # alternatives: disp
], capture_output=True, text=True)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1500:])
"""),

md("## Step 4 — Build Per-Attempt Delivery Table"),

code(f"""\
# build_delivery.py — joins detector output with leads CSV
# Source:
{read(PIPE / "build_delivery.py")}
"""),

code("""\
result = subprocess.run([
    sys.executable, str(PIPE / "build_delivery.py"),
    "--dir", str(OUT_DIR),
    "--leads", str(LEADS),
    "--year", str(YEAR),
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
"""),

md("## Step 5 — Aggregate Delivery Times by Pitcher"),

code(f"""\
# aggregate_delivery.py — per-pitcher delivery time under stealing conditions
# Source:
{read(PIPE / "aggregate_delivery.py")}
"""),

code("""\
from aggregate_delivery import main as aggregate_main
aggregate_main()
"""),

md("## Step 6 — Compute Runner Velocity Metric"),

code(f"""\
# delivery_velocity.py — avg_velocity_ftps = gain_to_release_ft / delivery_s
# Source:
{read(PIPE / "delivery_velocity.py")}
"""),

code("""\
result = subprocess.run([
    sys.executable, str(PIPE / "delivery_velocity.py"),
    "--dir", str(OUT_DIR),
    "--leads", str(LEADS),
    "--year", str(YEAR),
    "--runner", RUNNER_NAME.capitalize(),
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])

import pandas as pd
vel_csv = OUT_DIR / f"delivery_velocity_{YEAR}.csv"
if vel_csv.exists():
    display(pd.read_csv(vel_csv).head(10))
"""),

md("## Step 7 — Statcast Reference Cross-Check"),

code(f"""\
# statcast_ref.py — compares CV delivery metric against Savant baselines
# Source:
{read(PIPE / "statcast_ref.py")}
"""),

code("""\
result = subprocess.run([
    sys.executable, str(PIPE / "statcast_ref.py"),
    "--dir", str(OUT_DIR),
    "--leads", str(LEADS),
    "--year", str(YEAR),
    "--runner-id", str(RUNNER_ID),
    "--runner", RUNNER_NAME.capitalize(),
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
"""),

md("## Step 8 — Evaluate Accuracy vs Manual Labels"),

code(f"""\
# evaluate.py — accuracy / repeatability / coverage + go-no-go verdict
# Gate: release error ≤ 2 frames (±66 ms), within-pitcher std ≤ 0.10 s, coverage ≥ 70%
# Source:
{read(PIPE / "evaluate.py")}
"""),

code("""\
result = subprocess.run([
    sys.executable, str(PIPE / "evaluate.py"),
], capture_output=True, text=True, cwd=str(OUT_DIR))
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
"""),

md("""## Shortcut — run_runner.py (Steps 1–3 in one command)

Instead of running steps 1–3 individually, `run_runner.py` chains them:
```bash
python3 "CV Detection Pipeline/run_runner.py" 647304 naylor 2025
python3 "CV Detection Pipeline/run_runner.py" 665742 soto   2025 --no-detect
```
"""),

code(f"""\
# run_runner.py — one-command orchestrator for leads → clips → delivery
# Source:
{read(PIPE / "run_runner.py")}
"""),

md("""## build_pitcher_delivery_table.py — league-wide pitcher table"""),

code(f"""\
# build_pitcher_delivery_table.py — aggregates all runner-season pilot_results
# into a single pitcher delivery table for the Naylor Model
# Source:
{read(PIPE / "build_pitcher_delivery_table.py")}
"""),

]

out1 = HERE / "CV Detection Pipeline" / "CV_Detection_Pipeline.ipynb"
nbf.write(nb1, str(out1))
print(f"Wrote {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 2 — CV Attempt Model (pooled multi-runner evaluation)
# ─────────────────────────────────────────────────────────────────────────────
nb2 = nbf.v4.new_notebook()
nb2.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

# Read archived runner files
def arch(runner, year, filename):
    p = ARCH / f"{runner}_{year}" / filename
    return read(p) if p.exists() else f"# {p} not found"

nb2.cells = [

md("""# CV Attempt Model — Does Delivery Velocity Add Discrimination?

Pools all per-attempt data across every runner we have CV deliveries for and asks:
**does the CV-derived velocity metric (`avg_velocity_ftps`) add prediction power over
Statcast lead distances alone?**

Three feature blocks compared with leave-one-out CV AUC:
- **BASE** — `lead_at_firstmove_ft`, `gain_to_release_ft`, `lead_at_release_ft`
- **+DELIVERY** — BASE + `delivery_s`
- **+VELOCITY** — BASE + `delivery_s` + `avg_velocity_ftps`

The headline is the delta: `AUC(+VELOCITY) − AUC(BASE)`.

> ⚠️ **Small-sample caveat:** with only ~3–5 CS per runner, the classification AUC is
> high-variance. Treat it as a proof-of-harness. The **univariate separation table** and
> **run-value regression** are the more stable reads.

**Covered here:**
- Archived per-runner runs: Naylor 2025 · Naylor 2026 · Soto 2025 · Soto 2026
- Pooled multi-runner model (`attempt_model.py`)
"""),

md("## Setup"),

code("""\
import sys, csv, os
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path().resolve()
while not (REPO / "Figures").exists() and REPO.parent != REPO:
    REPO = REPO.parent

CV   = REPO / "Computer Vision"
ARCH = CV / "archive" / "Archived Runner Runs"
PIPE = CV / "CV Detection Pipeline"

if str(PIPE) not in sys.path:
    sys.path.insert(0, str(PIPE))

print("Repo root:", REPO)
"""),

md("## Archived Runner Run: Naylor 2025"),

md("### statcast_ref — Naylor 2025"),
code(f"{arch('Naylor', 2025, 'statcast_ref_2025.py')}"),

md("### delivery_velocity — Naylor 2025"),
code(f"{arch('Naylor', 2025, 'delivery_velocity_2025.py')}"),

md("### build_delivery — Naylor 2025"),
code(f"{arch('Naylor', 2025, 'build_delivery_2025.py')}"),

md("""\
### Load Naylor 2025 results"""),
code("""\
naylor25 = ARCH / "Naylor_2025" / "delivery_velocity_2025.csv"
if naylor25.exists():
    df = pd.read_csv(naylor25)
    print(f"Naylor 2025: {len(df)} attempts")
    display(df[["date","pitcher_name","result","lead_at_firstmove_ft",
                "gain_to_release_ft","delivery_s","avg_velocity_ftps"]].head(10))
else:
    print(f"Not found: {naylor25}  (run the CV Detection Pipeline first)")
"""),

md("## Archived Runner Run: Naylor 2026"),

md("### statcast_ref — Naylor 2026"),
code(f"{arch('Naylor', 2026, 'statcast_ref_2026.py')}"),

md("### delivery_velocity — Naylor 2026"),
code(f"{arch('Naylor', 2026, 'delivery_velocity_2026.py')}"),

code("""\
naylor26 = ARCH / "Naylor_2026" / "delivery_velocity_2026.csv"
if naylor26.exists():
    df = pd.read_csv(naylor26)
    print(f"Naylor 2026: {len(df)} attempts")
    display(df[["date","pitcher_name","result","gain_to_release_ft",
                "delivery_s","avg_velocity_ftps"]].head(10))
else:
    print(f"Not found: {naylor26}")
"""),

md("## Archived Runner Run: Soto 2025"),

md("### statcast_ref — Soto 2025"),
code(f"{arch('Soto', 2025, 'statcast_ref_2025.py')}"),

md("### delivery_velocity — Soto 2025"),
code(f"{arch('Soto', 2025, 'delivery_velocity_2025.py')}"),

code("""\
soto25 = ARCH / "Soto_2025" / "delivery_velocity_2025.csv"
if soto25.exists():
    df = pd.read_csv(soto25)
    print(f"Soto 2025: {len(df)} attempts")
    display(df[["date","pitcher_name","result","gain_to_release_ft",
                "delivery_s","avg_velocity_ftps"]].head(10))
else:
    print(f"Not found: {soto25}")
"""),

md("## Archived Runner Run: Soto 2026"),

md("### delivery_velocity — Soto 2026"),
code(f"{arch('Soto', 2026, 'delivery_velocity_2026.py')}"),

md("### statcast_ref — Soto 2026"),
code(f"{arch('Soto', 2026, 'statcast_ref_2026.py')}"),

md("""## Pooled Multi-Runner Attempt Model

Pools all available runner-seasons, then evaluates whether adding CV delivery velocity
improves discrimination over Statcast lead features alone.
"""),

code(f"""\
# attempt_model.py — pooled LOO-CV AUC comparison across feature blocks
{read(ARCH / "Naylor_model" / "attempt_model.py")}
"""),

code("""\
import subprocess, sys
result = subprocess.run(
    [sys.executable, str(ARCH / "Naylor_model" / "attempt_model.py")],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
"""),

md("""## Summary — What the CV Metric Adds

| Feature block | LOO-CV AUC | Δ vs BASE |
|---|---|---|
| BASE (leads only) | — | — |
| +DELIVERY | — | — |
| +VELOCITY (CV) | — | — |

Fill in from the output above. Key reads:
- **Univariate AUC** of `avg_velocity_ftps` vs SB/CS — most stable signal
- **run_value correlation** — does closing velocity track run value?
- **Classification delta** — directional, high-variance at small n

The CV metric's value is not (yet) in the AUC bump — it's in having a **per-pitch**
delivery-time estimate that replaces the league constant `LEAGUE_PITCHER_TTP = 1.30 s`
with a pitcher-specific measured value. That matters for the Naylor archetype: a slow
runner needs a slow-delivery pitcher; knowing *which* pitchers are genuinely slow (vs.
just having a weak arm) is a matchup edge.
"""),

]

out2 = ARCH / "CV_Attempt_Model.ipynb"
nbf.write(nb2, str(out2))
print(f"Wrote {out2}")
