# The Naylor Model

> ### Open this first
> | If you want to… | Open |
> |---|---|
> | **Read the findings** (coaches / R&D) | **[`Naylor_Model_v8_Report.docx`](Naylor_Model_v8_Report.docx)** |
> | **Run the model** end-to-end | **[`Naylor_Model.ipynb`](Naylor_Model.ipynb)** |
> | **See the data** | `Naylor_Model_Data.csv` (runners) · `Naylor_Model_Results.csv` (model results) |
> | **Improve the AUC** | [`AUC_Roadmap.md`](AUC_Roadmap.md) |
>
> Everything else is plumbing: `scripts/` (code) · `data/` (working files) · `Figures/` · `Reports/`
> (appendix + archives) · `Computer Vision/` (CV pilot) · `Previous Versions/`.

---

José Caballero led MLB in net stolen bases in 2025 running a quarter-second slower than Chandler Simpson. Shohei Ohtani was the second most productive base-stealer in 2024 despite being 0.14 seconds slower than Elly De La Cruz. Most strikingly, Josh Naylor stole 20 bases above average at 93.8% success while running slower than 97% of the league. Sprint speed is the most intuitive base-stealing metric — it is not the most essential one.

What separates these runners is technique — and technique is coachable precisely because it reflects what a player has learned to do with their body, not what their body is built to do. Sprint speed is structural. Primary lead distance, secondary lead timing, and first-step burst off the pitcher's first move are behavioral patterns that haven't permanently locked in, which means they can be shifted.

Sprint biomechanics research points to three specific targets every baseball player can develop regardless of raw speed: shorter ground contact time, more distance covered in the first five-foot window from the pitcher's first move, and earlier recognition of delivery cues. These aren't elite-only adaptations — they are timing and sequencing refinements accessible to any MLB-level runner. Naylor's edge isn't a physical gift; it's that he optimizes all three within a body most evaluators would write off.

That's why specificity with the biomechanics suite matters. Knowing a runner has a "slow jump" isn't actionable. Knowing exactly where in the ground contact phase they're losing time, and at which keyframe their secondary lead stalls, is. The more precisely the metric targets the problem, the more directly the coaching intervention follows.

---

## Navigation

| | |
|---|---|
| ⭐ **[Main Report — v8 (DOCX)](Naylor_Model_v8_Report.docx)** | **Start here.** Applied BLUF report for MLB R&D + coaches — the **steal-success equation** (what each trainable skill is worth per +1 SD, in points *and* extra bags), the SSSI skill matrix, the speed-vs-production quadrant, and a **2025 coaching target board** (green-light vs. technique-fix). Plain-English, self-contained pages, built around the *trait* — a slow runner who steals better than ~99% of MLB — not any one player |
| 🧾 **[v8 Technical Appendix (DOCX)](Reports/Naylor_Model_v8_Technical_Appendix.docx)** | Full model detail for auditors — Models A/B/C + de-leaked AUC, complete GLM weight table, full SSSI Top 25, xSB leaderboards, and the Blueprint Conversion Score with **team logos** |
| 📄 **[Comprehensive Report — v7 (DOCX)](Reports/Naylor_Model_v7_Report.docx)** | Prior all-in-one v7 report — Models A/B/C, AUC, GLM, SSSI, xSB **color-coded SD tables**, plus the full Blueprint Conversion Score section (§5.3 archetype profile, per-season Top 25 with **team logos**) |
| 📓 **[Master Notebook](Naylor_Model.ipynb)** | End-to-end pipeline in one notebook — data → SSSI → tuned-XGBoost Model B → GLM equation → xSB → report |
| 🗺️ **[AUC Roadmap](AUC_Roadmap.md)** | How to push the model's AUC higher — the untapped matchup variables, ranked |
| 📖 **[Variable Glossary](Reports/Variable_Glossary.pdf)** | Plain-English reference, trimmed 80-20 — every metric on a compact card (units, tiers, example, why it matters) + core model discussion (12 pp) |
| 🖼️ **[Figures](Figures/)** | All charts and visualizations |
| 📊 **[data/](data/)** | Working CSVs (DF_v7_*, benchmark, tuning) + xlsx workbooks; curated summaries are the two root CSVs |
| 🧠 **[Computer Vision](Computer%20Vision/)** | All CV analysis — Statcast Analysis Core (Blueprint model) + the CV delivery-time pilot |
| 🗂️ **[Previous Versions](Previous%20Versions/)** | Archived v3, v4, v5, v6 pipelines and outputs |

---

## Key Results

### The Steal-Success Equation — What Each Trainable Skill Is Worth (per +1 SD)
Every lever's coefficient (β), its points of success-rate change for a one-standard-deviation gain, and the plain-English translation into **extra bags over a 20-attempt season**. Bars are color-keyed: green = trainable, gray = opponent/context, orange = raw speed (barely moves the needle).
![Steal-Success Equation](Figures/Fig_v8_Equation.png)

### Expected SB Outcome (xSB) — Speed vs Production Quadrant
![xSB Quadrant](Figures/Fig_v8_xSB_Quadrant.png)

### Blueprint Conversion Score — Top 25 Per Season (2023–2026)
![BCS Top 25 by Year](Figures/Fig_v8_BCS_ByYear.png)

### Ground Covered Beyond Speed-Expected — Top 25 Per Season (2023–2026)
![Ground Covered by Year](Figures/Fig_v8_GroundCovered_ByYear.png)

### Feature Importance — Pre vs Post 2023
![Feature Importance](Figures/Fig_v7_Importance_PrePost.png)

### Model Accuracy (AUC)
![AUC](Figures/Fig_v7_AUC.png)

### GLM Weight Table — What Actually Moves the Needle
![GLM](Figures/Fig_v7_GLM_PlainEnglish.png)

---

## How It Works

The core signal is the **SB Residual**: a runner's actual success rate minus the rate their sprint speed alone would predict. Positive means they outperform their speed peers. The model is built on real Statcast data — sprint speed, 5-ft running splits (0–90 ft), catcher pop times, pitcher running-game suppression, and season SB/CS records from 2015–2026.

### Key Metrics

| Metric | What it captures |
|---|---|
| `sprint_speed` | Top running speed (ft/s) — structural baseline |
| `speed_capped` | Sprint speed capped at 28 ft/s — marginal benefit vanishes above this |
| `jump_time` | Time to cover the first 30 ft — first-step burst, independent of top speed |
| `accel_gap` | Jump time percentile minus sprint speed percentile — positive = faster off the line than top speed implies (the Naylor archetype) |
| `accel_topspeed_premium` | **(v7)** How few feet a runner needs to reach top speed, speed-adjusted — a small runway at high speed is a premium |
| `sb_residual` | Real SB% minus speed-expected SB% — ground-truth speed-adjusted steal skill |
| `lead_gain` | Distance gained in secondary lead — a coachable behavioral pattern |
| `xsb_outcome` | **(v7)** `z(net SB above avg) + z(sprint speed)` — combined speed-and-production lens; high = fast AND productive |
| `sb_potential_gap` | **(v7)** `z(sprint) − z(net SB)` — positive = fast but under-stealing (untapped, coachable); negative = over-performs speed |
| `avg_pop_faced` | Catcher pop time in this runner's matchups — battery context |
| `avg_pickoff_rate_faced` | Pitcher hold frequency — suppression context |

### The Steal-Success Equation (v8 — applied)

A logistic model turns the metrics above into one readable equation:

> **chance of a successful steal = baseline (≈ 78%) + Σ ( weight × how far above average the runner is, in SDs )**

Each lever's weight (β) is reported as the **points of success-rate change for a +1-SD gain**, then translated into **net bags over a typical 20-attempt season** so the units never leave plain English (`+5 pp ≈ one extra steal and one fewer caught`). Three *trainable* levers dominate — ground covered after the pitcher commits (**+25 pts ≈ +5 bags**), a quicker jump (**+17 pts ≈ +3 bags**), and reaching top speed in fewer feet (**+10 pts ≈ +2 bags**) — while raw top speed barely moves the needle. See `Figures/Fig_v8_Equation.png`.

### 2025 Coaching Target Board (v8 — applied)

The equation is turned into next steps via two honest, separated tracks — so a caught-prone runner is never simply told to "run more":

| Track | Who | The move |
|---|---|---|
| **Green-light** | Fast, efficient runners (≥ 80% success) who don't run enough | Just let them run — projected extra bags at their own rate |
| **Technique-fix** | High-volume runners caught too often (< 70% success) | Drill the *one* weakest trainable lever — projected success-rate gain |

The boards are priority rankings, not forecasts: *if unleashed* holds a runner at his own 2025 success rate and a modest ~20-attempt volume.

### The SSSI — Slow-Steal Skill Index

A weighted composite of nine z-scored features (v7 adds the Accel→Top-Speed Premium) designed to surface the Naylor/Soto archetype: elite-performing slow runners. Weights were optimised on 80% of runners with Naylor and Soto held out entirely — their ranking is a genuine out-of-sample result.

| Rank | Player | Season | SSSI |
|---|---|---|---|
| 1 | Josh Naylor | 2025 | +1.90 |
| 2 | Josh Naylor | 2026 | +1.84 |
| 3 | Freddie Freeman | 2024 | +1.71 |
| 5 | Juan Soto | 2025 | +1.43 |

### xSB — Expected Stolen-Base Outcome (v7)

A **complementary** lens to the SSSI. Where the SSSI surfaces slow-but-skilled stealers, **xSB = `z(net SB above avg) + z(sprint speed)`** surfaces the high-ceiling runners who are both fast *and* productive. The companion **`sb_potential_gap` = `z(sprint) − z(net SB)`** splits the league into four quadrants:

| Quadrant | Read |
|---|---|
| **Realized Burner** | Fast and productive — the complete package (e.g. Elly De La Cruz) |
| **Untapped Wheels** | Fast but under-stealing — coaching targets, split into *green-light* (efficient, just let them run) and *technique-fix* (caught too often, drill mechanics first) |
| **Crafty Technician** | Productive despite modest speed — the Naylor / Soto archetype |
| **Stationary** | Neither speed nor steal production |

xSB is descriptive, not predictive — it is deliberately kept out of the GBM (z(SB) would leak the outcome) and out of the SSSI composite.

### Blueprint Conversion Score — Top 5 All-Time (2023–2026)

| Rank | Player | Team | Season | BCS |
|---|---|---|---|---|
| 1 | Ryan McMahon | NYY | 2026 † | +8.05 |
| 2 | Agustín Ramírez | MIA | 2025 | +6.86 |
| 3 | Paul Goldschmidt | STL | 2024 | +4.59 |
| 6 | Josh Naylor | SEA | 2025 | +4.01 |
| 11 | Juan Soto | NYM | 2025 | +3.45 |

*† 2026 partial season (~1/3 complete, May 2026); min 3 tracked Statcast attempts.*

### Models

| Model | Unit | AUC | Purpose |
|---|---|---|---|
| **Model A** (per-attempt XGBoost) | Individual attempt | **0.739** | Strongest predictor — does *this* steal succeed |
| Model B (season XGBoost, Bayesian-tuned) | Runner-season | 0.624 (full) · 0.665 (post-23) | Ranks season-long skill |
| GLM | Runner-season | — | Interpretable weight table |

**v9 — the per-attempt model (AUC 0.739).** Moving from 673 season aggregates to the **~10,400
individual tracked attempts** (Statcast leads cache, `scripts/model_perattempt.py`) lifts CV AUC to
**0.739** — into the target range. The driver is the per-pitch **lead distances** (how much ground the
runner actually covered on that attempt), which is exactly this project's thesis. Leakage-checked: no
outcome-derived columns; catcher/pitcher tendencies are out-of-fold encoded (and, tellingly, *don't*
help — the leads carry the signal).

**Why not deep learning?** At ~10k rows and a dozen tabular features, gradient boosting (XGBoost/
CatBoost) is the right tool — neural nets need far more data and overfit here. Honest ceiling on public
data is ~0.74–0.78; reaching it further needs richer per-pitch matchup data (pitcher handedness, pitch
type), see [`AUC_Roadmap.md`](AUC_Roadmap.md).

**v8 model update (season model).** Model B was upgraded from a gradient-boosting classifier to a
Bayesian-tuned XGBoost; on the same de-leaked data, season CV AUC rose 0.589 → **0.624** overall and
0.588 → **0.665** post-2023. A six-classifier benchmark and the tuning history live in
`Naylor_Model_Results.csv`. The SSSI rankings and the steal-success equation are unchanged.

> **AUC caveat (de-leaking).** Earlier versions (v4–v6) reported AUCs of ~0.66–0.70, but those
> runs carried **duplicate runner-season rows** — repeated Statcast split measurements for the same
> player-season — that leaked across cross-validation folds and inflated the score. v7 averaged those
> duplicate splits into one row per runner-season, removing the leak; v8 keeps that fix and swaps in the
> tuned XGBoost. The de-leaked AUCs are **lower but honest** — not a regression. The historical bars in
> `Fig_v7_AUC.png` are kept for context only and are not a fair comparison.

---

## How to Run

```bash
# Refresh the headline model artifacts (no network — reads the cached feature CSV)
python3 scripts/model_xgb.py     # tuned-XGBoost Model B → AUC + importance CSVs, figures, 2 root CSVs
python3 scripts/build_v8_report.py  # ⭐ main report (root) + Technical Appendix (Reports/) — reads data/

# Compare models / re-tune (no network)
python3 scripts/benchmark_models.py # 6-classifier AUC bake-off → data/DF_benchmark_AUC.csv
python3 scripts/tune_xgboost.py     # Optuna Bayesian HPO → data/DF_xgb_tuned_params.csv

# Full data pipeline (requires network — pybaseball / Savant / MLB API)
python3 scripts/v7_explore.py    # SSSI, Model B, GLM, xSB, figures → data/, Figures/
python3 scripts/build_v7_report.py  # prior comprehensive v7 DOCX → Reports/
python3 scripts/write_glossary.py   # Variable Glossary → Reports/

# Blueprint + CV pipelines (Jupyter notebooks under Computer Vision/)
jupyter notebook "Computer Vision/notebooks/Data Pipeline.ipynb"
jupyter notebook "Computer Vision/notebooks/Blueprint Analysis.ipynb"
```

---

## Repository Structure

The repo root is intentionally minimal — the report, one notebook, and two curated CSVs:

```
The-Naylor-Model/
├── Naylor_Model.ipynb           ← ⭐ master notebook (data → SSSI → tuned-XGBoost Model B → GLM → xSB → report)
├── Naylor_Model_v8_Report.docx  ← ⭐ the applied BLUF report (XGBoost-updated)
├── Naylor_Model_Data.csv        ← runner-season master (features + SSSI rankings)
├── Naylor_Model_Results.csv     ← model results: GLM weights + AUC by era + benchmark + tuned params
├── AUC_Roadmap.md               ← how to push AUC higher (matchup-variable roadmap)
├── README.md
├── scripts/        ← v7_explore.py, model_xgb.py, build_v8_report.py, build_v7_report.py,
│                     write_glossary.py, benchmark_models.py, tune_xgboost.py, make_main_notebook.py
├── Figures/        ← all output PNGs (incl. per-season BCS figures + logos/)
├── data/           ← working CSVs (DF_v7_*, benchmark, tuning) + Naylor Blueprint.xlsx, v7 Model.xlsx
├── Reports/        ← v8 Technical Appendix + v7 report, glossary PDFs
├── Computer Vision/← all CV analysis (notebooks/ code/ data/ archive/) — see its own README
└── Previous Versions/  ← v3–v6 pipelines and outputs
```

---

## Data Sources

- Baseball Savant: sprint speed, running splits, catcher pop times, pitcher running-game leaderboard, base-stealing run value
- MLB Stats API: season SB/CS records (2015–2026)
- Statcast pitch-level feed: per-pitch runner context, battery matchups
