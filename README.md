# The Naylor Model

José Caballero led MLB in net stolen bases in 2025 running a quarter-second slower than Chandler Simpson. Shohei Ohtani was the second most productive base-stealer in 2024 despite being 0.14 seconds slower than Elly De La Cruz. Most strikingly, Josh Naylor stole 20 bases above average at 93.8% success while running slower than 97% of the league. Sprint speed is the most intuitive base-stealing metric — it is not the most essential one.

What separates these runners is technique — and technique is coachable precisely because it reflects what a player has learned to do with their body, not what their body is built to do. Sprint speed is structural. Primary lead distance, secondary lead timing, and first-step burst off the pitcher's first move are behavioral patterns that haven't permanently locked in, which means they can be shifted.

Sprint biomechanics research points to three specific targets every baseball player can develop regardless of raw speed: shorter ground contact time, more distance covered in the first five-foot window from the pitcher's first move, and earlier recognition of delivery cues. These aren't elite-only adaptations — they are timing and sequencing refinements accessible to any MLB-level runner. Naylor's edge isn't a physical gift; it's that he optimizes all three within a body most evaluators would write off.

That's why specificity with the biomechanics suite matters. Knowing a runner has a "slow jump" isn't actionable. Knowing exactly where in the ground contact phase they're losing time, and at which keyframe their secondary lead stalls, is. The more precisely the metric targets the problem, the more directly the coaching intervention follows.

---

## Navigation

| | |
|---|---|
| 📄 **[Full Report](reports/Naylor_Model_v6_Report.pdf)** | Complete v6 analysis — models, AUC, SSSI leaderboard, Naylor/Soto profile |
| 📖 **[Variable Glossary](reports/Variable_Glossary.pdf)** | Plain-English reference for every metric. Tier charts, real examples, full model discussion |
| 🖼️ **[Figures](figures/)** | All charts and visualizations |
| 📊 **[Data](data/)** | All output CSVs — leaderboards, SSSI rankings, model results |
| 🗂️ **[Previous Versions](Previous%20Versions/)** | Archived v3, v4, v5 pipelines and outputs |

---

## Key Results

### Naylor & Soto Profile
![Naylor Soto Profile](figures/Fig_v6_NaylorSoto_Profile.png)

### Feature Importance — Pre vs Post 2023
![Feature Importance](figures/Fig_v6_Importance_PrePost.png)

### Model Accuracy (AUC)
![AUC](figures/Fig_v6_AUC.png)

### GLM Weight Table — What Actually Moves the Needle
![GLM](figures/Fig_v6_GLM_PlainEnglish.png)

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
| `sb_residual` | Real SB% minus speed-expected SB% — ground-truth speed-adjusted steal skill |
| `lead_gain` | Distance gained in secondary lead — a coachable behavioral pattern |
| `avg_pop_faced` | Catcher pop time in this runner's matchups — battery context |
| `avg_pickoff_rate_faced` | Pitcher hold frequency — suppression context |

### The SSSI — Slow-Steal Skill Index

A weighted composite of eight z-scored features designed to surface the Naylor/Soto archetype: elite-performing slow runners. Weights were optimised on 80% of runners with Naylor and Soto held out entirely — their ranking is a genuine out-of-sample result.

| Rank | Player | Season | SSSI |
|---|---|---|---|
| 1 | Josh Naylor | 2025 | +1.90 |
| 2 | Josh Naylor | 2026 | +1.84 |
| 3 | Freddie Freeman | 2024 | +1.71 |
| 5 | Juan Soto | 2025 | +1.43 |

### Models

| Model | Unit | AUC | Purpose |
|---|---|---|---|
| **Model B** (season GBM) | Runner-season | 0.662–0.700 | Headline predictor |
| Model A (per-attempt GBM) | Individual attempt | ~0.59 | Strict noise-floor test |
| GLM | Runner-season | — | Interpretable weight table |

---

## How to Run

```bash
python3 v6_explore.py        # full pipeline → data/, figures/, reports/
python3 write_glossary.py    # regenerate Variable Glossary → reports/
```

---

## Repository Structure

```
The-Naylor-Model/
├── v6_explore.py              ← full v6 pipeline
├── write_glossary.py          ← Variable Glossary generator
├── figures/                   ← all output PNGs
├── data/                      ← all output CSVs (leaderboards, SSSI, model results)
├── reports/                   ← PDFs (full report + variable glossary)
└── Previous Versions/
    ├── v3/                    ← naylor_model.py + v3 outputs
    ├── v4/                    ← v4_explore.py + v4 outputs
    └── v5/                    ← v5_explore.py + v5 outputs
```

---

## Data Sources

- Baseball Savant: sprint speed, running splits, catcher pop times, pitcher running-game leaderboard, base-stealing run value
- MLB Stats API: season SB/CS records (2015–2026)
- Statcast pitch-level feed: per-pitch runner context, battery matchups
