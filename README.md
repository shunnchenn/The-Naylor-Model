# The Naylor Model

José Caballero led MLB in net stolen bases in 2025 running a quarter-second slower than Chandler Simpson. Shohei Ohtani was the second most productive base-stealer in 2024 despite being 0.14 seconds slower than Elly De La Cruz. Most strikingly, Josh Naylor stole 20 bases above average at 93.8% success while running slower than 97% of the league. Sprint speed is the most intuitive base-stealing metric — it is not the most essential one.

What separates these runners is technique — and technique is coachable precisely because it reflects what a player has learned to do with their body, not what their body is built to do. Sprint speed is structural. Primary lead distance, secondary lead timing, and first-step burst off the pitcher's first move are behavioral patterns that haven't permanently locked in, which means they can be shifted.

Sprint biomechanics research points to three specific targets every baseball player can develop regardless of raw speed: shorter ground contact time, more distance covered in the first five-foot window from the pitcher's first move, and earlier recognition of delivery cues. These aren't elite-only adaptations — they are timing and sequencing refinements accessible to any MLB-level runner. Naylor's edge isn't a physical gift; it's that he optimizes all three within a body most evaluators would write off.

That's why specificity with the biomechanics suite matters. Knowing a runner has a "slow jump" isn't actionable. Knowing exactly where in the ground contact phase they're losing time, and at which keyframe their secondary lead stalls, is. The more precisely the metric targets the problem, the more directly the coaching intervention follows.

---

## What This Model Does

The Naylor Model is a pitch-level stolen base intelligence system built to identify and quantify what makes runners with below-average sprint speed effective at stealing bases. It uses real Statcast data — sprint speed, running splits at 5-ft increments (0–90 ft), catcher pop times, pitcher running-game suppression, and real season SB/CS records — to separate what is structurally fixed from what is technique-driven and trainable.

The core signal is the **SB Residual**: a runner's actual success rate minus the rate their sprint speed alone would predict. Positive means they outperform their speed peers. Naylor's residual is large and positive. Trea Turner's is smaller than you'd expect — his success rate is exactly in line with what a 30 ft/s runner should do.

## Key Metrics

| Metric | What it captures |
|---|---|
| `sprint_speed` | Top running speed (ft/s) — structural baseline |
| `speed_capped` | Sprint speed capped at 28 ft/s — marginal benefit vanishes above this threshold |
| `jump_time` | Time to cover the first 30 ft — first-step burst independent of top speed |
| `accel_gap` | Percentile rank of jump time minus percentile rank of sprint speed — positive = faster off the line than top speed implies (the Naylor archetype) |
| `sb_residual` | Real shrunk SB% minus speed-expected SB% — ground-truth speed-adjusted steal skill |
| `lead_gain` | Distance gained in secondary lead — a coachable behavioral pattern |
| `avg_pop_faced` | Catcher pop time in matchups this runner faced — battery context |
| `avg_pickoff_rate_faced` | Pitcher hold frequency — suppression context |

## The SSSI — Slow-Steal Skill Index

A weighted composite of eight z-scored features designed to surface the Naylor/Soto archetype: elite-performing slow runners. Weights were optimised on 80% of runners with Naylor and Soto held out entirely — their ranking in the final index is a genuine out-of-sample result.

**2025 SSSI Top 5:** Naylor #1 · Naylor 2026 #2 · Freeman 2024 #3 · Soto 2025 #5

## Models

Three models serve different purposes:

- **Model A (per-attempt GBM):** One row per steal attempt with strict group cross-validation by runner ID. AUC ~0.59 — at the noise ceiling for individual attempt prediction.
- **Model B (season-level GBM):** One row per runner-season. AUC 0.662–0.700. The headline predictor.
- **GLM:** Logistic regression with one coefficient per feature. Not the best predictor — the most interpretable. Produces the "SB% Boost per Tier" and "Odds Multiplier" columns in the output tables.

## How to Run

```bash
python3 v6_explore.py
```

Outputs all CSVs, figures, and the full PDF report into the project directory.

## Repository Structure

| File | Contents |
|---|---|
| `v6_explore.py` | Full v6 pipeline |
| `write_glossary.py` | Generates the Variable Glossary PDF |
| `Variable_Glossary.pdf` | Plain-English reference for every metric — Statcast-style with tier charts, real examples, and a full model discussion |
| `Naylor_Model_v6_Report.pdf` | Complete analysis report |
| `Previous Versions/` | Archived v3, v4, v5 pipelines and outputs |
| `.cache/` | Pitch-level Statcast pickle cache (local only — not tracked) |

## Data Sources

- Baseball Savant: sprint speed, running splits, catcher pop times, pitcher running-game leaderboard, base-stealing run value
- MLB Stats API: season SB/CS records (2015–2026)
- Statcast pitch-level feed: per-pitch runner context, battery matchups
