# The Naylor Model

Pitch-level stolen-base intelligence focused on the slow-but-effective stealer archetype. The goal is to identify and quantify what makes runners with below-average sprint speed (Josh Naylor, Juan Soto, others) effective at stealing bases.

## Data

- **Real**: MLB Statcast sprint speed, running splits (0–90 ft at 5 ft increments), and real season SB / CS from the MLB Stats API (2023–2026).
- **Simulated pitch-level features** (lead distances, jump time, pitcher TTP, catcher pop time) calibrated to Baseball Savant ranges. Naylor (`647304`) and Soto (`665742`) have their lead-tendency parameter anchored to documented elite values rather than random draws.

## Key concepts

| Feature | What it captures |
|---|---|
| `speed_capped` | `min(sprint_speed, 28)` — past 28 ft/s the marginal effect on SB success is empirically ≈ 0 |
| `accel_0_30` | Time to cover the first 30 ft from a stop. Correlates with sprint speed only at r ≈ −0.76, so it adds independent signal. |
| `accel_gap` | `pct(accel_0_30) − pct(sprint_speed)`. Positive means the runner is faster off the line than their top speed suggests — the Naylor archetype. |
| `sb_residual` | Real shrunk SB% minus the speed-expected SB% (poly fit). This is the ground-truth, speed-adjusted demonstrated steal skill. |

## SSSI v3 — Slow-Steal Skill Index

Two flavours:

- **Fixed weights** (plan default):  
  `0.35 · z(sb_residual) + 0.25 · z(accel_gap) + 0.15 · z(lead_gain) + 0.10 · z(jump) + 0.10 · z(lead_off) − 0.05 · z(speed_capped)`

- **Optimised weights**: grid-searched to maximise the mean z-score of Naylor + Soto on the SSSI distribution. Reported alongside the fixed version with a sensitivity table.

## Models

- Mixed-effects logistic regressions for **steal attempt** and **steal success** with random intercepts for runner / pitcher / catcher (implemented as L2-penalised high-cardinality dummies).
- Seven candidate feature combinations compared via 5-fold stratified CV; gradient boosting comparator on the full feature set.
- Hyperparameter `C` tuned via `GridSearchCV` on the full model.

## How to run

```bash
python3 naylor_model.py
```

Outputs everything (CSVs, figures, multi-page PDF report) into the project directory.

## Files

| File | What it contains |
|---|---|
| `naylor_model.py` | The full v3 pipeline |
| `Naylor_Model_Report.pdf` | Multi-page report with all findings |
| `DF_Pitch_Level.csv` | One row per pitch with a runner on first (stored via Git LFS — large) |
| `DF_Runner_Stats.csv` | Runner-season aggregates |
| `DF_Skill_Index.csv` | SSSI v1 / capped / composite / v3 fixed / v3 optimised |
| `DF_Real_SB.csv` | Real SB, CS, SB%, expected SB%, residual |
| `DF_Acceleration.csv` | Running-splits-derived acceleration metrics |
| `DF_Model_Comparison.csv` | AUC / log-loss / Brier / ECE / Lift@20 across model variants |
| `DF_Speed_Cap_Analysis.csv` | Hinge regression confirming the 28 ft/s cap |
| `DF_Speed_Expectation.csv` | Polynomial-fit expected SB% vs speed |
| `DF_SSSI_Sensitivity.csv` | Rank stability under ±25% weight perturbations |
| `DF_Naylor_Profile.csv` | Naylor's headline numbers |
| `DF_Naylor_Contributions.csv` | Naylor log-odds breakdown by feature |
