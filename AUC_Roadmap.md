# AUC Roadmap — How to Push the Steal-Success Model Higher

## TL;DR — we hit the target by changing the *unit of analysis*

The season model was stuck near 0.62 because it had only 673 rows and a noisy season-average target.
Switching to a **per-attempt model** (`scripts/model_perattempt.py`, ~10,400 individual tracked
attempts) jumped CV AUC to **0.739** — into the 0.7–0.85 goal range — with no leakage.

| Model | Unit | Rows | AUC |
|---|---|---|---|
| v7 GBM (old Model B) | runner-season | 673 | 0.589 |
| v8 tuned XGBoost (Model B) | runner-season | 673 | 0.624 |
| **v9 per-attempt XGBoost (Model A)** | **individual attempt** | **~10,400** | **0.739** |

**What drove the jump:** the per-pitch **lead distances** (`lead_at_firstmove_ft`,
`gain_to_release_ft`, `lead_at_release_ft`) — how much ground the runner actually covered on *that*
attempt. That is exactly the project's thesis: ground covered, not raw speed, decides the steal.
Adding out-of-fold catcher/pitcher tendency encodings *did not help* (0.723 < 0.739) — the leads
already carry the signal.

**Leakage discipline (so 0.739 is honest):** no outcome-derived columns (`run_value` dropped); the
runner's own season success rate is excluded; catcher/pitcher encodings are computed out-of-fold.

## Would deep learning help? No.

At ~10k rows and ~10 tabular features, **gradient boosting is the right tool**. Neural nets need far
more data and overfit on tables this size — on tabular benchmarks XGBoost/CatBoost beat deep nets
until you have hundreds of thousands of rows. Deep learning would only make sense if we fed it *raw
pose/tracking sequences* (the CV pilot's territory), not these aggregates.

## The honest ceiling

Public Statcast data realistically tops out around **0.74–0.78** for per-attempt success. Stealing has
an irreducible coin-flip component (exact release, throw accuracy, tag, replay). 0.85 would likely
require proprietary data (catcher exchange video, pitcher tells) or label leakage — not worth chasing.
The remaining honest gains come from **matchup context**, below.

---

## The biggest gap: matchup context

Every current feature describes the **runner** (speed, jump, leads) or a season-averaged opponent
blur (`avg_pop_faced`, `avg_pickoff_rate_faced`). The model has almost no information about *who the
runner went against and in what situation*. That is the largest untapped lever — and it is matchup
data, which is why it is prioritized here.

### Tier 1 — needs a targeted Statcast re-pull (highest expected lift)

These per-pitch fields exist on Baseball Savant for every tracked attempt; the pipeline just hasn't
aggregated them to the runner-season level yet.

| Variable | Why it should move AUC | Feature to build (runner-season) |
|---|---|---|
| **Pitcher handedness `p_throws`** ⭐ | LHP see the runner and hold far better; this is the single most predictive matchup fact in steal analysis and is currently **entirely absent**. | `lhp_share` = share of attempts vs LHP; and split success by hand |
| **Pitch type / `pitch_name`** | Breaking balls & offspeed are slower to the plate → easier to steal; fastballs harder. Runners/coaches pick pitches. | `breaking_offspeed_share` at the attempt; success by pitch class |
| **`release_extension` / `release_speed`** | Extension shortens *effective* time-to-plate — a clean public proxy for the CV delivery-time metric the pilot chased, with full coverage. | `avg_release_extension_faced`, `avg_release_speed_faced` |
| **Catcher identity (not just pop time)** | Game-calling, transfer under pressure, and CS-above-expected vary well beyond a season-mean pop time. | catcher fixed-effect or `cs_above_exp_faced` |

**Why Tier 1 first:** `p_throws` alone is the classic missing variable in steal models and is cheap
to add once the attempt-level pull includes it. Expect the largest marginal AUC here.

### Tier 2 — already in the cached pitch-level data (cheap, no network)

The cached `DF_Pitch` table already carries `balls, strikes, outs, inning, on_1b, …`. These can be
aggregated immediately:

| Variable | Why | Feature |
|---|---|---|
| **Full count state** | The current `two_strike_share` is a weak single slice (GLM boost only +4 pp). Runners go on *favorable* counts; the full ball–strike profile carries more. | shares for hitter's counts (e.g. 1-0, 2-1, 3-1) and 2-strike |
| **Base/out state** | 1B-only vs 1B+2B changes the play (double steals, holding the bag); outs change aggressiveness. | `runner_on_2b_share`, `outs_when_running` |
| **Inning** | Late-inning leverage shifts both the decision to run and the defense's attention. | `late_inning_share` |

---

## How to execute

1. **Aggregate Tier-2 now** from `DF_Pitch` (no network) → add features in `scripts/v7_explore.py`,
   re-run `scripts/benchmark_models.py` to measure each block's *marginal* AUC before committing.
2. **Re-pull Tier-1** matchup fields (`p_throws`, `pitch_name`, `release_extension`) for the tracked
   attempts via the Savant per-pitch feed (`Computer Vision/code/fetch_*`), aggregate to runner-season
   shares, and re-run the benchmark + `scripts/tune_xgboost.py`.
3. **Re-tune on the enriched feature set** — the current best params were found on 18 features; a
   wider matrix deserves a fresh Optuna search.

### Guardrails
- **One row per runner-season.** The v4–v6 AUCs (0.66–0.70) were inflated by duplicate split rows
  leaking across CV folds. Keep the de-leaking; never let split-duplicates cross folds.
- **Measure marginal lift per block**, not just the final number — so we know *which* matchup data
  earned its place.
- **Frame gains honestly.** A move from 0.62 → 0.66 is meaningful for ranking skill; it is not a
  promise about any single attempt.

## The concrete next experiment

> The per-attempt model (0.739) already uses leads + base + runner skill. To push toward ~0.76:
> fetch **`p_throws`** and **`pitch_name`** for each `play_id` in the leads cache (Savant per-pitch
> feed — keyed by the `play_id` already stored), add `is_lhp` and `pitch_class` as per-attempt
> features in `scripts/model_perattempt.py`, and re-run. Pitcher handedness at the attempt level is
> the single most likely lift (LHP see and hold the runner). Then re-tune with Optuna on the enriched
> per-attempt matrix.
