# AUC Roadmap — How to Push the Per-Attempt Model Higher

## TL;DR — the model is per-attempt, and that's what got us into range

The project's model works at the **attempt grain**: ~10,366 individual tracked steal attempts
(`Scripts/model_perattempt.py`), not 673 season averages. That choice is what reached the target —
CV AUC **0.739**, with no leakage. (An earlier season-aggregate predictor topped out near 0.62 and
has been removed; season data now only powers the descriptive SSSI / xSB / Blueprint outputs.)

| Model | Unit | Rows | AUC |
|---|---|---|---|
| season aggregate (removed) | runner-season | 673 | ~0.62 |
| **per-attempt XGBoost (the model)** | **individual attempt** | **~10,366** | **0.739** |

**What drove it:** the per-pitch **lead distances** (`lead_at_firstmove_ft`, `gain_to_release_ft`,
`lead_at_release_ft`) — how much ground the runner actually covered on *that* attempt. Exactly the
project's thesis: ground covered, not raw speed, decides the steal. Adding out-of-fold
catcher/pitcher tendency encodings *did not help* (0.723 < 0.739) — the leads already carry the signal.

**Leakage discipline (so 0.739 is honest):** no outcome-derived columns (`run_value` dropped); the
runner's own season success rate is excluded; catcher/pitcher encodings are computed out-of-fold.

## Would deep learning help? No.

At ~10k rows and ~10 tabular features, **gradient boosting is the right tool**. Neural nets need far
more data and overfit on tables this size — on tabular benchmarks XGBoost/CatBoost beat deep nets
until you have hundreds of thousands of rows. Deep learning would only make sense if we fed it *raw
pose/tracking sequences* (the CV pilot's territory), not these features.

## The honest ceiling

Public Statcast data realistically tops out around **0.74–0.78** for per-attempt success. Stealing has
an irreducible coin-flip component (exact release, throw accuracy, tag, replay). 0.85 would likely
require proprietary data (catcher exchange video, pitcher tells) or label leakage — not worth chasing.
The remaining honest gains come from **per-attempt matchup context**, below.

---

## The biggest gap: per-attempt matchup context

Every current feature describes the **runner** (speed, jump, the leads on that attempt). The model has
almost no information about *who the runner went against and in what situation on that pitch*. That is
the largest untapped lever — and the leads cache already stores the `play_id` for every attempt, so
these fields can be joined per attempt without re-deriving anything.

### Tier 1 — needs a targeted Statcast re-pull (highest expected lift)

These per-pitch fields exist on Baseball Savant for every tracked attempt; join them to the leads
cache on `play_id` and add them as **per-attempt** features.

| Variable | Why it should move AUC | Per-attempt feature |
|---|---|---|
| **Pitcher handedness `p_throws`** ⭐ | LHP see the runner and hold far better; the single most predictive matchup fact in steal analysis, currently **entirely absent**. | `is_lhp` on the attempt |
| **Pitch type / `pitch_name`** | Breaking balls & offspeed are slower to the plate → easier to steal; fastballs harder. | `pitch_class` (fastball / breaking / offspeed) on the attempt |
| **`release_extension` / `release_speed`** | Extension shortens *effective* time-to-plate — a public proxy for the CV delivery-time metric, with full coverage. | `release_extension`, `release_speed` on the attempt |
| **Catcher identity (not just pop time)** | Game-calling and transfer under pressure vary well beyond a season-mean pop time. | out-of-fold catcher CS-above-expected encoding |

**Why Tier 1 first:** `p_throws` at the attempt level is the classic missing variable in steal models.
Expect the largest marginal AUC here.

### Tier 2 — cheap per-attempt count / state context

Pulled with the same Savant per-pitch feed (or the cached pitch table), keyed by `play_id`:

| Variable | Why | Per-attempt feature |
|---|---|---|
| **Count state** | Runners go on *favorable* counts; the ball–strike state carries real signal. | `balls`, `strikes` on the attempt |
| **Base/out state** | 1B-only vs 1B+2B changes the play; outs change aggressiveness. | `runner_on_2b`, `outs` on the attempt |
| **Inning** | Late-inning leverage shifts both the decision to run and the defense's attention. | `late_inning` flag |

---

## How to execute

1. **Join `play_id` → matchup fields** from the Savant per-pitch feed (the `play_id` is already stored
   in every leads-cache row), starting with Tier 1.
2. **Add them as per-attempt features** in `Scripts/model_perattempt.py` (extend `base_num`), re-run
   the 5-fold CV, and read the marginal AUC of each block before keeping it.
3. **Tune once the feature set is wider.** Model A is currently an untuned default spec; after adding
   matchup features, an Optuna search (with nested CV to keep the estimate honest) is the right step.

### Guardrails
- **Measure marginal lift per block**, not just the final number — so we know *which* matchup data
  earned its place.
- **Keep the leakage discipline.** No outcome-derived fields; encode catcher/pitcher identity
  out-of-fold; never let a runner's own outcome inform his features.
- **Frame gains honestly.** ~0.74 is near the public-data ceiling; new features should be judged on
  honest marginal AUC, not a promise about any single attempt.

## The concrete next experiment

> The per-attempt model (0.739) uses leads + base + runner skill. To push toward ~0.76: fetch
> **`p_throws`** and **`pitch_name`** for each `play_id` in the leads cache (Savant per-pitch feed),
> add `is_lhp` and `pitch_class` as per-attempt features in `Scripts/model_perattempt.py`, and re-run.
> Pitcher handedness at the attempt level is the single most likely lift (LHP see and hold the runner).
