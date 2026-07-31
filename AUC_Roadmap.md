# AUC Roadmap — How to Push the Per-Attempt Model Higher

## TL;DR — the model is per-attempt, and that's what got us into range

The project's model works at the **attempt grain**: ~10,366 individual tracked steal attempts
(`model/metrics.py` (`run_perattempt`)), not 673 season averages. That choice is what reached the target —
CV AUC **0.753**, with no leakage. (An earlier season-aggregate predictor topped out near 0.62 and
has been removed; season data now only powers the descriptive SSSI / xSB / Blueprint outputs.)

| Model | Unit | Rows | AUC |
|---|---|---|---|
| season aggregate (removed) | runner-season | 673 | ~0.62 |
| **per-attempt XGBoost (the model)** | **individual attempt** | **~10,366** | **0.753** |

**What drove it:** the per-pitch **lead distances** (`lead_at_firstmove_ft`, `gain_to_release_ft`,
`lead_at_release_ft`) — how much ground the runner actually covered on *that* attempt. Exactly the
project's thesis: ground covered, not raw speed, decides the steal.

**Battery tendency — corrected.** An earlier version reported that catcher/pitcher encodings *hurt*
(0.723 < 0.739). That was an encoding bug, not a finding: the training rows were target-encoded from
those same rows, so each row's own outcome leaked into its feature, the model over-trusted it, and the
clean validation encoding then behaved differently. With the encoding nested properly inside the
training fold the sign flips:

| variant (nested out-of-fold) | AUC |
|---|---|
| leads + base + runner skill | 0.741 |
| **+ catcher tendency** | **0.752** |
| + pitcher tendency | 0.743 |
| + both | 0.753 |

The **catcher** carries the gain — 147 catchers seen a median of 46 times each, enough to estimate a
real trait (pop time, arm). The **pitcher** adds almost nothing: 1,025 pitchers at a median of 6
attempts is too sparse to estimate a hold tendency, and the runner's lead distances already absorb
most of the pitcher's effect — he only goes when he likes the look. The leak scaled as ~1/(n+20) per
row, which is why the sparse pitcher encoding did the damage (0.708 on its own).

**Leakage discipline (so 0.753 is honest):** no outcome-derived columns (`run_value` dropped); the
runner's own season success rate is excluded; catcher/pitcher encodings are nested out-of-fold on both
the training and validation side.

## Classic era (2015–2022) and the 2026 refresh

The pool is now the full **2015–2026** span, split into two separately-fit models because the 2023
rule package (two-disengagement limit, bigger bases, pitch clock) moved the baselines too far to
pool:

| league constant | classic 2015–2022 | modern 2023–2026 |
|---|---|---|
| SB success | 75.0% | 80.4% |
| ground vs speed | 25.1 − 0.48·spd | 36.8 − 0.89·spd |
| per-attempt AUROC | 0.745 (leads+runner) | 0.783 (full) |

Same architecture (`model/metrics.py classic` reuses the same `fit_league`/`score`), own constants.
Classic = v11 + v12 only; the v13 decision model is deliberately **2023–2026** (the attempt-vs-
no-attempt question is about the current rule environment). The modern 2026 season was refreshed
current (through 2026-07-17), which re-fit the league on fuller data and republished the leaderboard
and live site; the 2023–2025 metrics moved only slightly (mean |ΔSteal+| ≈ 0.04) since only the
league constants shifted.

## v13 — what was tried next, and what it bought

**Validation hardened (the number holds).** Reported under three regimes instead of one:

| split | AUROC | note |
|---|---|---|
| random (5-fold) | 0.783 | optimistic — a runner's other attempts sit in training |
| GroupKFold by runner | 0.783 | no runner on both sides; essentially unchanged |
| **forward (train past → test future)** | **0.770** | the honest number, and how the model would actually be used |

**Calibration.** In-distribution the model is well calibrated (max gap 0.045, mean −0.002). The
forward holdout drifts (max gap 0.091, mean −0.012) because *the league is drifting*: SB success
runs 0.830 → 0.810 → 0.808 → **0.769** across 2023-26. A model trained on the past predicts an
easier game than the one it is scored in. That is a base-rate shift, not over-confidence, so no
isotonic correction was applied — it would hide the cause. See `DF_v12_Calibration.csv`.

**Post-pitch count retired.** `balls`/`strikes` arrive after the pitch (`strikes==3` is a
strikeout), so they are not what a coach sees. Dropping them costs **0.0011** AUROC
(0.7840 → 0.7829) and is worth it.

### Dead ends — measured, so nobody repeats them

- **More season-aggregate features do nothing.** ~60 unused columns already sit in
  `DF_v7_SSSI.csv` (`avg_pop_faced`, `avg_pickoff_rate_faced`, `avg_pre_release_velocity`,
  accel/split-time curves, rule-era flags). Adding all 13 relevant ones: **0.7840 → 0.7840**.
  Subsets: 0.7836 / 0.7845 / 0.7846 — all within ±0.0006. The model is saturated on
  runner-season aggregates.
- **Pitcher-side running game is not reachable, and would not help anyway.** Savant's
  `basestealing-run-value` leaderboard is RUNNER-only (its `type` param selects the metric view,
  not the entity; `type=Pit` silently returns runners), and the per-entity service returns 0 rows
  for a pitcher id. `pitch-tempo` *does* expose pitcher hold (median seconds with runners on vs
  bases empty) but **ignores the year parameter** — identical data for 2023-26 — so it cannot be
  used season-correctly. Computing pitcher lead-allowed from our own attempts with leave-one-out
  (year-correct, free) gives **+0.0006**. The wrong-year tempo snapshot gives **−0.0008**. This is
  consistent with the earlier finding that pitcher target-encoding was worth +0.003 vs catcher's
  +0.011: pitchers are too sparse, and the runner's realised lead already absorbs the pitcher.

**Calculator coverage.** The 3-input web calculator was fit on only 6,712 of 10,366 attempts,
because it merged the 408-runner-season leaderboard. Sprint speed now comes from the full Savant
leaderboard and Burst is recomputed offline against the same league line (min 3 tracked attempts,
shrunk toward the league mean below ~10). Now **10,063 attempts, and AUC improved 0.730 → 0.737**.

## v13 — modelling the decision, not the outcome

Every earlier model conditions on an attempt having happened, which is why sprint speed alone
scores AUROC 0.53 on the success model. v13 builds the denominator that never existed: **517,068
pitches with a runner on 1st and 2nd empty**, from 8,148 regular-season games, of which **11,917
are attempts (2.30%)**. Base-state replay agreed with MLB's own end-of-PA state on **99.25%** of
615,967 plate appearances.

Labelling is league-wide off the play-by-play feed (`runners[].details.eventType`), so it covers
**917 runners and 11,917 attempts** versus the 499 runners / 9,412 tracked 2B attempts Savant
publishes. A free by-product: pitch ordering gives the **true pre-pitch count**, retiring the
post-pitch caveat that forced `SITUATION_FEATS` out of v12.

**The headline: runners self-select, hard.** Attempt rate by sprint-speed quintile —

| slowest | slow | mid | fast | fastest |
|---|---|---|---|---|
| 0.70% | 1.22% | 1.87% | 3.08% | **4.87%** |

A **7x** spread. Fast runners take the chances; slow runners only go when the situation is already
favourable. That is the selection effect, measured — and it is the reason speed looks worthless
*inside* the success model while dominating the decision to run at all.

**Model.** P(attempt) on situation + personnel. AUPRC leads here because attempts are 2.3% of
opportunities (the inverse of the success model's 81%), so the no-skill floor is 0.023:

| split | AUPRC (floor 0.023) | AUROC | Brier |
|---|---|---|---|
| random | 0.0765 | 0.7799 | 0.0219 |
| group (by runner) | 0.0686 | 0.7704 | 0.0220 |
| **forward** | **0.0652** | **0.7620** | **0.0225** |

About 3x the no-skill floor, and the three regimes now agree — which they did not at first.

**A leak caught in the build.** The runner's own attempt rate is the natural "scouting report"
prior, but a same-season leave-one-out version still reads the season being predicted: random
AUPRC 0.2534 against grouped 0.0606, a 4x gap. Rebuilt as an expanding **prior-seasons-only** rate
the gap closes and forward calibration improves from max gap 0.171 to **0.016**. Third time this
same self-contamination pattern has appeared in this project.

**Known limitation.** The lead distances that carry all of v11/v12's signal exist only on tracked
attempts — Savant never publishes a lead for a pitch where the runner stayed. So the decision model
structurally cannot see the runner's *signal*, only the situation and the personnel. Not a
modelling choice; a data limitation.

## Collinearity audit — two structural defects found and fixed

A critical review asked whether Burst and ground gained are redundant. They are not (they are the
same measurement at two baselines, VIF ~1.2 in the calculator), but the audit that answered the
question surfaced **two real defects elsewhere**, both invisible to accuracy:

**1. Three lead features carrying two pieces of information.** `lead_at_release =
lead_at_firstmove + gain_to_release` to R² = **0.999895** — the residual caps at 0.1 ft, which is
just the rounding. VIFs were **1,588 / 5,836 / 9,532**. XGBoost's predictions were unaffected
(dropping any one moved AUROC ±0.002), but feature importance was split arbitrarily across
perfectly dependent columns, so the importance chart could not be quoted. `lead_at_release_ft` is
now dropped.

**2. The dummy-variable trap.** `pc_fastball + pc_breaking + pc_offspeed` sum to 1 on **99.94%** of
rows (7 "unknown" codes), giving pc_fastball a VIF of **561**. Fastball is now the reference
category and is omitted, in v12 and v13.

**3. Redundant speed measures.** `jump_time` correlates −0.59 with sprint speed (and bolts +0.71),
pushing sprint_speed to VIF 16.8. Dropping it takes max VIF to **6.6** *and* nudges AUROC up
(0.7820 → 0.7829). It is still carried in the data and shown on player cards — only removed as a
model feature.

Net effect on the headline: **v12 AUROC 0.7829, unchanged.** Nothing was traded away; the model is
simply now interpretable. A standing **VIF guard** in `model_eras.py` asserts max VIF < 10 across
every feature set, so this class of defect fails the build from now on — it is what caught defects
2 and 3.

**Metric reporting fixed too.** AUPRC was being published without its no-skill floor (0.939 against
a floor of 0.811 is a modest lift, not an excellent one), and v13's F1@0.5 is exactly **0.000**
because at a 2.3% base rate nothing crosses 0.5. Both now print with floor and tuned threshold.

**Calculator respecified by measurement.** Every candidate was fit on the sample it could ship on:

| spec | n | AUROC |
|---|---|---|
| speed + burst + gain (old) | 7,404 | 0.7259 |
| speed + burst + firstmove + gain + pop | 7,264 | 0.7470 |
| **speed + firstmove + gain + pop** | **10,844** | **0.7559** |

Burst needs a qualified runner-season, so carrying it discarded ~3,600 attempts while adding only
+0.002; catcher pop time was worth ~8× more. Burst stays the season-level technique metric on the
leaderboard — it is simply not a per-pitch input. Logistic regression was kept over XGBoost because
it **beat** it on the same features (0.726 vs 0.722).

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
2. **Add them as per-attempt features** in `model/metrics.py` (`run_perattempt`, extend the `feats` list), re-run
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

> The per-attempt model (0.753) uses leads + base + runner skill + catcher tendency. To push further: fetch
> **`p_throws`** and **`pitch_name`** for each `play_id` in the leads cache (Savant per-pitch feed),
> add `is_lhp` and `pitch_class` as per-attempt features in `model/metrics.py` (`run_perattempt`), and re-run.
> Pitcher handedness at the attempt level is the single most likely lift (LHP see and hold the runner).
