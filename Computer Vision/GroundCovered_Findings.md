# Ground Covered (Secondary Distance) — League-Wide Leaderboard

## The question
Do slow runners like **Josh Naylor** and **Juan Soto** steal bases on the *ground they
cover between the pitcher's first move and pitch release* rather than on raw speed — and
can we measure that "ground covered" without computer vision?

## The metric (native Statcast — no CV)
```
gain_to_release_ft = r_secondary_lead − r_primary_lead
                   = (lead at pitch release) − (lead at the pitcher's first move)
```
Feet of lead gained from the pitcher's first move to release, pulled per attempt straight
from Baseball Savant's basestealing-running-game service (`cv_pilot/fetch_leads.py`).
**No clips, no pose detector.**

> We earlier built a YOLOv8-pose pipeline to measure pitcher *delivery time*. On 128
> attempts it added **no** marginal predictive lift over the Statcast lead distances —
> leave-one-out AUC was **0.838** with the lead features alone vs **0.822** after adding
> CV delivery time (`cv_pilot/Naylor_model/attempt_auc.csv`). The "ground covered" signal
> was already in Statcast. So this leaderboard is pure Statcast.

## Data
- **2023–2025** (the Statcast lead-tracking era), all qualified runners (season SB+CS ≥ 5),
  built from `cv_pilot/discover_runners.py` (year-correct sprint speed + SB/CS).
- **754** runner-seasons with tracked attempts; **357 volume-qualified** (≥ 10 tracked
  attempts) are ranked. Thin samples (3–9 attempts) stay in the file but are left unranked —
  a 4-for-4 fluke with two big leads should not headline a leaderboard.

## Finding 1 — Ground covered is *not* a speed skill (it's mildly anti-correlated)
Regressing mean ground covered on sprint speed (+ season): slope **−0.68 ft per ft/s**,
correlation **−0.38** (−0.37 across all 754 seasons). **Slower runners cover *more* ground**,
not less. Speed does not buy secondary distance.

Why: selection. A fast runner can steal on a modest lead; a slow runner only survives as a
basestealer if he manufactures a big secondary lead with an early jump and good timing. The
slow basestealers who remain are precisely the elite-jump artists.

## Finding 2 — Ground covered predicts steal success
Runner-season level (volume-qualified): corr(mean ground covered, SB%) = **+0.21**;
corr(speed-adjusted residual, SB%) = **+0.26**. Per attempt (earlier work), ground covered
alone earns LOO-AUC **0.786**. More secondary distance ⇒ higher success.

## Finding 3 — Naylor is the archetype; Soto corroborates
**Josh Naylor 2025** — sprint **24.4 ft/s = 1.9th percentile**, the slowest volume-qualified
basestealer in the league — yet:
- Mean ground covered **16.74 ft = #3 of 357 league-wide** (behind only Agustín Ramírez and
  Ramón Laureano, both faster).
- **#1 among all slow (≤ 15th-pctile) basestealers** — ahead of Soto (14.18), Freeman
  (13.40), Machado (12.88).
- Speed-adjusted residual **+2.64 ft = #20 of 357 = 94.7th percentile**.
- Result: **22 / 23 = 95.7 % SB**.

**Juan Soto** — 2023: **16.12 ft** (#5 raw, #12 residual = 96.9th pctile); 2025: **14.18 ft**
on a **30-for-30** season (#23 raw). Soto's 2025 residual is more modest (#85) because at
25.8 ft/s the model already expects more of him — yet he still stole every base.

## Honest caveats
- The residual top-20 is **not** only slow players — fast elite basestealers (Trea Turner,
  Anthony Volpe, Sal Frelick) also cover ground above expectation. Secondary distance is a
  real skill some fast players share; it is simply the *only* path for the slow archetype.
  The honest claim is therefore **"among slow runners Naylor/Soto are the best, and they
  rank top-tier league-wide despite bottom-percentile speed,"** not "only slow players cover
  ground." On the top-25 chart Naylor (24.4 ft/s) is by far the slowest runner present.
- 2023 (bigger bases / pickoff limits) inflates leads; handled with a season term in the
  residual model.
- Small-sample seasons are excluded from ranking (kept in the CSV as `volume_qualified=False`).

## Outputs
- `data/DF_GroundCovered_Leaderboard.csv` — 754 runner-seasons; volume-qualified rows ranked
  by speed-adjusted residual; raw + residual ranks, leads, SB%, sprint speed.
- `figures/Fig_GroundCovered_Scatter.png` — ground covered vs sprint speed, OLS fit,
  Naylor/Soto starred at the low-speed / high-ground corner.
- `figures/Fig_GroundCovered_TopN.png` — top-25 runner-seasons by speed-adjusted residual.
- `cv_pilot/ground_covered_leaderboard.py` — the (CV-free) driver; reruns are free via
  `cv_pilot/discovery/leads_cache/`.

## Bottom line
The thesis holds, and it is pure Statcast: secondary distance covered first move → release
is a near-speed-independent skill that predicts steal success, and Josh Naylor — the slowest
qualified basestealer in baseball — is its poster child, covering the 3rd-most ground in MLB.
No computer vision required.
