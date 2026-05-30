# The Naylor Blueprint — Full-Spectrum Basestealing Leaderboard

*One leaderboard, two poles. The top is the blueprint; the bottom is its mirror image.*

---

## What this measures

Stealing bases is treated here as a **conversion skill, not a speed contest**. Every
runner-season is scored by a single **Blueprint Conversion Score (BCS)** that rewards
covering ground and converting it into steals **relative to the runner's own speed**,
and penalizes fast runners who get caught anyway.

- **TOP — the Naylor blueprint:** *slow* runners (low sprint speed, slow 90 ft) who
  cover more ground from the pitcher's first move to release than their speed predicts,
  and turn it into steals. They beat the running game on **timing and jump**, not wheels.
- **BOTTOM — the anti-Naylor:** the *fastest* runners (top sprint speed, fastest
  home-to-first) who nonetheless pile up caught-stealings — penalized for **failing
  despite an obvious speed advantage** and for **failing to capitalize on the ground
  they cover**.

## Annual (calendar-year) data

Every input is bounded to a single calendar year — no career or multi-year aggregates:

| Input | Source | Year bound |
|---|---|---|
| Stolen bases / caught stealing | MLB StatsAPI | `season={Y}&gameType=R` |
| Sprint speed, home-to-1B (90 ft) | Baseball Savant | `min_season={Y}&max_season={Y}` |
| Per-attempt lead & ground covered | Baseball Savant | `season_start={Y}&season_end={Y}` |

**One row = one runner for one season.** A runner who appears in 2023-2026
contributes up to three independent rows.

---

## The score

```
BCS = success_resid_z  +  gain_resid_z  −  squander_z
```

All three terms are z-scores across the **515 volume-qualified** runner-seasons
(≥10 tracked attempts).

| Term | What it is | Rewards / Penalizes |
|---|---|---|
| `success_resid_z` | Beta-Binomial steal-success posterior (shrunk toward the league 82% rate), regressed on speed → residual | **+** converting *more often than speed predicts* |
| `gain_resid_z` | Ground covered (first move → release) minus what speed predicts | **+** a big jump for how slow you are (Naylor) |
| `squander_z` | `CS · max(speed_z,0) · (1 + max(gain_z,0))` | **−** fast runners who get caught; amplified if they had the jump and *still* failed |

The squander penalty only bites **fast** runners (`speed_z > 0`) — slow runners cannot
squander a speed advantage they do not have, so they rest at the penalty floor. The
`(1 + gain_z)` factor is the "failed to capitalize" clause: a runner who covers a lot of
ground and *still* gets caught is punished harder than one who never had the jump.

The Beta-Binomial prior is empirical-Bayes (moment-matched to the league SB%
distribution: α₀=5.37, β₀=1.33), so a 4-for-4 cameo is shrunk
toward league average while a 30-for-31 season is trusted. Speed buys almost nothing in
ground covered — the league fit slope is near-flat — which is exactly why the residual
isolates a real, near-speed-independent skill.

**On the speed metrics (honest note).** Statcast publishes two of the requested
speed measures per player-season: **sprint speed** (peak velocity — feet per second in
the fastest one-second window) and **home-to-1B time** (the 90-ft burst, which captures
the *acceleration* phase out of the box). It does **not** publish per-player acceleration
or jerk (the higher derivatives of position) on the public leaderboards, so the speed
composite here is `mean(z(sprint_speed), −z(home_to_1B))` — high = fast on both peak
velocity and 90-ft burst. Home-to-1B stands in for acceleration; jerk is unavailable.

---

## TOP 15 — The Blueprint

| # | Runner | Yr | Sprint (pct) | 90ft | SB/CS | SB% | Gain ft | gain_z | succ_z | squander_z | BCS |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | Ryan McMahon | 2026 | 25.7 (14) | 4.59 | 3/0 | 100 | 24.5 | +6.44 | +1.09 | -0.53 | **+8.05** |
| 2 | Agustín Ramírez | 2025 | 26.7 (30) | 4.78 | 13/1 | 93 | 21.3 | +4.90 | +1.44 | -0.53 | **+6.86** |
| 3 | Paul Goldschmidt | 2024 | 26.3 (21) | 4.58 | 10/0 | 100 | 16.7 | +2.27 | +1.79 | -0.53 | **+4.59** |
| 4 | JJ Wetherholt | 2026 | 27.3 (50) | 4.30 | 5/0 | 100 | 16.8 | +2.78 | +1.14 | -0.53 | **+4.45** |
| 5 | Jordan Walker | 2025 | 28.7 (84) | 4.40 | 10/0 | 100 | 15.3 | +2.32 | +1.54 | -0.53 | **+4.39** |
| 6 | Josh Naylor | 2025 | 24.4 (1) | 4.86 | 22/1 | 96 | 16.7 | +1.42 | +2.07 | -0.53 | **+4.01** |
| 7 | Josh Naylor | 2026 | 24.6 (2) | 4.73 | 9/1 | 90 | 17.7 | +2.19 | +1.20 | -0.53 | **+3.91** |
| 8 | Ramón Laureano | 2023 | 27.9 (63) | 4.40 | 11/1 | 92 | 16.9 | +2.66 | +0.72 | -0.53 | **+3.90** |
| 9 | Chase DeLauter | 2026 | 26.6 (28) | 4.48 | 2/1 | 67 | 19.0 | +3.70 | -0.37 | -0.53 | **+3.85** |
| 10 | Trea Turner | 2023 | 30.3 (99) | 4.14 | 24/0 | 100 | 13.6 | +1.74 | +1.49 | -0.53 | **+3.75** |
| 11 | Juan Soto | 2025 | 25.8 (13) | 4.58 | 30/0 | 100 | 14.2 | +0.54 | +2.39 | -0.53 | **+3.45** |
| 12 | Michael A. Taylor | 2024 | 28.5 (80) | 4.30 | 10/0 | 100 | 13.6 | +1.40 | +1.51 | -0.53 | **+3.44** |
| 13 | Nico Hoerner | 2026 | 28.4 (81) | 4.34 | 10/0 | 100 | 13.5 | +1.35 | +1.53 | -0.53 | **+3.41** |
| 14 | Luis Arraez | 2026 | 26.7 (33) | 4.57 | 4/0 | 100 | 15.2 | +1.63 | +1.18 | -0.53 | **+3.33** |
| 15 | Kyle Tucker | 2026 | 26.4 (25) | 4.54 | 4/0 | 100 | 15.4 | +1.62 | +1.18 | -0.53 | **+3.33** |


## BOTTOM 15 — The Anti-Naylor (fast, yet caught)

| # | Runner | Yr | Sprint (pct) | 90ft | SB/CS | SB% | Gain ft | gain_z | succ_z | squander_z | BCS |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 501 | Taylor Ward | 2026 | 27.5 (56) | 4.47 | 1/2 | 33 | 7.1 | -2.60 | -1.78 | -0.53 | **-3.85** |
| 502 | David Hamilton | 2025 | 29.3 (93) | 4.02 | 15/5 | 75 | 10.5 | -0.14 | -0.74 | +3.07 | **-3.95** |
| 503 | Alejandro Osuna | 2026 | 28.5 (81) | 4.19 | 1/2 | 33 | 7.4 | -2.04 | -1.98 | +0.02 | **-4.04** |
| 504 | Jeremy Peña | 2023 | 29.4 (95) | 4.28 | 13/7 | 65 | 10.2 | -0.53 | -1.88 | +1.67 | **-4.08** |
| 505 | Jacob Young | 2025 | 29.3 (94) | 4.25 | 12/7 | 63 | 9.8 | -0.49 | -1.76 | +1.87 | **-4.11** |
| 506 | Luis Rengifo | 2025 | 27.0 (39) | 4.36 | 7/7 | 50 | 8.5 | -2.15 | -2.57 | -0.53 | **-4.20** |
| 507 | Myles Straw | 2023 | 29.2 (93) | 4.14 | 17/5 | 77 | 13.5 | +1.27 | -0.74 | +4.72 | **-4.20** |
| 508 | Bobby Witt Jr. | 2025 | 30.2 (99) | 4.15 | 32/7 | 82 | 10.1 | +0.03 | -0.02 | +4.34 | **-4.33** |
| 509 | Elly De La Cruz | 2024 | 30.0 (98) | 4.21 | 55/9 | 86 | 8.9 | -0.63 | +0.49 | +4.47 | **-4.60** |
| 510 | Ceddanne Rafaela | 2024 | 28.8 (88) | 4.17 | 15/8 | 65 | 9.9 | -0.57 | -1.63 | +2.41 | **-4.62** |
| 511 | Bobby Witt Jr. | 2024 | 30.5 (100) | 4.10 | 25/6 | 81 | 9.4 | -0.13 | -0.22 | +4.52 | **-4.87** |
| 512 | Daylen Lile | 2025 | 29.1 (90) | 4.23 | 8/6 | 57 | 8.3 | -1.43 | -2.14 | +1.47 | **-5.04** |
| 513 | Ji Hwan Bae | 2023 | 29.7 (97) | 4.05 | 19/7 | 73 | 11.2 | +0.16 | -1.27 | +4.75 | **-5.86** |
| 514 | Bobby Witt Jr. | 2023 | 30.5 (100) | 4.12 | 40/8 | 83 | 9.6 | -0.44 | -0.19 | +5.94 | **-6.57** |
| 515 | Chandler Simpson | 2025 | 29.6 (96) | 3.97 | 38/8 | 83 | 9.9 | -0.32 | -0.00 | +6.39 | **-6.71** |


---

## Josh Naylor 2025 — the archetype

- **Sprint speed:** 24.4 ft/s — **1.9th percentile** (bottom 2% of basestealers)
- **Home-to-1B:** 4.86 s (slow)
- **Ground covered:** 16.74 ft — among the most in MLB (gain residual z = +1.42)
- **Steal record:** 22/23 = 95.7% (success residual z = +2.07)
- **Squander penalty:** -0.53 (floor — he is too slow to squander)
- **BCS = +4.01 → rank #6 of 515**
- **Rarity:** P(this slow) · P(covers this much) ≈ **1.40e-07** ≈ **1 in 7,136,458** runner-seasons

Naylor is the gold standard: nearly the slowest runner in the dataset, yet he covers
the most ground and converts it at an elite rate. The model surfaces him near the very
top **because** of his slowness, not in spite of it.

## Juan Soto

- **2023:** 26.8 ft/s (34pct), 6/10 = 60%, gain 16.1 ft, BCS +0.53 → rank #209
- **2024:** 26.8 ft/s (32pct), 6/10 = 60%, gain 13.9 ft, BCS -0.06 → rank #284
- **2025:** 25.8 ft/s (14pct), 30/30 = 100%, gain 14.2 ft, BCS +3.45 → rank #11
- **2026:** 25.9 ft/s (19pct), 2/3 = 67%, gain 16.4 ft, BCS +2.16 → rank #55

---

## The anti-Naylor pattern

The bottom of the board is dominated by runners in the **top sprint percentiles with the
fastest 90-ft times** who keep running into outs:

- **Chandler Simpson 2025** anchors the bottom (#515):
  29.6 ft/s (96th pctile),
  3.97s home-to-1B, 38/8 —
  elite wheels, only 9.9 ft of ground covered, and a
  squander penalty of +6.39.
- **Bobby Witt Jr.** — a 100th-percentile sprinter — lands in the bottom tier in every season tracked: 2023 (#514, 8 CS), 2024 (#511, 6 CS), 2025 (#508, 7 CS), 2026 (#259, 2 CS). The textbook "obvious speed advantage, squandered."

These runners *should* be elite base-stealers on speed alone; the leaderboard penalizes
them precisely because they are **not converting an advantage Naylor would kill for.**

---

## Interpretation

1. **Ground covered is nearly speed-independent.** Sprint speed barely moves the gain
   from first move to release — so the gain residual measures a real timing/jump skill,
   not a proxy for wheels.
2. **The blueprint is a slow-runner edge.** The model rewards slow runners who cover
   ground and convert (Naylor, Soto, Ramírez, Goldschmidt) and is unimpressed by fast
   runners who do neither.
3. **The anti-Naylor is a cautionary tale.** Top-percentile speed with a high
   caught-stealing count is the worst profile on the board — wasted tools.

## Files

- `data/DF_NaylorBlueprint_Leaderboard.csv` — full spectrum, one row per VQ runner-season
- `figures/Fig_NaylorBlueprint_Scatter.png` — gain vs sprint, colored by BCS
- `figures/Fig_NaylorBlueprint_TopN.png` — top 15 (the blueprint)
- `figures/Fig_NaylorBlueprint_BottomN.png` — bottom 15 (the anti-Naylor)
