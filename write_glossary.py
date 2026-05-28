#!/usr/bin/env python3
"""
Variable Glossary  —  The Naylor Model
======================================
Produces Variable_Glossary.pdf: a standalone reference document explaining
every variable used in the v3/v4 model.  Goal:  a person who has never seen
this project should be able to read the glossary and understand exactly
what each column means, how it is computed, what units it is in, and what
sign of effect is expected on stolen-base success.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

OUTPUT_DIR = Path("/Users/shunchen/Desktop/The-Naylor-Model")
OUT_PATH   = OUTPUT_DIR / "Variable_Glossary.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# Glossary entries.  Each entry:
#   name        — variable name as it appears in the CSVs / model
#   short       — one-line plain-English description
#   long        — multi-line explanation, formula, units
#   source      — where the underlying data comes from
#   expected    — expected sign of effect on SB success (↑ = helps, ↓ = hurts)
# ─────────────────────────────────────────────────────────────────────────────

ENTRIES = [

# ── Core speed / acceleration ────────────────────────────────────────────────
{"section": "Speed & Acceleration"},

{"name": "sprint_speed",
 "short": "Maximum foot-speed during the fastest 1-second window of a play.",
 "long":  "Baseball Savant's 'Sprint Speed'.  Computed only from 'competitive' "
          "runs (Statcast keeps the top ~2/3 of a player's sprints to remove "
          "easy jogs).  Units: feet per second.  League average is roughly "
          "27.0 ft/s.  Elite is 29+.  Naylor sits around 24.4 ft/s, which is "
          "the ~3rd percentile of qualified MLB runners.",
 "source": "pybaseball.statcast_sprint_speed(year, min_opp=1)",
 "expected": "+ on SB success (faster = harder to throw out)",
},

{"name": "speed_capped",
 "short": "sprint_speed clipped at 28 ft/s.",
 "long":  "Empirically the marginal SB-success effect of speed flattens above "
          "~28 ft/s.  We confirmed this with a piecewise hinge regression: "
          "below-28 slope ≈ 0.21 (z-units), above-28 slope ≈ 0.06.  Capping "
          "at 28 prevents the model from over-rewarding speed once you are "
          "already 'fast enough', and lets variables like lead and jump "
          "matter more for elite-speed runners.",
 "source": "min(sprint_speed, 28.0)",
 "expected": "+ but flat at the top",
},

{"name": "accel_0_30  (seconds to 30 ft)",
 "short": "Time, in seconds, to cover the first 30 feet from contact.",
 "long":  "Pulled directly from Statcast running splits as "
          "seconds_since_hit_030.  This is the cleanest single number for "
          "FIRST-BURST acceleration — the part of running that matters MOST "
          "for steals (you are only running ~90 ft and the throw is in the "
          "air long before you hit top speed).  League average ~1.78 s.  "
          "Lower is better.  Naylor 2025: 1.87 (slow on paper) — but he is "
          "fast in proportion to his small steps, see accel_gap below.",
 "source": "statcast_running_splits → seconds_since_hit_030",
 "expected": "− on SB success (less time = better)",
},

{"name": "accel_5_30",
 "short": "Time from 5 ft to 30 ft.  Excludes the bat-flip/box variance.",
 "long":  "Subtracts seconds_since_hit_005 from seconds_since_hit_030 so the "
          "metric is not contaminated by how quickly the runner cleared the "
          "batter's box.  Pure 'first-burst once you are running' speed.",
 "source": "seconds_since_hit_030 − seconds_since_hit_005",
 "expected": "− on SB success",
},

{"name": "maintain_30_90",
 "short": "Time from 30 ft to 90 ft — top-speed maintenance.",
 "long":  "Captures the second half of the sprint, where elite top-end "
          "matters.  For an SB attempt the runner only goes ~85 ft, so this "
          "is partly a proxy for late-burst.  Lower = better.",
 "source": "seconds_since_hit_090 − seconds_since_hit_030",
 "expected": "− on SB success",
},

{"name": "total_90",
 "short": "Total time, contact-to-90-ft (i.e. home-to-first).",
 "long":  "Useful as a single 'how fast does this guy go a long way' number, "
          "but redundant with accel_0_30 + maintain_30_90.  League avg ~4.05 s.",
 "source": "seconds_since_hit_090",
 "expected": "− on SB success",
},

{"name": "accel_gap",
 "short": "Percentile rank of acceleration minus percentile rank of speed.",
 "long":  "POSITIVE  ⇒ runner is faster off the line than their top speed "
          "implies.  Naylor archetype.\n"
          "Formula:   pctile(accel_0_30, inverted)  −  pctile(sprint_speed).\n"
          "We invert accel_0_30 because LOW time = HIGH percentile.\n"
          "Range roughly −80 to +80 percentile points.",
 "source": "Computed from sprint_speed + accel_0_30 percentiles within season",
 "expected": "+ on SB success (independent of raw speed)",
},

{"name": "bolts",
 "short": "Number of plays in a season where the runner exceeded 30 ft/s.",
 "long":  "Baseball Savant's 'Bolts' count.  A simple count of elite-speed "
          "moments.  Most slow runners have 0 bolts.",
 "source": "Sprint-speed table → 'bolts' column",
 "expected": "+ on SB success",
},

# ── Lead distance (REAL in v4, simulated in v3) ──────────────────────────────
{"section": "Lead Distance (real Baseball Savant data, v4)"},

{"name": "primary_lead  /  lead_off_dist",
 "short": "Average distance off first base when the pitcher starts the motion.",
 "long":  "The lead the runner takes BEFORE the pitcher commits.  Real "
          "league average is roughly 11.5 ft.  Aggressive baserunners push "
          "13+ ft; conservative is 10.5 ft.\n"
          "In v3 this was a simulated draw N(lead_tendency_z, 0.6).\n"
          "In v4 we use the real Baseball Savant figure: r_primary_lead.",
 "source": "baseballsavant.mlb.com/leaderboard/basestealing-run-value "
           "→ r_primary_lead (real, 2015 onward)",
 "expected": "+ on SB success (longer head start)",
},

{"name": "secondary_lead",
 "short": "Average distance off first base at the moment of pitch release.",
 "long":  "Primary lead PLUS the ground covered while the pitcher is "
          "delivering.  League average ~14.5 ft.  Elite secondary leads "
          "are 16+ ft.",
 "source": "Baseball Savant → r_secondary_lead",
 "expected": "+ on SB success",
},

{"name": "lead_gain  /  r_sec_minus_prim_lead",
 "short": "DISTANCE COVERED FROM PITCHER'S FIRST MOVE TO PITCH RELEASE.",
 "long":  "secondary_lead − primary_lead.  This is the single best measure "
          "of what the user calls 'jerk' — how much ground the runner "
          "covered in the tiny window where the pitcher is committed but "
          "the ball is not yet released.  Big lead_gain ⇒ the runner read "
          "the pitcher early AND had explosive first steps.\n"
          "Naylor archetype lives here: his lead_gain is consistently >3.5 ft "
          "despite mediocre raw acceleration on a hit ball.",
 "source": "Baseball Savant → r_sec_minus_prim_lead",
 "expected": "+ on SB success (strongly)",
},

{"name": "*_sbx  variants (e.g. r_sec_minus_prim_lead_sbx)",
 "short": "Same metric, restricted to plays that ENDED in a SB attempt.",
 "long":  "Baseball Savant publishes the lead metrics two ways: averaged "
          "over ALL plays the runner had on first (the base metric) and "
          "averaged only over plays where the runner went (the _sbx "
          "variant).  The _sbx version is cleaner for predicting SB "
          "success but smaller-sample.",
 "source": "Baseball Savant",
 "expected": "Same direction as base metric",
},

# ── Jump / reaction ──────────────────────────────────────────────────────────
{"section": "Jump & Reaction"},

{"name": "jump_time  (derived)",
 "short": "Seconds from pitcher's first move to runner's break for second.",
 "long":  "Not directly published by Statcast as a stand-alone leaderboard "
          "field.  We approximate it by combining lead_gain with the "
          "running-splits acceleration profile: time to cover lead_gain "
          "feet from a standing start, given the runner's accel_0_30.\n"
          "Lower = better jump.  Range roughly 0.30 – 0.55 s.",
 "source": "Derived from r_sec_minus_prim_lead + statcast_running_splits",
 "expected": "− on SB success (less time = better)",
},

{"name": "reaction_quality",
 "short": "Composite of jump_time (low) + lead_gain (high).",
 "long":  "v3 used a simulated U(0.6, 0.95).  v4 builds it from real "
          "components.  A 0–1 score where 1 = optimal jump.",
 "source": "v4: composite of real jump_time + lead_gain",
 "expected": "+ on SB success",
},

# ── Pitcher / catcher context ────────────────────────────────────────────────
{"section": "Pitcher & Catcher Context"},

{"name": "pitcher_ttp  /  pitcher_delivery_time",
 "short": "Pitcher Time-To-Plate: seconds from first motion to ball in glove.",
 "long":  "Slow deliveries (≥1.40 s) give the runner more time.  Fast "
          "deliveries (1.15 s slide-step) crush attempts.  League avg ~1.30 s.\n"
          "v3 SIMULATED this with N(1.30, 0.10).\n"
          "v4 DROPPED — couldn't be derived.\n"
          "v5 USES LEAGUE CONSTANT 1.30 s.  Real per-pitch delivery time is "
          "not publicly available; Savant's pitch-tempo CSV is buggy at "
          "source (median_seconds_empty equals median_seconds_onbase for "
          "every row, and year= filter is ignored).",
 "source": "v5: league constant 1.30 s",
 "expected": "+ on SB success (longer TTP helps the runner)",
},

{"name": "pop_time_2b  /  pop_2b_sba",
 "short": "Pop time on stolen-base attempts to 2nd base (sec).",
 "long":  "REAL per-catcher-per-year data (v5).  From Savant's catcher "
          "poptime leaderboard.  Elite catchers run pop times of 1.85 s "
          "and below.  Bad pop times (2.05+) make stealing trivial.  "
          "League avg ~1.95 s.  Available 2018+ (not 2020).",
 "source": "pybaseball.statcast_catcher_poptime(year, min_2b_att)",
 "expected": "− on SB success (faster catcher = harder to steal)",
},

{"name": "exchange_2b_3b_sba",
 "short": "Catcher's glove-to-throwing-hand transfer time (sec).",
 "long":  "Component of pop time.  Real per-catcher (v5).",
 "source": "pybaseball.statcast_catcher_poptime",
 "expected": "− on SB success",
},

{"name": "maxeff_arm_2b_3b_sba",
 "short": "Catcher's max-effort arm strength on SB throws (mph).",
 "long":  "Higher arm speed → faster throw → tighter window for steal.",
 "source": "pybaseball.statcast_catcher_poptime",
 "expected": "− on SB success",
},

{"name": "pitcher_lead_allowed_prim / sec / gain",
 "short": "Per-pitcher career: how much primary, secondary, lead-gain runners take.",
 "long":  "From Savant pitcher-running-game CSV.  Pitchers good at holding "
          "runners SHOW UP HERE with smaller lead values.  Career-aggregate "
          "snapshot (year filter ignored).  Cross-joined to each attempt by "
          "pitcher_id.",
 "source": "https://baseballsavant.mlb.com/leaderboard/pitcher-running-game",
 "expected": "+ for runner side (more lead allowed = easier steal)",
},

{"name": "matchup_lead_diff",
 "short": "runner_lead_gain  −  pitcher_lead_allowed",
 "long":  "Positive = this runner-pitcher matchup favors the runner.  "
          "Negative = the pitcher's running-game suppression overcomes "
          "the runner's lead profile.  v5 matchup feature.",
 "source": "v5: derived",
 "expected": "+ on SB success",
},

# ── Real SB performance / residuals ──────────────────────────────────────────
{"section": "Real SB Performance & Residuals"},

{"name": "SB  /  CS",
 "short": "Real stolen bases / caught stealing in the calendar season.",
 "long":  "Hard counts from the MLB Stats API.  We use them as ground "
          "truth.  In v3 we required SB+CS ≥ 10 to qualify for the model.",
 "source": "statsapi.mlb.com/api/v1/stats?stats=season&group=hitting",
 "expected": "Ground-truth outcome",
},

{"name": "real_sb_pct",
 "short": "Shrunk Bayesian estimate of true SB% with k=5.",
 "long":  "Formula:  (SB + k·LeagueSB%) / (SB + CS + k),  k = 5.\n"
          "Shrinkage prevents 1-for-1 runners from being scored as 100%.\n"
          "League SB% is roughly 78% in the modern era.",
 "source": "Computed from SB/CS",
 "expected": "+ proxy for skill (used as model target sometimes)",
},

{"name": "expected_sb_pct",
 "short": "Polynomial-predicted SB% given only the runner's sprint_speed.",
 "long":  "We fit a 2nd-order polynomial of real_sb_pct on sprint_speed "
          "across all qualified runners.  This is the SB% you'd expect "
          "for a 'replacement-level baserunner' with the same speed.",
 "source": "np.polyfit(sprint_speed, real_sb_pct, deg=2)",
 "expected": "Baseline for residual",
},

{"name": "sb_residual",
 "short": "real_sb_pct  −  expected_sb_pct.  THE key v3 signal.",
 "long":  "Positive sb_residual ⇒ the runner over-performs what their "
          "raw speed predicts.  This is the empirical, speed-adjusted "
          "demonstrated steal skill.  Naylor 2025: +0.108 (huge).  "
          "Soto 2025: +0.105.  This is what the SSSI is built around.",
 "source": "real_sb_pct − expected_sb_pct",
 "expected": "+ direct measure of slow-steal skill",
},

# ── Context dummies / situational ────────────────────────────────────────────
{"section": "Situational Variables"},

{"name": "late_game",
 "short": "Indicator: inning ≥ 7.",
 "long":  "Defenses may be more attentive late; runners may take fewer risks.",
 "source": "Pitch-level data",
 "expected": "Small − on attempts",
},

{"name": "outs",
 "short": "Number of outs (0/1/2) in the half-inning.",
 "long":  "Two-out steals are riskier; managers send less.",
 "source": "Pitch-level data",
 "expected": "Ambiguous on success; − on attempt rate",
},

{"name": "p_throws_L,  stand_L",
 "short": "Dummies: pitcher throws left / batter stands left.",
 "long":  "Lefty pitchers face the runner directly → harder to steal.\n"
          "Lefty batter blocks the catcher's throwing lane → slightly easier.",
 "source": "Pitch-level data",
 "expected": "p_throws_L:  −     stand_L:  +",
},

# ── Compound indices ─────────────────────────────────────────────────────────
{"section": "Compound Skill Indices (SSSI)"},

{"name": "Z-score notation  (suffix _z)",
 "short": "Any variable with _z is standardized to mean 0, SD 1.",
 "long":  "Standardization happens within the qualified-runner pool.  So "
          "lead_off_z = (lead_off_dist − pool mean) / pool SD.  A value of "
          "+2 means '2 standard deviations above league'.  This lets us "
          "add unlike variables (feet, seconds, percentiles) into one "
          "composite score.",
 "source": "sklearn.preprocessing.StandardScaler",
 "expected": "Same direction as un-standardized variable",
},

{"name": "SSSI_v3_fixed",
 "short": "Fixed-weight Slow-Steal Skill Index.",
 "long":  "Plan-default linear combination of z-scores:\n"
          "    0.35 · z(sb_residual)\n"
          "  + 0.25 · z(accel_gap)\n"
          "  + 0.15 · z(lead_gain)\n"
          "  + 0.10 · z(reaction_quality)   (= 'jump')\n"
          "  + 0.10 · z(lead_off_dist)\n"
          "  − 0.05 · z(speed_capped)\n"
          "Higher = better slow-steal archetype.  Designed to reward "
          "skill independent of raw speed.",
 "source": "Computed in v3 pipeline",
 "expected": "+ identifies slow-but-effective basestealers",
},

{"name": "SSSI_v3_opt  (optimised)",
 "short": "Grid-searched weights that maximise Naylor+Soto z-score.",
 "long":  "From 1,728 weight combinations on a 6-d grid we picked:\n"
          "    0.25 · z(sb_residual)\n"
          "  + 0.10 · z(accel_gap)\n"
          "  + 0.05 · z(lead_gain)\n"
          "  + 0.05 · z(reaction_quality)\n"
          "  + 0.20 · z(lead_off_dist)\n"
          "  − 0.20 · z(speed_capped)\n"
          "Under these weights Naylor 2025 ranks #1 (SSSI = 2.10), "
          "Soto 2025 ranks #3 (SSSI = 1.61).",
 "source": "Grid search inside v3 pipeline",
 "expected": "Optimal for the slow-steal archetype",
},

{"name": "lead_tendency_z  (latent variable)",
 "short": "Latent runner-specific tendency to take big leads.",
 "long":  "In v3 this was a random N(0,1) draw — except Naylor and Soto "
          "were anchored at +2.0 (documented elite leads).  In v4 this "
          "is REPLACED with real r_primary_lead z-score and the anchoring "
          "is removed.",
 "source": "v3 simulated; v4 = z(r_primary_lead)",
 "expected": "+ on SB success",
},

# ── Per-pitch features (v5) ──────────────────────────────────────────────────
{"section": "Per-Pitch Features  (v5 only)"},

{"name": "y_attempt  /  y_success",
 "short": "Binary labels per attempt: was there an attempt? did it succeed?",
 "long":  "Parsed from the Statcast `des` (description) text field.  Regex "
          "matches 'steals (N) 2nd base' → SB; 'caught stealing 2nd' → CS; "
          "'picks off ... at 1st' → PK.  CRITICAL LIMITATION: the count, "
          "outs, and inning attached to an attempt are those of the at-bat's "
          "FINAL pitch (where the SB des was recorded), not the actual "
          "mid-at-bat pitch on which the SB happened.",
 "source": "v5: regex on Statcast des field",
 "expected": "Ground-truth outcome (y_success is the target of Model A)",
},

{"name": "balls,  strikes  (count)",
 "short": "Pre-pitch count at the at-bat's final pitch (per-pitch field).",
 "long":  "Real per-pitch from Statcast.  Used to compute count-leverage "
          "(see is_count_HL below).",
 "source": "Statcast",
 "expected": "Context — interaction with attempt rate",
},

{"name": "count_label",
 "short": "String 'balls-strikes', e.g. '1-1', '3-2'.",
 "long":  "Used to bucket pitches into the 12 possible counts.",
 "source": "v5: derived",
 "expected": "Categorical context",
},

{"name": "is_count_HL  (high-leverage count flag)",
 "short": "1 if this count's attempt rate is ≥ 60% of the peak count's rate.",
 "long":  "Per user spec, low-attempt counts are DROPPED from per-attempt "
          "training because they add noise.  The empirical HL set is "
          "computed at runtime; typical HL counts: 0-0, 1-0, 2-1, 1-1, 3-1.",
 "source": "v5: derived from league attempt-rate table",
 "expected": "Flag — used as a feature AND a training filter",
},

{"name": "pct_in_HL  (per runner-season)",
 "short": "Fraction of a runner's attempts that occurred in HL counts.",
 "long":  "Aggregate version of is_count_HL.  Runners who pick their spots "
          "well will have high pct_in_HL.",
 "source": "v5: derived",
 "expected": "+ on overall SB%",
},

{"name": "outs_when_up,  inning,  inning_topbot",
 "short": "Game-state at the moment of the at-bat (real per-pitch).",
 "long":  "Outs:  0/1/2.  Two-out steals are riskier; managers send less.\n"
          "Inning:  1-15+.  Late-game leverage influences attempt rate.\n"
          "Top/bot:  game half.",
 "source": "Statcast per-pitch",
 "expected": "Context",
},

{"name": "pre_rel_vel  (NEW v5 metric)",
 "short": "Runner's avg velocity from pitcher's first move to release  (ft/s).",
 "long":  "Formula:  r_sec_minus_prim_lead  /  pitcher_delivery_proxy_s\n"
          "        = lead_gain (ft) / 1.30 s (league-constant proxy)\n"
          "Variation in this metric is driven ENTIRELY by lead_gain "
          "because we have no real per-pitcher delivery time.  Still "
          "useful as a normalised distance-per-time metric.\n"
          "User's intent: a runner who covers more ground in the same "
          "pitcher-delivery window is more dangerous, even against fast "
          "pitchers.",
 "source": "v5: derived from runner career lead + league constant",
 "expected": "+ on SB success",
},

{"name": "post_rel_dist  (NEW v5 metric)",
 "short": "Distance runner covers during catcher's pop window  (ft).",
 "long":  "Formula:  sprint_speed × pop_2b_sba  −  accel_correction\n"
          "where accel_correction = 0.5 × max(0, accel_0_30 − 1.65)\n"
          "                                × sprint_speed × pop_time.\n"
          "Slower-acceleration runners haven't reached top speed yet so "
          "the naive sprint × pop product over-counts.  The correction "
          "scales the penalty with the runner's accel-time deficit.\n"
          "User's intent: even a slow runner who covers a lot of ground "
          "during the pop window is dangerous on the back end.",
 "source": "v5: derived from real per-attempt pop + runner profile",
 "expected": "+ on SB success",
},

{"name": "pre_rel_vel_avg,  post_rel_dist_avg  (per runner-season)",
 "short": "Per-attempt mean of the above two metrics over a season.",
 "long":  "Aggregated for use in Model B (season-level GBM) and SSSI v5.",
 "source": "v5: groupby(runner_id, season).mean()",
 "expected": "+ on overall SB%",
},

{"name": "avg_pop_faced",
 "short": "Average catcher pop time across this runner's SB attempts.",
 "long":  "Real per-attempt catcher pop (v5) → mean per runner-season.  "
          "Runners who attack weak-arm catchers will have HIGH avg_pop_faced.",
 "source": "v5: derived from per-attempt join",
 "expected": "+ on overall SB%",
},

{"name": "seconds_since_hit_005  …  seconds_since_hit_090",
 "short": "18 raw Statcast running-splits columns (5/10/15/.../90 ft).",
 "long":  "Each column is seconds elapsed from contact at that distance "
          "milestone.  v5 explores three representations of this curve "
          "(see curve_a, curve_b, curve_c below).  Lower = faster.",
 "source": "pybaseball.statcast_running_splits(year, raw_splits=True)",
 "expected": "− on SB success (less time = better)",
},

{"name": "curve_a,  curve_b,  curve_c,  curve_rmse",
 "short": "Quadratic fit to runner's velocity curve over 5-90 ft.",
 "long":  "Fits time = a·d² + b·d + c to the 18 split points per runner-season.\n"
          "a captures curvature (fatigue / late drag).\n"
          "b is the linear time/distance slope (≈ 1/avg-speed).\n"
          "c is the intercept (initial reaction time after contact).\n"
          "rmse measures how well a quadratic fits the actual splits.\n"
          "v5 'Path B' representation of the splits.",
 "source": "v5: np.polyfit per row across 18 split columns",
 "expected": "Combined ≈ 1 strong feature",
},

{"name": "SSSI_v5",
 "short": "v5 composite — adds pre_rel_vel_z and post_rel_dist_z to v4.",
 "long":  "Linear combination of 8 z-scored terms.  Weights are grid-searched "
          "on 80% of runners (Naylor and Soto are HELD OUT from the grid "
          "search to avoid overfitting them).",
 "source": "Computed in v5_explore.py",
 "expected": "+ identifies slow-steal archetype with new metrics included",
},

]

# ─────────────────────────────────────────────────────────────────────────────
# Render to PDF
# ─────────────────────────────────────────────────────────────────────────────
def render(entries, path):
    PAGE_W, PAGE_H = 8.5, 11
    LEFT, RIGHT, TOP, BOTTOM = 0.6, 0.6, 0.6, 0.6
    LINE_H = 0.16        # inches per text line approx
    PER_PAGE_USABLE = PAGE_H - TOP - BOTTOM

    with PdfPages(path) as pdf:
        # Cover page
        fig = plt.figure(figsize=(PAGE_W, PAGE_H)); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.text(0.5, 0.78, "The Naylor Model", ha="center", va="center",
                fontsize=28, fontweight="bold")
        ax.text(0.5, 0.72, "Variable Glossary", ha="center", va="center",
                fontsize=22)
        ax.text(0.5, 0.65, "Reference for every column in the v3 / v4 / v5 model",
                ha="center", va="center", fontsize=12, style="italic",
                color="#444")
        ax.text(0.5, 0.30,
                "Every variable has:\n"
                "  •  a plain-English definition\n"
                "  •  the exact formula and units\n"
                "  •  the data source\n"
                "  •  the expected direction of effect on steal success",
                ha="center", va="center", fontsize=11, color="#222",
                linespacing=1.6)
        ax.text(0.5, 0.10,
                "Generated for The Naylor Model · 2026",
                ha="center", va="center", fontsize=9, color="#888")
        pdf.savefig(fig); plt.close(fig)

        # Content pages — flow entries page by page
        y_cursor = PAGE_H - TOP
        fig = plt.figure(figsize=(PAGE_W, PAGE_H)); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

        def new_page():
            nonlocal fig, ax, y_cursor
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(PAGE_W, PAGE_H)); fig.patch.set_facecolor("white")
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            y_cursor = PAGE_H - TOP

        def need(space_in):
            nonlocal y_cursor
            if y_cursor - space_in < BOTTOM:
                new_page()

        for e in entries:
            if "section" in e:
                need(0.7)
                # section bar
                y_cursor -= 0.35
                ax.add_patch(plt.Rectangle(
                    (LEFT/PAGE_W, y_cursor/PAGE_H),
                    (PAGE_W - LEFT - RIGHT)/PAGE_W, 0.025,
                    transform=fig.transFigure, facecolor="#1F3A5F",
                    edgecolor="none"))
                ax.text(LEFT/PAGE_W + 0.01, y_cursor/PAGE_H + 0.008,
                        e["section"], transform=fig.transFigure,
                        fontsize=14, fontweight="bold", color="white",
                        va="center")
                y_cursor -= 0.20
                continue

            # estimate height: name (1) + short (1) + long (~3-6) + source (1) + expected (1) + spacing
            long_lines = sum(len(ln)//90 + 1 for ln in e["long"].split("\n"))
            est_h = 0.18 + 0.16 + 0.14*long_lines + 0.14 + 0.14 + 0.16
            need(est_h)

            # name (bold, large)
            y_cursor -= 0.20
            ax.text(LEFT/PAGE_W, y_cursor/PAGE_H, e["name"],
                    transform=fig.transFigure, fontsize=12.5, fontweight="bold",
                    color="#0B2545", va="center")
            # short (italic)
            y_cursor -= 0.18
            ax.text(LEFT/PAGE_W, y_cursor/PAGE_H, e["short"],
                    transform=fig.transFigure, fontsize=10.5, style="italic",
                    color="#222", va="center")
            # long (wrap manually)
            for raw_line in e["long"].split("\n"):
                # naive wrap at 95 chars
                line = raw_line
                while len(line) > 95:
                    cut = line.rfind(" ", 0, 95)
                    if cut < 30: cut = 95
                    y_cursor -= 0.14
                    ax.text(LEFT/PAGE_W, y_cursor/PAGE_H, line[:cut],
                            transform=fig.transFigure, fontsize=9.5,
                            color="#333", va="center", family="serif")
                    line = line[cut:].lstrip()
                y_cursor -= 0.14
                ax.text(LEFT/PAGE_W, y_cursor/PAGE_H, line,
                        transform=fig.transFigure, fontsize=9.5,
                        color="#333", va="center", family="serif")
            # source
            y_cursor -= 0.14
            ax.text(LEFT/PAGE_W, y_cursor/PAGE_H,
                    f"Source:   {e['source']}",
                    transform=fig.transFigure, fontsize=9, color="#555",
                    va="center", family="monospace")
            # expected
            y_cursor -= 0.14
            ax.text(LEFT/PAGE_W, y_cursor/PAGE_H,
                    f"Expected effect:   {e['expected']}",
                    transform=fig.transFigure, fontsize=9, color="#555",
                    va="center")
            y_cursor -= 0.10  # spacer

        pdf.savefig(fig); plt.close(fig)

    print(f"Wrote {path}")

if __name__ == "__main__":
    render(ENTRIES, OUT_PATH)
