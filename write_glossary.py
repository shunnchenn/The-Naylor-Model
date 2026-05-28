#!/usr/bin/env python3
"""
Variable Glossary  —  The Naylor Model  ·  v6  (Statcast-style)
================================================================

Re-written for v6 with the goal of being readable by anyone who watches
baseball.  No statistical jargon.  Every variable has:

  ▸ Plain-English definition       (what is this?)
  ▸ Units                          (how is it measured?)
  ▸ League average                 (what's normal?)
  ▸ Elite threshold                (what's great?)
  ▸ Real example                   (which player has this number?)
  ▸ Tier chart                     (Elite / Above Avg / Avg / Below / Poor)
  ▸ How it affects steal success   (why does it matter?)

The companion model report uses two columns instead of "coef_z":

    SB% Boost per Tier   = pp change in predicted SB success rate
                            when a runner improves the feature by 1 tier
                            (1 standard deviation).
    Odds Multiplier      = same idea, multiplicative on the odds.
                            >1 means the feature helps, <1 means it hurts.

Both are computed from the same underlying logistic-regression coefficient.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

OUTPUT_DIR = Path("/Users/shunchen/Desktop/The-Naylor-Model")
OUT_PATH   = OUTPUT_DIR / "reports" / "Variable_Glossary.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# Statcast-style entries.
#
#   name        — variable name as it appears in the CSVs / model
#   plain       — 1 sentence in everyday language
#   units       — e.g. "ft/s", "ft", "seconds", "count"
#   avg         — league-average value (typical runner)
#   elite       — what counts as elite
#   example     — "Naylor 2025: 24.4 ft/s (3rd percentile)"
#   tiers       — list of (label, range_string)
#   impact      — how this variable affects SB success in plain English
#   source      — where the data comes from
# ─────────────────────────────────────────────────────────────────────────────

ENTRIES = [

# ── Speed & Acceleration ───────────────────────────────────────────────────
{"section": "Speed & Acceleration"},

{"name":   "Sprint Speed",
 "plain":  "Top running speed measured during the player's fastest sprints.",
 "units":  "feet per second (ft/s)",
 "avg":    "27.0",
 "elite":  "29.0 +",
 "example":"Bobby Witt Jr.: 30.2  (elite)   ·   Josh Naylor: 24.4  (3rd %ile)",
 "tiers":  [("Elite",       "29.0 +"),
            ("Above avg",   "27.5 – 29.0"),
            ("Average",     "26.5 – 27.5"),
            ("Below avg",   "25.5 – 26.5"),
            ("Poor",        "≤ 25.5")],
 "impact": "Faster runners are obviously harder to throw out, but the model "
           "finds it matters LESS than raw acceleration on a 90-ft steal.  "
           "Once you're above ~28 ft/s the extra speed is wasted because "
           "you're past second base before the throw arrives anyway.",
 "source": "Baseball Savant Sprint Speed leaderboard",
},

{"name":   "Speed Capped",
 "plain":  "Sprint speed, but anything over 28 ft/s is clipped to 28.",
 "units":  "ft/s",
 "avg":    "27.0",
 "elite":  "28.0 (capped)",
 "example":"Witt 30.2 → capped at 28.0   ·   Naylor 24.4 → unchanged",
 "tiers":  [("≥ 28 ft/s",   "treated as 28.0 (no extra reward)"),
            ("< 28 ft/s",   "kept as-is — every fraction matters")],
 "impact": "The model adds this version because the marginal benefit of "
           "more speed vanishes above 28 ft/s.  Capping prevents the model "
           "from over-rewarding pure speedsters and lets it find smaller "
           "advantages like jump and lead.",
 "source": "computed from Sprint Speed",
},

{"name":   "Jump Time  (accel_0_30)",
 "plain":  "How fast you cover the first 30 feet from contact.",
 "units":  "seconds",
 "avg":    "1.78",
 "elite":  "≤ 1.65",
 "example":"Corbin Carroll: 1.67  ·  Josh Naylor: 1.87 (slow)  ·  Nasim Nuñez: 1.64 (elite)",
 "tiers":  [("Elite",       "≤ 1.68"),
            ("Above avg",   "1.69 – 1.74"),
            ("Average",     "1.75 – 1.80"),
            ("Below avg",   "1.81 – 1.86"),
            ("Poor",        "≥ 1.87")],
 "impact": "The single most important running metric for steals — the throw "
           "is in the air before you hit top speed, so what happens in the "
           "first 30 ft decides everything.  In the v5/v6 simple GLM this "
           "feature has the strongest weight of any speed variable.",
 "source": "Statcast Running Splits  ·  seconds_since_hit_030",
},

{"name":   "Acceleration Phase  (accel_5_30)",
 "plain":  "Time from 5 ft to 30 ft — pure first-burst, excluding bat-flip.",
 "units":  "seconds",
 "avg":    "1.32",
 "elite":  "≤ 1.22",
 "example":"Witt: 1.23 (elite)  ·  Naylor: 1.38",
 "tiers":  [("Elite",       "≤ 1.22"),
            ("Above avg",   "1.23 – 1.28"),
            ("Average",     "1.29 – 1.34"),
            ("Below avg",   "1.35 – 1.40"),
            ("Poor",        "≥ 1.41")],
 "impact": "Cleaner version of Jump Time — removes how quickly the runner "
           "cleared the batter's box.  Less noisy proxy of biomechanical "
           "first-burst.",
 "source": "Statcast Running Splits  ·  030 − 005",
},

{"name":   "Top-Speed Phase  (maintain_30_90)",
 "plain":  "Time from 30 ft to 90 ft — back half of the sprint.",
 "units":  "seconds",
 "avg":    "2.27",
 "elite":  "≤ 2.15",
 "example":"Witt: 2.10 (elite)  ·  Naylor: 2.44",
 "tiers":  [("Elite",       "≤ 2.15"),
            ("Above avg",   "2.16 – 2.22"),
            ("Average",     "2.23 – 2.30"),
            ("Below avg",   "2.31 – 2.38"),
            ("Poor",        "≥ 2.39")],
 "impact": "Matters less for SBs than Jump Time — a steal of 2nd only "
           "requires covering ~85 ft.  But it captures conditioning and "
           "shows up in pre-2023 data as a stronger predictor than today.",
 "source": "Statcast Running Splits  ·  090 − 030",
},

{"name":   "Total 90 (total_90)",
 "plain":  "Full home-to-first time from contact.",
 "units":  "seconds",
 "avg":    "4.05",
 "elite":  "≤ 3.95",
 "example":"Carroll: 3.94 (elite)  ·  Naylor: 4.31 (15th percentile)",
 "tiers":  [("Elite",       "≤ 3.95"),
            ("Above avg",   "3.96 – 4.02"),
            ("Average",     "4.03 – 4.10"),
            ("Below avg",   "4.11 – 4.18"),
            ("Poor",        "≥ 4.19")],
 "impact": "A summary number — equals Jump Time + Top-Speed Phase.  Useful "
           "for ranking 'overall fast' runners.  Naylor's 4.31 puts him in "
           "the bottom 25%, yet his accel_gap (below) shows he's still "
           "quick OFF THE LINE — the steal-relevant part.",
 "source": "Statcast Running Splits  ·  seconds_since_hit_090",
},

{"name":   "Accel Gap",
 "plain":  "How much faster the runner is off the line than their top speed implies.",
 "units":  "percentile points",
 "avg":    "0",
 "elite":  "+ 25 or more",
 "example":"Naylor: +22 to +30 across seasons  ·  Witt: −20 (slower off the line for his speed)",
 "tiers":  [("Naylor archetype",  "+15 or more  (fast jump despite low speed)"),
            ("Above expected",    "+5 to +15"),
            ("As expected",       "−5 to +5"),
            ("Below expected",    "−15 to −5"),
            ("Pure speedster",    "≤ −15 (fast top speed but slow start)")],
 "impact": "This is the heart of the Naylor archetype.  POSITIVE values "
           "mean the runner is faster off the line than their top speed "
           "would predict — exactly what a slow-but-effective stealer "
           "looks like.  Pure speedsters often have NEGATIVE accel_gap "
           "because they coast on top-end speed.",
 "source": "computed:  pctile(Jump Time, inverted)  −  pctile(Sprint Speed)",
},

{"name":   "Bolts",
 "plain":  "Number of plays in a season where the runner topped 30 ft/s.",
 "units":  "count",
 "avg":    "13",
 "elite":  "100 +",
 "example":"Witt: 250+  ·  Carroll: 200+  ·  Naylor: 0 (never hit 30 ft/s)",
 "tiers":  [("Elite",       "100 +"),
            ("Above avg",   "30 – 100"),
            ("Average",     "5 – 30"),
            ("Below avg",   "1 – 5"),
            ("Never",       "0")],
 "impact": "A simple count of elite-speed moments.  Big leaderboard "
           "presence; the model treats it as a speed credential.  Zero "
           "bolts doesn't disqualify you (Naylor proves that), but it "
           "puts you in the slow-steal archetype the model is designed "
           "to identify.",
 "source": "Baseball Savant",
},

# ── Lead Distance ──────────────────────────────────────────────────────────
{"section": "Lead Distance  (real Baseball Savant)"},

{"name":   "Primary Lead",
 "plain":  "Average distance off first base when the pitcher starts his motion.",
 "units":  "feet (ft)",
 "avg":    "11.5",
 "elite":  "13.0 +",
 "example":"Trea Turner: 13.1  (elite)  ·  Mookie Betts: 12.4  ·  Naylor: 9.8  (small lead)",
 "tiers":  [("Aggressive",  "13.0 +"),
            ("Above avg",   "12.0 – 13.0"),
            ("Average",     "11.0 – 12.0"),
            ("Conservative","10.0 – 11.0"),
            ("Very small",  "≤ 10.0")],
 "impact": "Bigger lead = less distance to cover.  But coaches push leads "
           "to where the pitcher gets nervous, so big leads can also "
           "increase pickoff risk.  Notably Naylor has one of the SMALLEST "
           "primary leads in MLB (9.8 ft) yet steals 90%+ — his edge is "
           "elsewhere.",
 "source": "Savant Basestealing Run Value · r_primary_lead "
           "(career snapshot — Savant ignores year filter on this endpoint)",
},

{"name":   "Secondary Lead",
 "plain":  "Average distance off first base at the moment of pitch release.",
 "units":  "feet (ft)",
 "avg":    "14.5",
 "elite":  "16.0 +",
 "example":"Mookie Betts: 18.1 (elite)  ·  Naylor: 13.7",
 "tiers":  [("Elite",       "16.0 +"),
            ("Above avg",   "15.0 – 16.0"),
            ("Average",     "14.0 – 15.0"),
            ("Below avg",   "13.0 – 14.0"),
            ("Poor",        "≤ 13.0")],
 "impact": "Primary lead PLUS ground covered while the pitcher is "
           "delivering.  Big secondary leads compress the steal distance "
           "AND give the runner momentum.",
 "source": "Savant · r_secondary_lead (career snapshot)",
},

{"name":   "Lead Gain  (the 'jerk' metric)",
 "plain":  "Distance covered between pitcher's first move and pitch release.",
 "units":  "feet (ft)",
 "avg":    "3.5",
 "elite":  "4.5 +",
 "example":"Mookie Betts: 5.6 (elite — reads pitchers)  ·  Naylor: 3.9 (above avg)",
 "tiers":  [("Elite",       "4.5 +"),
            ("Above avg",   "3.8 – 4.5"),
            ("Average",     "3.2 – 3.8"),
            ("Below avg",   "2.6 – 3.2"),
            ("Poor",        "≤ 2.6")],
 "impact": "The user's 'jerk' metric.  Captures TWO things at once: how "
           "well the runner READS the pitcher's first move AND how "
           "explosively they take their secondary lead.  Big lead-gain "
           "is a hallmark of smart baserunners (Betts, Chisholm Jr., "
           "Frazier, Arraez).",
 "source": "Savant · r_secondary_lead − r_primary_lead",
},

# ── New v5/v6 metrics ──────────────────────────────────────────────────────
{"section": "v5 / v6 NEW Metrics"},

{"name":   "Pre-Release Velocity  (pre_rel_vel)",
 "plain":  "How much ground the runner covers per second between pitcher "
           "first move and pitch release.",
 "units":  "ft / sec",
 "avg":    "2.7",
 "elite":  "3.5 +",
 "example":"Jazz Chisholm Jr. 2025: 4.34 (elite)  ·  Naylor: 2.98  ·  Soto: 2.39",
 "tiers":  [("Elite",       "3.5 +"),
            ("Above avg",   "3.0 – 3.5"),
            ("Average",     "2.5 – 3.0"),
            ("Below avg",   "2.0 – 2.5"),
            ("Poor",        "≤ 2.0")],
 "impact": "Even if a pitcher has a fast delivery, a runner who covers a "
           "lot of ground in that small window is still dangerous.  We use "
           "a league-constant delivery (1.30 s) as the divisor because "
           "per-pitch pitcher TTP isn't publicly available — so variation "
           "is driven by Lead Gain.  The leaderboard separates 'smart' "
           "baserunners from pure speedsters.",
 "source": "Lead Gain  ÷  1.30 s",
},

{"name":   "Post-Release Distance  (post_rel_dist)",
 "plain":  "How much ground the runner covers during the catcher's pop time.",
 "units":  "feet (ft)",
 "avg":    "51.5",
 "elite":  "58.0 +",
 "example":"Victor Scott II 2025: 59.5 (elite)  ·  Naylor: 41.5 (last %ile)",
 "tiers":  [("Elite",       "58.0 +"),
            ("Above avg",   "54.0 – 58.0"),
            ("Average",     "50.0 – 54.0"),
            ("Below avg",   "46.0 – 50.0"),
            ("Poor",        "≤ 46.0")],
 "impact": "After the pitcher releases the ball, the runner has ~1.95 "
           "seconds (the catcher's pop time) to cover as much ground as "
           "possible before the throw arrives.  Slower runners still "
           "accelerating during this window cover LESS than sprint × pop "
           "implies — we subtract an acceleration tax for them.  This is "
           "the speedster-archetype metric: Witt, Carroll, Scott II "
           "dominate.",
 "source": "Sprint Speed × catcher pop_2b_sba − acceleration correction",
},

{"name":   "3-2 Count Attempt Share  (pct_in_HL)",
 "plain":  "Share of a runner's stolen-base attempts that came on a 3-2 (full) count.",
 "units":  "fraction (0.0 – 1.0)",
 "avg":    "≈ 0.55",
 "elite":  "varies",
 "example":"Naylor 2023:  100%  ·  Naylor 2025:  0%  ·  Most runners: 50–80%",
 "tiers":  [("Heavy 3-2 stealer",  "0.75 +"),
            ("Mostly 3-2",        "0.50 – 0.75"),
            ("Balanced",          "0.25 – 0.50"),
            ("Rarely 3-2",        "0.05 – 0.25"),
            ("Never 3-2",         "0.00")],
 "impact": "WHAT THIS REALLY MEASURES — there's a Statcast quirk: when we "
           "parse SB attempts from play descriptions, the count attached "
           "is the AT-BAT's FINAL count, not the count when the runner "
           "actually broke for second.  So an SB that happens mid-AT-BAT "
           "where the AB ended on 3-2 looks like 'an SB on a 3-2 count' "
           "even if it really happened on 0-0.  This means pct_in_HL is "
           "really 'fraction of this runner's attempts where the AB "
           "happened to end on 3-2'.  It carries a small signal but mostly "
           "reflects how often the at-bat went to full count.  "
           "v6 plan: fix by pulling MLB Stats API play-by-play.",
 "source": "regex on Statcast `des` field",
},

# ── Real SB performance / residuals ────────────────────────────────────────
{"section": "Real Stolen-Base Performance"},

{"name":   "SB / CS",
 "plain":  "Hard counts: stolen bases and caught stealings in the season.",
 "units":  "count",
 "avg":    "varies",
 "elite":  "30 SB +",
 "example":"Naylor 2025:  30 SB / 2 CS = 93.8% success",
 "tiers":  [("Elite volume",  "30 + SB"),
            ("Above avg",     "20 – 30 SB"),
            ("Average",       "10 – 20 SB"),
            ("Low",           "< 10 SB (excluded from model)")],
 "impact": "Ground truth.  We require SB + CS ≥ 10 in a season for the "
           "runner-season to qualify (smaller samples are too noisy).",
 "source": "MLB Stats API",
},

{"name":   "Real SB %  (shrunk)",
 "plain":  "Stolen-base success rate, adjusted to prevent tiny samples from looking 100%.",
 "units":  "fraction (0–1)",
 "avg":    "0.78",
 "elite":  "0.85 +",
 "example":"Naylor 2025:  0.917 (30 SB, 2 CS)",
 "tiers":  [("Elite",     "0.88 +"),
            ("Above avg", "0.80 – 0.88"),
            ("Average",   "0.72 – 0.80"),
            ("Below avg", "0.65 – 0.72"),
            ("Poor",      "≤ 0.65")],
 "impact": "Shrinkage: a 1-for-1 runner doesn't look like 100%.  Formula: "
           "(SB + 5·league%) ÷ (SB + CS + 5).  League-mean prior 'pulls' "
           "small samples toward 78%.",
 "source": "computed",
},

{"name":   "Expected SB %",
 "plain":  "The SB% you'd expect from this runner based ONLY on their speed.",
 "units":  "fraction (0–1)",
 "avg":    "≈ Real SB %",
 "elite":  "n/a",
 "example":"Naylor 2025 expected:  0.79 (3rd %ile speed)  ·  actual: 0.92",
 "tiers":  [("Used as a baseline", "compare against Real SB% to get residual")],
 "impact": "Baseline.  Subtracting from Real SB% gives the 'speed-adjusted "
           "skill' signal.",
 "source": "2nd-order polynomial fit on Sprint Speed across qualified runners",
},

{"name":   "SB Residual  (the KEY signal)",
 "plain":  "How much better (or worse) the runner is than their speed predicts.",
 "units":  "fraction (typically −0.20 to +0.20)",
 "avg":    "0",
 "elite":  "+0.10 or more",
 "example":"Naylor 2025: +0.122 (huge overperformer)  ·  Soto 2025: +0.128",
 "tiers":  [("Massive overperformer",  "+0.10 +"),
            ("Above expectation",      "+0.04 to +0.10"),
            ("As expected",            "−0.04 to +0.04"),
            ("Below expectation",      "−0.10 to −0.04"),
            ("Far below",              "≤ −0.10")],
 "impact": "POSITIVE values mean the runner steals MORE successfully than "
           "their raw speed suggests.  This is the Naylor / Soto story.  "
           "We weight it heavily in the SSSI index.",
 "source": "Real SB%  −  Expected SB%",
},

# ── Battery context (real per-attempt) ────────────────────────────────────
{"section": "Battery Context  (real per-attempt)"},

{"name":   "Catcher Pop Time  (pop_2b_sba)",
 "plain":  "Catcher's time from glove to second base on a steal attempt.",
 "units":  "seconds",
 "avg":    "1.95",
 "elite":  "≤ 1.85",
 "example":"Patrick Bailey: 1.85 (elite)  ·  league avg: 1.95",
 "tiers":  [("Elite arm",   "≤ 1.85"),
            ("Above avg",   "1.86 – 1.92"),
            ("Average",     "1.93 – 1.97"),
            ("Below avg",   "1.98 – 2.04"),
            ("Poor",        "≥ 2.05")],
 "impact": "REAL per-catcher-per-year data starting 2018.  Low pop time = "
           "throw arrives at 2B sooner = harder to steal.  In Model A "
           "this is one of the top 5 features.",
 "source": "Statcast Catcher Poptime leaderboard",
},

{"name":   "Catcher Arm Strength  (maxeff_arm_2b_3b_sba)",
 "plain":  "Maximum-effort throw velocity to second base.",
 "units":  "mph",
 "avg":    "84.0",
 "elite":  "88.0 +",
 "example":"Korey Lee 2024: 88.3 mph (elite)",
 "tiers":  [("Elite",       "88.0 +"),
            ("Above avg",   "85.0 – 88.0"),
            ("Average",     "82.0 – 85.0"),
            ("Below avg",   "79.0 – 82.0"),
            ("Poor",        "≤ 79.0")],
 "impact": "Component of Pop Time.  Some catchers have weak arms but quick "
           "transfers; some have rockets but slow exchanges.  v5/v6 use "
           "both as separate features.",
 "source": "Statcast Catcher Poptime",
},

{"name":   "Catcher Exchange  (exchange_2b_3b_sba)",
 "plain":  "Time from glove receive to throwing-hand release.",
 "units":  "seconds",
 "avg":    "0.64",
 "elite":  "≤ 0.60",
 "example":"Patrick Bailey 2024: 0.60",
 "tiers":  [("Elite",       "≤ 0.60"),
            ("Above avg",   "0.61 – 0.63"),
            ("Average",     "0.64 – 0.66"),
            ("Below avg",   "0.67 – 0.69"),
            ("Poor",        "≥ 0.70")],
 "impact": "Fast exchange compensates for a weak arm and vice-versa.",
 "source": "Statcast Catcher Poptime",
},

{"name":   "Pitcher Lead Allowed",
 "plain":  "Average lead distance runners take against THIS pitcher (career).",
 "units":  "feet (ft)",
 "avg":    "3.5  (lead gain allowed)",
 "elite":  "≤ 2.8  (pitcher is great at suppressing leads)",
 "example":"Best at holding runners → low value;  Slow to plate → high value",
 "tiers":  [("Suppresses leads", "≤ 2.8"),
            ("Above avg",        "2.8 – 3.3"),
            ("Average",          "3.3 – 3.7"),
            ("Below avg",        "3.7 – 4.1"),
            ("Gives up leads",   "≥ 4.1")],
 "impact": "Career-aggregate.  When this number is small, the pitcher is "
           "good at holding runners.  Big = runners feast.  Joined to "
           "each SB attempt for matchup-level modelling.",
 "source": "Savant Pitcher Running-Game Run Value",
},

{"name":   "Pitcher TTP  (Time-to-Plate)",
 "plain":  "Seconds from pitcher's first motion to ball-in-catcher's-glove.",
 "units":  "seconds",
 "avg":    "1.30",
 "elite":  "≤ 1.15  (slide-step)",
 "example":"league constant used in v5/v6 (real per-pitch not published)",
 "tiers":  [("Slide step",  "≤ 1.15"),
            ("Fast",        "1.16 – 1.25"),
            ("Average",     "1.26 – 1.35"),
            ("Slow",        "1.36 – 1.45"),
            ("Very slow",   "≥ 1.46")],
 "impact": "Real per-pitch TTP would be the single most useful new data "
           "source — but Statcast doesn't publish it.  Savant's "
           "pitch-tempo CSV is buggy at source (the 'with-runners-on' "
           "column equals the 'bases-empty' column for every row).  "
           "v6 substitutes the league constant 1.30 s.",
 "source": "league constant 1.30 (v6)",
},

# ── Situational / context ─────────────────────────────────────────────────
{"section": "Game Situation"},

{"name":   "Inning",
 "plain":  "Which inning the play occurred.",
 "units":  "integer 1 – 15",
 "avg":    "—",
 "elite":  "—",
 "example":"Late-game (7+) shows different attempt rates than early.",
 "tiers":  [("Early",       "1 – 3"),
            ("Middle",      "4 – 6"),
            ("Late",        "7 – 9"),
            ("Extras",      "10 +")],
 "impact": "Surprisingly important in Model A — the highest SHAP feature.  "
           "Late-inning leverage changes manager decisions and runner "
           "risk tolerance.",
 "source": "Statcast pitch-level",
},

{"name":   "Outs (outs_when_up)",
 "plain":  "Number of outs in the half-inning (0, 1, or 2).",
 "units":  "0 / 1 / 2",
 "avg":    "—",
 "elite":  "—",
 "example":"Two-out steals are riskier; managers send less often.",
 "tiers":  [("0 outs",  "highest attempt rate"),
            ("1 out",   "average"),
            ("2 outs",  "lowest — risk of ending inning")],
 "impact": "Used as a feature; not a huge signal but small reliable effect.",
 "source": "Statcast pitch-level",
},

{"name":   "Balls / Strikes  (count)",
 "plain":  "Number of balls / strikes BEFORE the current pitch.",
 "units":  "integers 0–3 / 0–2",
 "avg":    "—",
 "elite":  "—",
 "example":"3-2 (full count) is by far the most common SB-attempt context.",
 "tiers":  [("Full count 3-2",   "highest by far (1.9% per pitch)"),
            ("Two-strike",       "0.27 – 0.29%"),
            ("All other counts", "≈ 0% (data quirk — see 3-2 share above)")],
 "impact": "Feeds into 3-2 Count Attempt Share.  See limitation note "
           "in that entry — exact count at SB is not in `des` text.",
 "source": "Statcast pitch-level",
},

# ── Compound indices ──────────────────────────────────────────────────────
{"section": "Composite Scores"},

{"name":   "SSSI v5  (Slow-Steal Skill Index)",
 "plain":  "A weighted composite score that identifies slow-but-effective stealers.",
 "units":  "z-score units (typically −3 to +3)",
 "avg":    "0",
 "elite":  "+2.0 or more",
 "example":"Naylor 2025: +1.99  ·  Soto 2025: +1.26  ·  league mean: 0",
 "tiers":  [("Elite slow-steal", "+2.0 +"),
            ("Above avg",        "+1.0 to +2.0"),
            ("Average",          "−0.5 to +1.0"),
            ("Below avg",        "−1.5 to −0.5"),
            ("Poor",             "≤ −1.5")],
 "impact": "Weighted average of eight standardized features (see v5 "
           "report).  Designed to find the Naylor / Soto archetype: "
           "elite-performing slow runners.  Weights were tuned on 80% "
           "of runners with Naylor + Soto HELD OUT to avoid overfitting.",
 "source": "computed in v5/v6 pipeline",
},

# ── Coefficient interpretation ────────────────────────────────────────────
{"section": "How to Read the Model's Weight Table"},

{"name":   "SB % Boost per Tier  (standardised logit coefficient)",
 "plain":  "If a runner improves this feature by 1 tier "
           "(1 standard deviation), how much does the model's predicted "
           "SB success rate go up?",
 "units":  "percentage points (pp)",
 "avg":    "0",
 "elite":  "+5 pp or more = a strongly helpful feature",
 "example":"Jump Time has  −10.6 pp  → improving Jump Time by 1 tier "
           "raises predicted SB% from 62% to 51% (-10.6 pp).  Wait, that's "
           "WORSE?  No — Jump Time is in seconds, and LOWER is better.  "
           "So a 1-tier INCREASE (slower) hurts by 10.6 pp.  The model "
           "shows direction with a sign.",
 "tiers":  [("Strongly helps",   "+5 pp +"),
            ("Helps a little",   "+1 to +5 pp"),
            ("Neutral",          "−1 to +1 pp"),
            ("Hurts a little",   "−5 to −1 pp"),
            ("Hurts a lot",      "≤ −5 pp")],
 "impact": "Replaces the old 'coef_z' column.  Plain-English version of "
           "the same number.  Compute as:\n"
           "    P_baseline = sigmoid(intercept)\n"
           "    P_after    = sigmoid(intercept + coef)\n"
           "    boost_pp   = (P_after − P_baseline) × 100",
 "source": "logistic regression coefficient × baseline probability slope",
},

{"name":   "Odds Multiplier  (odds ratio per SD)",
 "plain":  "If a runner improves by 1 tier, what happens to their "
           "ODDS of success?",
 "units":  "multiplicative factor",
 "avg":    "1.0",
 "elite":  "1.20 +  = feature multiplies odds by 20%",
 "example":"Lead Gain has Odds Multiplier 1.16  → improving by 1 tier "
           "multiplies the runner's success-odds by 1.16.\n"
           "Jump Time has Odds Multiplier 0.63  → 1 tier SLOWER cuts the "
           "odds to 63% of baseline.",
 "tiers":  [("Strongly helps",   "1.20 +"),
            ("Helps",            "1.05 – 1.20"),
            ("Neutral",          "0.95 – 1.05"),
            ("Hurts",            "0.80 – 0.95"),
            ("Strongly hurts",   "≤ 0.80")],
 "impact": "Replaces the old 'OR/SD' column.  Multiply with current odds "
           "to get new odds.  Equivalent to e^coefficient.",
 "source": "exp(logistic coefficient)",
},



# ── Model Discussion ──────────────────────────────────────────────────────────
{"section": "Understanding the Results"},

{"discussion": "The Naylor Model: Why It Exists",
 "paragraphs": [
     ("The Central Question", False,
      "Josh Naylor stole 30 bases in 2025 at a 93.8% success rate. "
      "Baseball Savant ranks him in the 3rd percentile for sprint speed — meaning roughly 97% of "
      "major leaguers are faster on their feet. Yet he is one of the most effective base stealers "
      "in the game. Standard leaderboards that sort by raw steal totals or raw success rate would "
      "never flag him as elite, because they compare him against fast runners operating in a "
      "completely different context. The Naylor Model was built to answer a specific question: "
      "which runners are systematically outperforming what their speed alone would predict — and "
      "why? The answer turns out to involve first-step quickness, lead distance, catcher arm "
      "strength, and how well a runner reads specific pitcher deliveries, not top-end speed."),

     ("Where the Data Comes From", False,
      "Everything in this model starts with publicly available Baseball Savant data. "
      "Pitch-by-pitch Statcast records (2018–2026, roughly 150,000 to 250,000 pitches per season "
      "with a runner on first base) provide the raw event log. From these records the model "
      "extracts every stolen base attempt and caught stealing on second base. Sprint speed, jump "
      "time, and 5-foot running splits come from Savant's sprint and acceleration leaderboards. "
      "Real catcher pop times (2018+) come from Savant's catcher pop-time leaderboard. Pitcher "
      "running-game suppression numbers (pickoff rate, step-off rate, hold time) come from "
      "Savant's running-game leaderboard. Lead distance is a career-level average per runner, "
      "fetched from Savant's base-stealing run value table — note that Savant's year filter on "
      "this endpoint is silently ignored; every season query returns the same career snapshot, so "
      "lead data is treated as a career constant rather than a per-season number."),
 ]
},

{"discussion": "The SB Residual: The Core Signal",
 "paragraphs": [
     ("Why Not Just Use Success Rate?", False,
      "A runner with a 90% success rate is not automatically better than one with 80%. If the "
      "first runner only steals against catchers with slow pop times and pitchers who ignore "
      "runners, the 90% might be expected given context. The 80% runner might be attempting far "
      "tougher steals. The model's primary signal — called the SB Residual — corrects for this "
      "by asking: given this runner's sprint speed, what success rate would we predict for an "
      "average runner at that speed, and how much does this specific runner beat or miss that "
      "expectation? The residual is simply (real SB%) minus (speed-expected SB%). Positive "
      "values mean the runner outperforms their speed peers. Naylor's residual is large and "
      "positive. Trea Turner's is smaller than you might expect because his success rate is "
      "exactly in line with what a 30 ft/s runner should be able to do."),

     ("Bayesian Shrinkage: Why Small Sample Rates Are Distrusted", False,
      "A runner who goes 1-for-1 has a 100% success rate, but that number is meaningless. "
      "The model applies Bayesian shrinkage with a constant of k=5: every runner's success rate "
      "is computed as (SB + 5 × league_average) / (SB + CS + 5). This pulls small-sample rates "
      "toward the league mean and only lets extreme rates stand when the sample is large enough. "
      "A runner with 30 attempts and 28 successes (93.3%) stays near 93.3% because the sample "
      "is real. A runner with 1 attempt and 1 success (100%) gets pulled back toward 78%."),
 ]
},

{"discussion": "The Three Models — A, B, and the GLM",
 "paragraphs": [
     ("Why Are There Multiple Models?", False,
      "The pipeline produces three separate models for different purposes. They are not "
      "three versions of the same thing; each answers a different question, and each has "
      "a different unit of observation, different features, and different intended use."),

     ("Model A: Per-Attempt GBM (the strict one)", False,
      "Model A is a gradient-boosting machine trained on individual pitch-level steal attempts "
      "(roughly 30,000–60,000 rows across 2018–2026). Each row is one specific attempt: this "
      "runner, this pitcher, this catcher, this count, this inning. The model tries to predict "
      "whether that single attempt succeeded. To prevent data leakage — where the same runner's "
      "career-aggregate lead numbers appear in both training and test rows — the model uses "
      "'group cross-validation by runner ID,' meaning a runner's attempts are never split across "
      "training and testing. This is the strictest, most honest design. Its AUC is around "
      "0.57–0.59, which looks low. But consider: the baseline success rate in the data is "
      "already about 54–58% (managers only send runners on attempts they believe have a good "
      "chance), which means a coin flip already gets you 54%. The per-attempt model at 0.59 "
      "is barely above that. Individual SB attempts are highly noisy — a bad hop, a perfect "
      "slide, a pitcher glancing at the wrong moment — and these are not captured in any data "
      "source we have."),

     ("Model B: Season-Level GBM (the better performer)", False,
      "Model B aggregates all of a runner's per-attempt features up to the season level and "
      "then asks: did this runner have a ABOVE-average success rate this season? One row per "
      "runner-season. A runner who attempted 25 steals and succeeded 22 times (88%) is a 'yes'; "
      "one who attempted 20 and succeeded 13 (65%) is a 'no.' By smoothing over a full season "
      "of attempts, the noise largely cancels out, and the real signal — is this person actually "
      "good at stealing — becomes clearer. Model B's AUC is 0.662–0.700 depending on the era. "
      "This is the model whose performance we report as the headline number."),

     ("The GLM: Plain-English Weights", False,
      "The GLM (Simple Logistic Regression) is not designed to be the best predictor. It is "
      "designed to be the most interpretable. Unlike the GBMs (which learn thousands of "
      "nonlinear interactions between variables), the GLM has one coefficient per feature, and "
      "those coefficients translate directly into the 'SB% Boost per Tier' and 'Odds Multiplier' "
      "columns you see in the model output. If you want to know whether catcher pop time matters "
      "more than jump time, read the GLM weights — the GBM will give a better prediction but "
      "won't tell you which feature drove the prediction."),
 ]
},

{"discussion": "The Baseline P(Success): What Does 62% Mean?",
 "paragraphs": [
     ("The Intercept as a Starting Point", False,
      "Every logistic model has an intercept — the predicted probability when all features are "
      "exactly at their league-average values. In the Naylor Model this is called the Baseline "
      "P(success), and it is approximately 60–62%. That number means: an imaginary runner with "
      "exactly average sprint speed, exactly average jump time, exactly average lead distance, "
      "facing exactly average catcher pop time and average pitcher suppression, would be expected "
      "to succeed on about 62% of steal attempts. This is higher than the naive 50/50 because "
      "managers already filter out the bad attempts — they tend to send runners only when the "
      "matchup looks favorable."),

     ("How the Boost Numbers Work", False,
      "The 'SB% Boost per Tier' column tells you how much the model's prediction shifts when "
      "you improve a single feature by one standard deviation (roughly one tier on the chart). "
      "Start at the baseline 62%. If you improve Jump Time from 'average' to 'above average' "
      "(one tier better = lower seconds), the model raises its prediction by roughly +10 "
      "percentage points to about 72%. If the runner also has above-average lead distance (+4 "
      "pp), the model is now at about 76%. These boosts are approximate and additive near the "
      "baseline; they compress at extreme probabilities because success rate can't exceed 100%. "
      "Negative boost values mean the feature correlates with LOWER success when it improves — "
      "this almost never happens except when a feature is in seconds (where lower = better, so "
      "a 'tier 1 increase' actually means slower, which hurts)."),
 ]
},

{"discussion": "AUC: Why 0.66–0.70 Is Actually Good",
 "paragraphs": [
     ("What AUC Measures", False,
      "AUC (Area Under the ROC Curve) measures how well a binary classifier separates the "
      "'yes' and 'no' outcomes. An AUC of 0.50 means the model is no better than a coin flip. "
      "An AUC of 1.00 means perfect prediction. The Naylor Model v6 achieves AUC of about "
      "0.66 on the full 2018–2026 dataset, 0.70 on the pre-2023 era, and 0.68 on the "
      "post-2023 era."),

     ("Why the Ceiling Is Real, Not a Bug", False,
      "Stolen-base success has a large irreducible noise component. The throw, the catcher's "
      "grip on a given day, whether the pitcher noticed the runner leaning, the umpire's safe/out "
      "call on a bang-bang play — none of these appear in any public data feed. Academic "
      "baseball research generally finds prediction ceilings around 0.68–0.72 for SB success "
      "even with the best features. The Naylor Model is operating at that ceiling. Pushing higher "
      "would require private TrackMan or Hawkeye delivery-timing data that teams have internally "
      "but do not publish. The pre-2023 era's higher AUC (0.70) is because the signal is cleaner: "
      "the 2023 rule changes (larger bases, pickoff limits) structurally shifted the success rate "
      "league-wide, and the model needed to adapt to a new regime."),
 ]
},

{"discussion": "The SSSI: Designing for the Naylor Archetype",
 "paragraphs": [
     ("What Is the SSSI?", False,
      "The Slow-Steal Skill Index (SSSI) is not a model prediction — it is a composite "
      "scouting score. It weights eight features in a way specifically designed to find "
      "runners who outperform their speed: high SB residual, high Accel Gap (first-step "
      "quickness relative to top speed), good lead distance, elite catcher arm exploitation, "
      "and strong pre-release velocity. Each feature is z-scored (converted to standard "
      "deviations from the mean) so they are all on the same scale before weighting."),

     ("How Weights Were Chosen Without Overfitting", False,
      "The SSSI weights were chosen by a grid search — testing thousands of weight "
      "combinations and picking the one that best separates elite stealers from average "
      "runners in historical data. The obvious danger is that we might find weights that "
      "work perfectly for Naylor and Soto just because we tested them. To prevent this, "
      "Naylor and Soto were excluded from the weight search entirely. Their data was held "
      "out. The 80% of remaining runners were used for optimization; the held-out 20% "
      "(including Naylor and Soto's seasons) were used only to verify the final result. "
      "Naylor 2025 ranking #1 and Soto 2025 ranking #5 in the final SSSI is therefore "
      "a genuine out-of-sample result, not a product of having been used in optimization."),

     ("SSSI vs Model B: Different Tools", False,
      "SSSI ranks every runner on a single composite number. Model B predicts whether a "
      "runner will have above-average success this season. They overlap but are not the "
      "same. Model B will give Freddie Freeman a high score in 2024 because his season "
      "stats show real above-average stealing. SSSI will also reward him but additionally "
      "weight the fact that he achieved it with well-below-average sprint speed — a "
      "harder feat. The SSSI is the 'how impressive is this' score; Model B is the "
      "'will this runner steal successfully this year' prediction."),
 ]
},

{"discussion": "Known Limitations and Data Quirks",
 "paragraphs": [
     ("Lead Data Is Not Per-Season", False,
      "The Savant base-stealing run value table is supposed to support year-by-year queries, "
      "but the year filter is silently ignored by the API — every query returns the same "
      "career snapshot regardless of what year you request. This means 'lead distance' and "
      "'lead gain' in this model are career averages, not the specific numbers from a given "
      "season. For most runners this barely matters (their approach is consistent), but for "
      "a runner who dramatically changed their lead strategy mid-career, the per-season "
      "comparison is slightly misleading."),

     ("The 3-2 Count Artifact", False,
      "Statcast's pitch-level data records the balls-strikes count at each pitch. Stolen "
      "base attempts, however, are not tagged as individual pitch rows — they are embedded "
      "in the 'description text' (des field) of the plate appearance's final pitch. As a "
      "result, every parsed SB attempt inherits the count from the LAST pitch of the "
      "at-bat, not the actual pitch the runner broke on. The 3-2 count is the most common "
      "last-pitch count, so 3-2 shows up as having a 1.9% attempt rate while every other "
      "count shows near-zero. This is a data artifact, not a real strategic pattern. "
      "The '3-2 Count Attempt Share' variable was retained in v6 as a rough proxy for "
      "late-count aggressiveness but should not be interpreted as a precise tactical measure."),

     ("Pitcher Delivery Time Unavailable", False,
      "Per-pitcher delivery time from first movement to release is the single most valuable "
      "missing variable. Longer deliveries give runners more time to build speed before the "
      "catcher receives the ball. This data exists inside MLB's proprietary TrackMan system "
      "but is not published. Savant's 'pitch tempo' CSV was tested but found to have a bug: "
      "the median_seconds_onbase column equals median_seconds_empty for every pitcher, "
      "suggesting the tempo values do not actually condition on base state. The model uses "
      "a league-constant delivery proxy (1.30 seconds) instead. Any future version that "
      "gains access to real per-pitcher delivery times would likely push AUC significantly "
      "above the current 0.70 ceiling."),
 ]
},

]


# ─────────────────────────────────────────────────────────────────────────────
# Render to PDF
# ─────────────────────────────────────────────────────────────────────────────
TIER_COLOR = ["#10B981", "#3B82F6", "#6B7280", "#F59E0B", "#DC2626"]

def render(entries, path):
    PAGE_W, PAGE_H = 8.5, 11
    LEFT, RIGHT, TOP, BOTTOM = 0.55, 0.55, 0.55, 0.55

    with PdfPages(path) as pdf:

        # ─── Cover page ────────────────────────────────────────────────
        fig = plt.figure(figsize=(PAGE_W, PAGE_H)); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.text(0.5, 0.82, "The Naylor Model",
                ha="center", va="center", fontsize=30, fontweight="bold")
        ax.text(0.5, 0.74, "Variable Glossary",
                ha="center", va="center", fontsize=22, color="#1F3A5F")
        ax.text(0.5, 0.66, "Statcast-style reference  ·  v6",
                ha="center", va="center", fontsize=13, style="italic",
                color="#555")

        ax.text(0.5, 0.50,
                "Every variable in this glossary tells you:\n\n"
                "• what it actually measures (plain English)\n"
                "• the units and a typical value\n"
                "• what's elite and what's poor\n"
                "• a real player example\n"
                "• why it affects steal success",
                ha="center", va="center", fontsize=11.5, color="#222",
                linespacing=1.9)

        ax.text(0.5, 0.22,
                "No statistical jargon.\n"
                "Replace 'coef_z' and 'OR/SD' with\n"
                "'SB % Boost per Tier' and 'Odds Multiplier'.",
                ha="center", va="center", fontsize=11, color="#444",
                linespacing=1.6)

        ax.text(0.5, 0.07,
                "Generated for The Naylor Model · 2026",
                ha="center", va="center", fontsize=9, color="#888")
        pdf.savefig(fig); plt.close(fig)

        # ─── Variable pages ────────────────────────────────────────────
        for entry in entries:
            if "section" in entry:
                _render_section_page(pdf, entry["section"])
            elif "discussion" in entry:
                _render_discussion_page(pdf, entry)
            else:
                _render_variable_page(pdf, entry)

    print(f"Wrote {path}")


def _render_section_page(pdf, title):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    # Big section header
    ax.add_patch(plt.Rectangle((0.06, 0.45), 0.88, 0.12,
                                transform=fig.transFigure,
                                facecolor="#0B2545", edgecolor="none"))
    ax.text(0.5, 0.51, title,
            transform=fig.transFigure,
            ha="center", va="center",
            fontsize=24, fontweight="bold", color="white")
    pdf.savefig(fig); plt.close(fig)


def _render_variable_page(pdf, e):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    # === Header bar ===
    ax.add_patch(plt.Rectangle((0.0, 0.91), 1.0, 0.07,
                                transform=fig.transFigure,
                                facecolor="#0B2545", edgecolor="none"))
    ax.text(0.06, 0.945, e["name"],
            transform=fig.transFigure,
            fontsize=17, fontweight="bold", color="white", va="center")

    y = 0.86

    # === Plain English ===
    ax.text(0.06, y, "What it is",
            transform=fig.transFigure,
            fontsize=11, fontweight="bold", color="#1F3A5F")
    y -= 0.025
    y = _wrap_text(ax, e["plain"], LEFT_X=0.06, RIGHT_X=0.94, y=y,
                    fontsize=11, color="#222", fig=fig)
    y -= 0.025

    # === Quick-facts panel (units / avg / elite) ===
    facts_y = y - 0.005
    facts = [("Units",        e.get("units",   "—")),
             ("League avg",   e.get("avg",     "—")),
             ("Elite",        e.get("elite",   "—"))]
    box_w = 0.88 / 3
    for i, (label, val) in enumerate(facts):
        x0 = 0.06 + i * box_w
        ax.add_patch(plt.Rectangle((x0+0.005, facts_y - 0.06),
                                    box_w-0.01, 0.058,
                                    transform=fig.transFigure,
                                    facecolor="#F3F4F6", edgecolor="#D1D5DB"))
        ax.text(x0 + box_w/2, facts_y - 0.018, label,
                transform=fig.transFigure,
                ha="center", fontsize=8.5, color="#6B7280")
        ax.text(x0 + box_w/2, facts_y - 0.043, val,
                transform=fig.transFigure,
                ha="center", fontsize=12, fontweight="bold", color="#0B2545")
    y = facts_y - 0.085

    # === Example ===
    ax.text(0.06, y, "Real example",
            transform=fig.transFigure,
            fontsize=11, fontweight="bold", color="#1F3A5F")
    y -= 0.024
    y = _wrap_text(ax, e["example"], LEFT_X=0.06, RIGHT_X=0.94, y=y,
                    fontsize=10.5, color="#333", fig=fig,
                    family="serif", italic=True)
    y -= 0.025

    # === Tier chart ===
    ax.text(0.06, y, "Tier chart",
            transform=fig.transFigure,
            fontsize=11, fontweight="bold", color="#1F3A5F")
    y -= 0.030
    for i, (label, rng) in enumerate(e.get("tiers", [])):
        color = TIER_COLOR[min(i, len(TIER_COLOR)-1)]
        ax.add_patch(plt.Rectangle((0.06, y - 0.020), 0.018, 0.018,
                                    transform=fig.transFigure,
                                    facecolor=color, edgecolor="none"))
        ax.text(0.085, y - 0.011, label,
                transform=fig.transFigure,
                fontsize=10, color="#222", fontweight="bold", va="center")
        ax.text(0.31, y - 0.011, rng,
                transform=fig.transFigure,
                fontsize=10, color="#444", family="serif", va="center")
        y -= 0.026
    y -= 0.010

    # === Impact ===
    ax.text(0.06, y, "Why it matters",
            transform=fig.transFigure,
            fontsize=11, fontweight="bold", color="#1F3A5F")
    y -= 0.024
    y = _wrap_text(ax, e["impact"], LEFT_X=0.06, RIGHT_X=0.94, y=y,
                    fontsize=10, color="#333", fig=fig)
    y -= 0.012

    # === Source ===
    if y > 0.05:
        ax.text(0.06, y, "Source",
                transform=fig.transFigure,
                fontsize=10, fontweight="bold", color="#1F3A5F")
        y -= 0.020
        ax.text(0.06, y, e.get("source", "—"),
                transform=fig.transFigure,
                fontsize=9.5, color="#555", family="monospace")

    pdf.savefig(fig); plt.close(fig)


def _render_discussion_page(pdf, entry):
    """Render a multi-paragraph discussion section.

    entry keys:
        discussion  — page title string
        paragraphs  — list of (heading, italic_bool, body_text)
    """
    LEFT_X, RIGHT_X = 0.06, 0.94

    # We may need multiple pages if the content is long.
    # Collect all paragraphs then flush pages as needed.
    paragraphs = entry.get("paragraphs", [])

    def new_page():
        fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        # Header bar
        ax.add_patch(plt.Rectangle((0.0, 0.91), 1.0, 0.07,
                                    transform=fig.transFigure,
                                    facecolor="#0B2545", edgecolor="none"))
        ax.text(0.06, 0.945, entry["discussion"],
                transform=fig.transFigure,
                fontsize=15, fontweight="bold", color="white", va="center")
        return fig, ax, 0.86

    fig, ax, y = new_page()

    for (heading, italic, body) in paragraphs:
        # heading
        if y < 0.14:
            pdf.savefig(fig); plt.close(fig)
            fig, ax, y = new_page()

        ax.text(LEFT_X, y, heading,
                transform=fig.transFigure,
                fontsize=11, fontweight="bold", color="#1F3A5F")
        y -= 0.026

        # body — word-wrapped
        width_chars = int((RIGHT_X - LEFT_X) * 96)
        for raw_line in body.split("\n"):
            words = raw_line.split(" ")
            line = ""
            for w in words:
                test = (line + " " + w).strip()
                if len(test) > width_chars:
                    if y < 0.10:
                        pdf.savefig(fig); plt.close(fig)
                        fig, ax, y = new_page()
                    ax.text(LEFT_X, y, line,
                            transform=fig.transFigure,
                            fontsize=10.5, color="#222",
                            style="italic" if italic else "normal")
                    y -= 0.019
                    line = w
                else:
                    line = test
            if line:
                if y < 0.10:
                    pdf.savefig(fig); plt.close(fig)
                    fig, ax, y = new_page()
                ax.text(LEFT_X, y, line,
                        transform=fig.transFigure,
                        fontsize=10.5, color="#222",
                        style="italic" if italic else "normal")
                y -= 0.019

        y -= 0.022  # gap after paragraph

    pdf.savefig(fig); plt.close(fig)


def _wrap_text(ax, text, LEFT_X, RIGHT_X, y, fontsize, color, fig,
                family="sans-serif", italic=False):
    """Naive word-wrap rendered with figure-fraction coordinates."""
    width_chars = int((RIGHT_X - LEFT_X) * 92)
    style = "italic" if italic else "normal"
    for raw in text.split("\n"):
        words = raw.split(" ")
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if len(test) > width_chars:
                ax.text(LEFT_X, y, line,
                        transform=fig.transFigure,
                        fontsize=fontsize, color=color, family=family,
                        style=style)
                y -= 0.018
                line = w
            else:
                line = test
        if line:
            ax.text(LEFT_X, y, line,
                    transform=fig.transFigure,
                    fontsize=fontsize, color=color, family=family,
                    style=style)
            y -= 0.018
    return y


if __name__ == "__main__":
    render(ENTRIES, OUT_PATH)
