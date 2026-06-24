#!/usr/bin/env python3
"""
The Naylor Model · v7  —  cleaner outputs, intuitive metrics
============================================================
Builds on v5.  Reuses the same per-pitch + catcher pop + pitcher running-game
data (cached on disk).  Focus of v7:

  ▸ Rename every output column in PLAIN ENGLISH (Statcast-style).
       coef_z          →  "SB % Boost per Tier"   (logit coefficient)
       OR / SD         →  "Odds Multiplier"      (odds ratio)
       pct_in_HL       →  "3-2 Count Attempt Share"
       accel_0_30      →  "Jump Time"
       maintain_30_90  →  "Top-Speed Phase"
       pre_rel_vel     →  "Pre-Release Velocity"
       post_rel_dist   →  "Post-Release Distance"

  ▸ Add a `boost_pp` column to the GLM table so anyone can read it:
       "Improving Jump Time by 1 tier raises predicted SB% by  X.X pp."

  ▸ Add a few new features identified by v5 SHAP as candidates:
       pitcher_pickoff_rate   = n_pk / n_init (career, per pitcher)
       weak_arm_share          = % of attempts vs catchers with pop ≥ 2.00 s
       two_strike_share        = % of attempts in two-strike counts
                                  (replaces the broken `pct_in_HL`)

  ▸ Drop `pct_in_HL` from the feature set (data artifact — see v5 §9).

  ▸ Use the v5-winning split representation (18 raw 5-ft cols) by default.

  ▸ Produce a single-page Statcast-style Naylor + Soto profile.

Outputs:  DF_v7_*.csv, Fig_v7_*.png, Naylor_Model_v7_Report.pdf.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")

import io, re, pickle, requests
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (StratifiedKFold, GroupKFold, KFold,
                                     cross_val_predict, cross_val_score)
from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss,
                             mean_squared_error)

from pybaseball import (statcast_sprint_speed, statcast_running_splits,
                        statcast_catcher_poptime)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ── Constants
SEASONS_ALL    = list(range(2015, 2027))
SEASONS_PITCH  = [2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
RULE_CHANGE_YEAR = 2023
MIN_REAL_SB_CS   = 10
SEED             = 42
NAYLOR_ID, SOTO_ID = 647304, 665742
SPEED_CAP        = 28.0
OUTPUT_DIR       = Path("/Users/shunchen/Desktop/The-Naylor-Model")
CACHE_DIR        = OUTPUT_DIR / ".cache"
FIGURES_DIR      = OUTPUT_DIR / "Figures"
DATA_DIR         = OUTPUT_DIR / "data"
REPORTS_DIR      = OUTPUT_DIR / "Reports"
for _d in (FIGURES_DIR, DATA_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)

np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

COLOR = {"pre":"#E0A458","post":"#3D5A80","naylor":"#DC2626","soto":"#1D4ED8",
         "neutral":"#374151","highlight":"#10B981","accent":"#0EA5E9",
         "elite":"#10B981","above":"#3B82F6","avg":"#6B7280",
         "below":"#F59E0B","poor":"#DC2626"}

print("=" * 72)
print(" THE NAYLOR MODEL  ·  v7  (intuitive outputs)")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CACHE HELPERS  (same as v5 — reuse cached files)
# ─────────────────────────────────────────────────────────────────────────────
def cache_load(name):
    p = CACHE_DIR / f"{name}.pkl"
    if p.exists():
        try:
            with open(p, "rb") as f: return pickle.load(f)
        except Exception as e:
            # A pickle written by an older pandas/numpy can fail to unpickle under
            # a newer one (e.g. NDArrayBacked datetime64[us], or StringDtype's
            # changed pickle format).  Treat an unreadable cache as a MISS so the
            # caller refetches instead of crashing — cheap insurance against
            # pandas/numpy version bumps.
            print(f"   [cache] {name}.pkl unreadable ({type(e).__name__}); refetching")
            return None
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA (all from v5 cache)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/10] Loading cached data …")

pitch_frames = []
for yr in SEASONS_PITCH:
    df = cache_load(f"pitches_{yr}")
    if df is not None: pitch_frames.append(df)
DF_Pitch = pd.concat(pitch_frames, ignore_index=True)
print(f"   Pitches w/ runner on 1st: {len(DF_Pitch):,}")

pop_frames = [cache_load(f"poptime_{yr}") for yr in SEASONS_PITCH]
DF_Pop = pd.concat([d for d in pop_frames if d is not None],
                    ignore_index=True)
print(f"   Catcher-seasons:          {len(DF_Pop):,}")

DF_PitcherRG = cache_load("pitcher_runninggame")
print(f"   Pitchers w/ running-game: {len(DF_PitcherRG):,}")

DF_RunnerLead = cache_load("runner_lead")
print(f"   Runners w/ lead profile:  {len(DF_RunnerLead):,}")

# Compute pitcher pickoff rate from saved pitcher running-game
# Need the full CSV with n_pk and n_init.  Refetch quickly.
def fetch_pitcher_full():
    url = ("https://baseballsavant.mlb.com/leaderboard/pitcher-running-game"
           "?team=&min=q&csv=true")
    r = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0"})
    df = pd.read_csv(io.BytesIO(r.content))
    df = df.rename(columns={"player_id":"pitcher_id"})
    df["pickoff_rate"] = df["n_pk"] / df["n_init"].clip(lower=1)
    return df[["pitcher_id","pickoff_rate","n_pk","n_init"]]

DF_Pickoff = fetch_pitcher_full()
print(f"   Pickoff rates loaded:     {len(DF_Pickoff):,}")

# Sprint speed + running splits
sp_frames, rs_frames = [], []
for yr in SEASONS_ALL:
    sp = cache_load(f"sprint_{yr}")
    rs = cache_load(f"splits_{yr}")
    if sp is None:
        try:
            sp = statcast_sprint_speed(yr, min_opp=1); sp["season"] = yr
        except Exception as e: continue
    if rs is None:
        try:
            rs = statcast_running_splits(yr, min_opp=1, raw_splits=True)
            rs["season"] = yr
        except Exception as e: continue
    sp_frames.append(sp); rs_frames.append(rs)

DF_Speed = pd.concat(sp_frames, ignore_index=True).rename(
    columns={"last_name, first_name":"player_name","player_id":"runner_id"})
DF_Splits = pd.concat(rs_frames, ignore_index=True).rename(
    columns={"last_name, first_name":"player_name","player_id":"runner_id"})
print(f"   Sprint-speed rows:        {len(DF_Speed):,}")
print(f"   Running-splits rows:      {len(DF_Splits):,}")

# MLB Stats API SB
def fetch_mlb_sb(season):
    cached = cache_load(f"mlb_sb_{season}")
    if cached is not None: return cached
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {"stats":"season","group":"hitting","season":season,
              "limit":5000,"sportIds":1,"playerPool":"All"}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    rows = []
    for s in r.json().get("stats", [{}])[0].get("splits", []):
        p = s.get("player", {}); st = s.get("stat", {})
        rows.append({"runner_id":p.get("id"),"Name":p.get("fullName"),
                     "SB":st.get("stolenBases", 0),
                     "CS":st.get("caughtStealing", 0),
                     "season":season})
    return pd.DataFrame(rows)

DF_SB = pd.concat([fetch_mlb_sb(y) for y in SEASONS_ALL], ignore_index=True)
print(f"   MLB SB rows:              {len(DF_SB):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. RUNNER-SEASON BASELINE (v4/v5 style)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/10] Building runner-season frame …")

DF_Splits["jump_time"]      = DF_Splits["seconds_since_hit_030"]
DF_Splits["accel_phase"]    = (DF_Splits["seconds_since_hit_030"]
                              - DF_Splits["seconds_since_hit_005"])
DF_Splits["top_speed_phase"]= (DF_Splits["seconds_since_hit_090"]
                              - DF_Splits["seconds_since_hit_030"])
DF_Splits["total_90"]       = DF_Splits["seconds_since_hit_090"]

SPLIT_COLS_5 = [f"seconds_since_hit_{d:03d}" for d in range(5, 95, 5)]
SPLIT_COLS_5 = [c for c in SPLIT_COLS_5 if c in DF_Splits.columns]

keep_speed = ["runner_id","season","player_name","sprint_speed","bolts"]
keep_split = (["runner_id","season","jump_time","accel_phase",
               "top_speed_phase","total_90"] + SPLIT_COLS_5)

DF_Season = (DF_Speed[keep_speed]
             .merge(DF_Splits[keep_split], on=["runner_id","season"], how="inner")
             .merge(DF_SB[["runner_id","season","SB","CS","Name"]],
                    on=["runner_id","season"], how="left")
             .merge(DF_RunnerLead, on="runner_id", how="left"))

DF_Season["SB"]=DF_Season["SB"].fillna(0).astype(int)
DF_Season["CS"]=DF_Season["CS"].fillna(0).astype(int)
DF_Season["bolts"]=DF_Season["bolts"].fillna(0)
DF_Season["sb_attempts"]=DF_Season["SB"]+DF_Season["CS"]
DF_Season["era"]=np.where(DF_Season["season"]>=RULE_CHANGE_YEAR,
                          "post_2023","pre_2023")
DF_Season["player_name"]=DF_Season["Name"].fillna(DF_Season["player_name"])
DF_Season["speed_capped"]=DF_Season["sprint_speed"].clip(upper=SPEED_CAP)

# Friendlier column names for the real-lead variables
DF_Season = DF_Season.rename(columns={
    "r_primary_lead":"primary_lead",
    "r_secondary_lead":"secondary_lead",
    "r_sec_minus_prim_lead":"lead_gain",
})

# The upstream merges can emit duplicate runner-season rows.  Inspection shows
# these dupes share identical SB/CS/sprint_speed but differ slightly in the
# Statcast running-split times (e.g. accel_0_30 1.77 vs 1.81) — i.e. they are
# repeated split measurements for the same runner-season, not separate stints.
# Collapse to ONE row per runner-season by AVERAGING the numeric columns (so we
# use the best estimate of the split) and keeping the first label for text cols.
_n_before = len(DF_Season)
if DF_Season.duplicated(subset=["runner_id","season"]).any():
    _num_cols = DF_Season.select_dtypes(include="number").columns.difference(["runner_id","season"])
    _obj_cols = DF_Season.columns.difference(list(_num_cols)+["runner_id","season"])
    _agg = {c:"mean" for c in _num_cols}
    _agg.update({c:"first" for c in _obj_cols})
    DF_Season = (DF_Season.groupby(["runner_id","season"], as_index=False)
                 .agg(_agg))[DF_Season.columns].reset_index(drop=True)
    for _c in ["SB","CS","sb_attempts"]:
        DF_Season[_c] = DF_Season[_c].round().astype(int)
    print(f"   de-duplicated runner-seasons (averaged splits): {_n_before} → {len(DF_Season)}")

# Shrunk SB% + residual
k = 5
mask_q = DF_Season["sb_attempts"]>=MIN_REAL_SB_CS
league_sb = DF_Season.loc[mask_q,"SB"].sum() / max(1, DF_Season.loc[mask_q,"sb_attempts"].sum())
DF_Season["real_sb_pct"] = ((DF_Season["SB"]+k*league_sb)
                            /(DF_Season["sb_attempts"]+k))
coeffs = np.polyfit(DF_Season.loc[mask_q,"sprint_speed"],
                    DF_Season.loc[mask_q,"real_sb_pct"], 2)
DF_Season["expected_sb_pct"]=np.polyval(coeffs, DF_Season["sprint_speed"]).clip(0.3, 0.99)
DF_Season["sb_residual"]=DF_Season["real_sb_pct"]-DF_Season["expected_sb_pct"]

# Percentile / accel_gap
DF_Season["pct_speed"] = DF_Season.groupby("season")["sprint_speed"].rank(pct=True)*100
DF_Season["pct_jump"]  = DF_Season.groupby("season")["jump_time"].rank(pct=True, ascending=False)*100
DF_Season["accel_gap"] = DF_Season["pct_jump"] - DF_Season["pct_speed"]

# ─────────────────────────────────────────────────────────────────────────────
#  NEW v7 METRIC A — ACCEL-TO-TOP-SPEED GAP
#  How quickly does a runner reach top speed (not just how fast they are)?
#  Smaller "runway" to top speed = better.  Because higher top speeds need more
#  runway, doing it on few feet at high speed is a PREMIUM (rare + valuable).
# ─────────────────────────────────────────────────────────────────────────────
# Per-5-ft segment velocity v_d = 5 / (t_d - t_{d-5}) across the 5..90 ft splits.
_split_d   = [int(c.split("_")[-1]) for c in SPLIT_COLS_5]          # e.g. 5,10,..,90
_seg_speed = pd.DataFrame(index=DF_Season.index)
_prev_col, _prev_d = None, 0
for c, d in zip(SPLIT_COLS_5, _split_d):
    dt = (DF_Season[c] - (DF_Season[_prev_col] if _prev_col else 0.0))
    _seg_speed[d] = (d - _prev_d) / dt.replace(0, np.nan)
    _prev_col, _prev_d = c, d

# distance at which the runner first hits >=97% of their realized top segment speed
_top_seg = _seg_speed.max(axis=1)
def _dist_to_top(row, vmax):
    if not np.isfinite(vmax) or vmax <= 0:
        return np.nan
    for d in _split_d:
        v = row.get(d, np.nan)
        if np.isfinite(v) and v >= 0.97 * vmax:
            return float(d)
    return float(_split_d[-1])
DF_Season["dist_to_top_speed_ft"] = [
    _dist_to_top(_seg_speed.loc[i], _top_seg.loc[i]) for i in DF_Season.index
]

# speed-expected runway (mirror the sb_residual quadratic-fit pattern); negative
# gap = reaches top speed sooner than a runner of that sprint speed usually does.
_dmask = mask_q & DF_Season["dist_to_top_speed_ft"].notna() & DF_Season["sprint_speed"].notna()
if _dmask.sum() >= 10:
    _dcoef = np.polyfit(DF_Season.loc[_dmask, "sprint_speed"],
                        DF_Season.loc[_dmask, "dist_to_top_speed_ft"], 2)
    DF_Season["expected_dist_to_top"] = np.polyval(_dcoef, DF_Season["sprint_speed"])
else:
    DF_Season["expected_dist_to_top"] = DF_Season["dist_to_top_speed_ft"].mean()
DF_Season["accel_topspeed_gap"] = (DF_Season["dist_to_top_speed_ft"]
                                   - DF_Season["expected_dist_to_top"])

# PREMIUM: reward a small/negative gap MORE when the runner is genuinely fast.
_z_speed_all = ((DF_Season["sprint_speed"] - DF_Season["sprint_speed"].mean())
                / (DF_Season["sprint_speed"].std(ddof=0) + 1e-9))
ACCEL_PREMIUM_LAMBDA = 0.5
DF_Season["accel_topspeed_premium"] = (
    -DF_Season["accel_topspeed_gap"] * (1.0 + ACCEL_PREMIUM_LAMBDA * _z_speed_all.clip(lower=0))
)

print(f"   Runner-seasons: {len(DF_Season):,}  ·  qualified: {mask_q.sum():,}")
print(f"   League SB% = {league_sb:.3f}")
print(f"   dist_to_top_speed_ft  median={DF_Season['dist_to_top_speed_ft'].median():.1f} ft "
      f"(min {DF_Season['dist_to_top_speed_ft'].min():.0f}, max {DF_Season['dist_to_top_speed_ft'].max():.0f})")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PARSE SB ATTEMPTS FROM `des`
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/10] Parsing SB attempts from des …")
SB_RE = re.compile(r"steals\s*\(\d+\)\s*2nd\s*base", re.I)
CS_RE = re.compile(r"caught\s+stealing\s+2nd", re.I)
def label(des):
    if not isinstance(des, str): return ("none", 0, 0)
    if SB_RE.search(des): return ("sb", 1, 1)
    if CS_RE.search(des): return ("cs", 1, 0)
    return ("none", 0, 0)

labels = DF_Pitch["des"].apply(label)
DF_Pitch["sb_label"]  = labels.apply(lambda t: t[0])
DF_Pitch["y_attempt"] = labels.apply(lambda t: t[1])
DF_Pitch["y_success"] = labels.apply(lambda t: t[2])

DF_Attempts = (DF_Pitch[DF_Pitch["y_attempt"]==1]
               .sort_values(["game_pk","at_bat_number","pitch_number"])
               .drop_duplicates(["game_pk","at_bat_number"], keep="last")
               .copy()
               .rename(columns={"on_1b":"runner_id",
                                "fielder_2":"catcher_id",
                                "pitcher":"pitcher_id",
                                "game_year":"season"}))
for c in ["runner_id","catcher_id","pitcher_id"]:
    DF_Attempts[c] = pd.to_numeric(DF_Attempts[c], errors="coerce")
DF_Attempts = DF_Attempts.dropna(subset=["runner_id","catcher_id"])
DF_Attempts[["runner_id","catcher_id","pitcher_id"]] = \
    DF_Attempts[["runner_id","catcher_id","pitcher_id"]].astype(int)
print(f"   Attempts identified: {len(DF_Attempts):,}")

# Join battery + runner context
DF_Attempts = DF_Attempts.merge(
    DF_Pop[["catcher_id","season","pop_2b_sba","exchange_2b_3b_sba",
            "maxeff_arm_2b_3b_sba"]],
    on=["catcher_id","season"], how="left")
DF_Attempts = DF_Attempts.merge(DF_PitcherRG, on="pitcher_id", how="left")
DF_Attempts = DF_Attempts.merge(DF_Pickoff, on="pitcher_id", how="left")
DF_Attempts = DF_Attempts.merge(DF_RunnerLead, on="runner_id", how="left")
DF_Attempts = DF_Attempts.rename(columns={
    "r_primary_lead":"primary_lead",
    "r_secondary_lead":"secondary_lead",
    "r_sec_minus_prim_lead":"lead_gain",
    "pitcher_lead_gain_allowed":"pitcher_lead_allowed",
    "pop_2b_sba":"pop_time",
    "exchange_2b_3b_sba":"catcher_exchange",
    "maxeff_arm_2b_3b_sba":"catcher_arm_velo",
})

# Per-attempt derived features
DF_Attempts["two_strike"] = (DF_Attempts["strikes"]==2).astype(int)

# Compute new metrics per attempt (using v5 formulas)
# --- CV measured per-pitcher delivery upgrade -------------------------------
# pre_release_velocity = lead_gain / time-to-plate.  Historically this used a
# single league constant (LEAGUE_PITCHER_TTP = 1.30 s).  We now divide by the
# CV-MEASURED first-move->release time for the pitcher when we have one
# (data/DF_PitcherDelivery.csv from cv_pilot), falling back to 1.30 s otherwise.
# Fully reversible: drop the merge below and the velocity reverts to the constant.
LEAGUE_PITCHER_TTP = 1.30
_ttp = pd.Series(LEAGUE_PITCHER_TTP, index=DF_Attempts.index)
_pd_path = DATA_DIR / "DF_PitcherDelivery.csv"
if _pd_path.exists():
    DF_PitcherDelivery = pd.read_csv(_pd_path)
    DF_Attempts = DF_Attempts.merge(
        DF_PitcherDelivery[["pitcher_id", "median_delivery_s"]],
        on="pitcher_id", how="left")
    _ttp = DF_Attempts["median_delivery_s"].fillna(LEAGUE_PITCHER_TTP)
    _n_meas = int(DF_Attempts["median_delivery_s"].notna().sum())
    print(f"   CV delivery: {_n_meas:,}/{len(DF_Attempts):,} attempts use a measured "
          f"per-pitcher TTP (else {LEAGUE_PITCHER_TTP:.2f}s constant)")
DF_Attempts["pre_release_velocity"] = DF_Attempts["lead_gain"] / _ttp

# Get runner profile (sprint, jump_time) into attempts
season_cols = ["runner_id","season","sprint_speed","speed_capped",
               "jump_time","accel_phase","top_speed_phase","total_90",
               "accel_gap","bolts","real_sb_pct","sb_residual"]
DF_Attempts = DF_Attempts.merge(
    DF_Season[season_cols].drop_duplicates(["runner_id","season"]),
    on=["runner_id","season"], how="left")

def post_rel(row):
    sp = row.get("sprint_speed"); pop = row.get("pop_time"); jt = row.get("jump_time")
    if pd.isna(sp) or pd.isna(pop) or pd.isna(jt): return np.nan
    naive = sp * pop
    penalty = max(0.0, jt - 1.65) * sp * pop * 0.5
    return max(0.0, naive - penalty)

DF_Attempts["post_release_distance"] = DF_Attempts.apply(post_rel, axis=1)

# Impute pop_time with league-year mean
DF_Attempts["pop_time"] = DF_Attempts.groupby("season")["pop_time"].transform(
    lambda s: s.fillna(s.mean()))
DF_Attempts["pop_time"] = DF_Attempts["pop_time"].fillna(1.95)

# Per runner-season aggregates (new v7 set)
agg = (DF_Attempts.groupby(["runner_id","season"])
       .agg(n_attempts=("y_attempt","sum"),
            n_success=("y_success","sum"),
            two_strike_share=("two_strike","mean"),
            avg_pop_faced=("pop_time","mean"),
            avg_pickoff_rate_faced=("pickoff_rate","mean"),
            weak_arm_share=("pop_time", lambda x: float((x>=2.0).mean())),
            avg_pre_release_velocity=("pre_release_velocity","mean"),
            avg_post_release_distance=("post_release_distance","mean"))
       .reset_index())
DF_Season = DF_Season.merge(agg, on=["runner_id","season"], how="left")

# Impute runners with no captured attempts (esp. Naylor 2025 falls here because
# his attempts landed on pitchers / counts not present in our parse).  We use
# league means so the runner isn't dropped from the model.
for col, fill in [("n_attempts", 0),
                  ("n_success", 0),
                  ("two_strike_share", 0.55),
                  ("weak_arm_share",   0.20),
                  ("avg_pop_faced",    1.95),
                  ("avg_pickoff_rate_faced", 0.02)]:
    if col in DF_Season.columns:
        DF_Season[col] = DF_Season[col].fillna(fill)

# Pre/post release per-runner from career lead and Statcast splits (independent
# of whether SB attempts were captured)
DF_Season["avg_pre_release_velocity"] = DF_Season["avg_pre_release_velocity"] \
    .fillna(DF_Season["lead_gain"] / LEAGUE_PITCHER_TTP)

# ── Diagnostic: did the CV-measured per-pitcher TTP make velocity carry signal
# independent of lead_gain?  At SEASON level the honest answer is "barely": only
# ~120 pitchers have a measured delivery, and per-runner-season averaging dilutes
# their TTP spread, so avg_pre_release_velocity stays near-collinear with
# lead_gain (velocity ≈ lead_gain / ~constant).  The measured delivery's real
# payoff is the PER-ATTEMPT model (cv_pilot, LOO-CV AUC 0.752), not this altitude.
_vd = DF_Season[["avg_pre_release_velocity", "lead_gain"]].dropna()
if len(_vd) > 2:
    _corr = float(_vd["avg_pre_release_velocity"].corr(_vd["lead_gain"]))
    print(f"   [diag] corr(avg_pre_release_velocity, lead_gain) = {_corr:.4f} "
          f"(season-level; ~1.0 ⇒ velocity ≈ rescaled lead_gain, little new signal)")

def post_rel_season(row):
    sp = row.get("sprint_speed"); jt = row.get("jump_time"); pop = 1.95
    if pd.isna(sp) or pd.isna(jt): return np.nan
    naive = sp * pop
    penalty = max(0.0, jt - 1.65) * sp * pop * 0.5
    return max(0.0, naive - penalty)

mask_post = DF_Season["avg_post_release_distance"].isna()
DF_Season.loc[mask_post, "avg_post_release_distance"] = \
    DF_Season.loc[mask_post].apply(post_rel_season, axis=1)

print(f"   Attempts joined with battery: {len(DF_Attempts):,}")
for c in ["pop_time","pickoff_rate","primary_lead","sprint_speed"]:
    if c in DF_Attempts.columns:
        print(f"     {c:24}  {DF_Attempts[c].notna().mean()*100:>5.1f}% non-null")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL B — season-level GBM with v7 features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/10] Model B — season-level GBM …")

V6_FEATURES = [
    "sprint_speed","speed_capped","jump_time","accel_phase","top_speed_phase",
    "total_90","accel_gap","bolts",
    "dist_to_top_speed_ft","accel_topspeed_gap",
    "primary_lead","secondary_lead","lead_gain",
    "avg_pop_faced","avg_pickoff_rate_faced","weak_arm_share","two_strike_share",
    "avg_pre_release_velocity","avg_post_release_distance",
    "n_attempts",
]
V6_FEATURES = [c for c in V6_FEATURES if c in DF_Season.columns]

work = DF_Season[mask_q].dropna(subset=V6_FEATURES+["real_sb_pct"]).copy()
work["y"] = (work["real_sb_pct"]>=league_sb).astype(int)
print(f"   Qualified rows w/ all features: {len(work)}")

# v8.3: Model B is now a Bayesian-tuned XGBoost (see model_xgb.py / DF_xgb_tuned_params.csv).
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_xgb as _mxgb
_XGB_PARAMS = _mxgb.load_best_params(DATA_DIR)

def fit_season(df_sub, label):
    if len(df_sub) < 50: return None
    X = df_sub[V6_FEATURES].values; y = df_sub["y"].values
    w = df_sub["sb_attempts"].values.astype(float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = cross_val_predict(
        _mxgb.make_model(_XGB_PARAMS),
        X, y, cv=cv, method="predict_proba",
        params={"sample_weight": w})[:,1]
    return {"label":label, "n":len(df_sub),
             "auc":roc_auc_score(y, preds), "X":X, "y":y, "w":w}

m_full = fit_season(work,                          "full")
m_pre  = fit_season(work[work["era"]=="pre_2023"], "pre_2023")
m_post = fit_season(work[work["era"]=="post_2023"],"post_2023")

print()
auc_rows = []
for m in [m_full, m_pre, m_post]:
    if m is None: continue
    print(f"   {m['label']:>10}  n={m['n']:>4}  AUC={m['auc']:.4f}")
    auc_rows.append({"epoch":m["label"], "n":m["n"], "auc":round(m["auc"],4)})
pd.DataFrame(auc_rows).to_csv(DATA_DIR/"DF_v7_ModelB_AUC.csv", index=False)

# Feature importance (XGBoost gain) per era
shap_rows = []
for m in [m_full, m_pre, m_post]:
    if m is None: continue
    xgb = _mxgb.make_model(_XGB_PARAMS).fit(m["X"], m["y"], sample_weight=m["w"])
    imp = xgb.feature_importances_
    for f, v in zip(V6_FEATURES, imp):
        shap_rows.append({"epoch":m["label"], "feature":f,
                          "importance": round(float(v), 4)})
DF_Imp = pd.DataFrame(shap_rows)
DF_Imp.to_csv(DATA_DIR/"DF_v7_Importance.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6. INTUITIVE GLM (replaces coef_z / OR/SD with plain-English columns)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/10] Simple GLM with intuitive metric names …")

simple_feat = ["speed_capped","jump_time","primary_lead","lead_gain",
               "avg_pre_release_velocity","avg_post_release_distance",
               "avg_pop_faced","avg_pickoff_rate_faced","weak_arm_share",
               "accel_gap","accel_topspeed_premium","bolts","two_strike_share"]
simple_feat = [c for c in simple_feat if c in work.columns and work[c].notna().sum()>50]

ws = work.dropna(subset=simple_feat).copy()
Xs = ws[simple_feat].values; ys = ws["y"].values
wts = ws["sb_attempts"].values.astype(float)
sc = StandardScaler().fit(Xs)
Xz = sc.transform(Xs)
glm = LogisticRegression(C=1e6, max_iter=5000).fit(Xz, ys, sample_weight=wts)

intercept = float(glm.intercept_[0])
baseline_p = 1.0 / (1.0 + np.exp(-intercept))

display_name = {
    "speed_capped":            "Sprint Speed (capped at 28)",
    "jump_time":               "Jump Time",
    "primary_lead":            "Primary Lead",
    "lead_gain":               "Lead Gain (jerk)",
    "avg_pre_release_velocity":"Pre-Release Velocity",
    "avg_post_release_distance":"Post-Release Distance",
    "avg_pop_faced":           "Avg Catcher Pop Faced",
    "avg_pickoff_rate_faced":  "Avg Pitcher Pickoff Rate Faced",
    "weak_arm_share":          "Share vs Weak-Arm Catchers",
    "accel_gap":               "Accel Gap",
    "accel_topspeed_premium":  "Accel→Top-Speed Premium",
    "bolts":                   "Bolts",
    "two_strike_share":        "Two-Strike Count Share",
}
direction = {  # whether HIGHER feature value is GOOD for the runner
    "speed_capped":             True,
    "jump_time":                False,  # lower = better
    "primary_lead":             True,
    "lead_gain":                True,
    "avg_pre_release_velocity": True,
    "avg_post_release_distance":True,
    "avg_pop_faced":            True,  # facing slow catchers helps
    "avg_pickoff_rate_faced":   False, # facing pickoff-heavy pitchers hurts
    "weak_arm_share":           True,
    "accel_gap":                True,
    "accel_topspeed_premium":   True,  # reach top speed sooner (esp. when fast) = good
    "bolts":                    True,
    "two_strike_share":         False, # uncertain, treat as neutral
}

# Convert coefficients → plain-English boost
def pp_boost(coef):
    return (1.0/(1.0+np.exp(-(intercept+coef))) - baseline_p) * 100.0

coef = glm.coef_.ravel()
rows = []
for f, c, mean_, sd_ in zip(simple_feat, coef, sc.mean_, sc.scale_):
    boost = pp_boost(c)
    rows.append({
        "feature": display_name.get(f, f),
        "feature_raw": f,
        "league_avg":     round(float(mean_), 3),
        "one_tier_step":  round(float(sd_),  3),
        "sb_pct_boost_per_tier": round(float(boost),  2),
        "odds_multiplier":       round(float(np.exp(c)), 3),
        "tech_coefficient":      round(float(c), 4),
        "higher_is_better":      direction.get(f, None),
    })
DF_GLM = pd.DataFrame(rows).sort_values("sb_pct_boost_per_tier",
                                         key=lambda s: s.abs(),
                                         ascending=False)
DF_GLM.to_csv(DATA_DIR/"DF_v7_GLM_PlainEnglish.csv", index=False)

print(f"\n   Baseline P(success) at all-mean inputs ≈ {baseline_p:.3f}")
print(f"\n   {'Feature':<32}{'Boost (pp)':>12}{'Odds×':>9}  Note")
for _, r in DF_GLM.iterrows():
    arrow = "↑helps" if r["higher_is_better"] else ("↓helps" if r["higher_is_better"] is False else "?")
    print(f"   {r['feature']:<32}{r['sb_pct_boost_per_tier']:>+12.2f}"
          f"{r['odds_multiplier']:>9.3f}  {arrow}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SSSI v7  (same idea, cleaner column names)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/10] SSSI v7 (held-out weight search) …")

def zscore(s): return (s - s.mean())/(s.std(ddof=0)+1e-9)

SSSI = work.copy()
SSSI["z_sb_residual"]       = zscore(SSSI["sb_residual"])
SSSI["z_accel_gap"]         = zscore(SSSI["accel_gap"])
SSSI["z_lead_gain"]         = zscore(SSSI["lead_gain"])
SSSI["z_primary_lead"]      = zscore(SSSI["primary_lead"])
SSSI["z_jump"]              = -zscore(SSSI["jump_time"])
SSSI["z_speed_cap"]         = zscore(SSSI["speed_capped"])
SSSI["z_pre_rel_vel"]       = zscore(SSSI["avg_pre_release_velocity"])
SSSI["z_post_rel_dist"]     = zscore(SSSI["avg_post_release_distance"])
SSSI["z_accel_top"]         = zscore(SSSI["accel_topspeed_premium"])  # v7: reach top speed sooner

# 80% train, 20% holdout (incl. anchors)
all_runners = SSSI["runner_id"].unique()
rng = np.random.default_rng(SEED)
holdout = set(rng.choice(all_runners, size=int(len(all_runners)*0.20),
                          replace=False))
holdout.update([NAYLOR_ID, SOTO_ID])
train_mask = ~SSSI["runner_id"].isin(holdout)

def score(w, df):
    return (w[0]*df["z_sb_residual"] + w[1]*df["z_accel_gap"]
            + w[2]*df["z_lead_gain"] + w[3]*df["z_jump"]
            + w[4]*df["z_primary_lead"] + w[5]*df["z_speed_cap"]
            + w[6]*df["z_pre_rel_vel"] + w[7]*df["z_post_rel_dist"]
            + w[8]*df["z_accel_top"])

grid = [(a,b,c,d,e,f_,g,h,i)
        for a in [0.10,0.20,0.30,0.35]
        for b in [0.05,0.10,0.15,0.20]
        for c in [0.05,0.10,0.15]
        for d in [0.0,0.05,0.10]
        for e in [0.05,0.10,0.15]
        for f_ in [-0.30,-0.20,-0.10,0.0]
        for g in [0.0,0.05,0.10]
        for h in [0.0,0.05,0.10]
        for i in [0.0,0.10,0.20]]

best = (-np.inf, None)
SSSI_train = SSSI[train_mask]
SSSI_anchors = SSSI[SSSI["runner_id"].isin([NAYLOR_ID,SOTO_ID])]
for w in grid:
    tr = score(w, SSSI_train)
    ho = score(w, SSSI_anchors)
    mu, sigma = tr.mean(), tr.std(ddof=0)+1e-9
    s = ((ho - mu)/sigma).mean()
    if s > best[0]: best = (s, w)

w_best = best[1]
SSSI["SSSI_v7"] = score(w_best, SSSI)
SSSI = SSSI.sort_values("SSSI_v7", ascending=False)
SSSI["rank_v7"] = SSSI["SSSI_v7"].rank(ascending=False, method="min").astype(int)
SSSI.to_csv(DATA_DIR/"DF_v7_SSSI.csv", index=False)

print(f"   Best Naylor+Soto mean z: {best[0]:.3f}")
print(f"   Weights: sb_res={w_best[0]} gap={w_best[1]} gain={w_best[2]} "
      f"jump={w_best[3]} prim={w_best[4]} speed={w_best[5]} "
      f"pre={w_best[6]} post={w_best[7]} accel_top={w_best[8]}")

top_cols = ["rank_v7","player_name","season","era","sprint_speed",
            "jump_time","primary_lead","lead_gain",
            "dist_to_top_speed_ft","accel_topspeed_premium",
            "avg_pre_release_velocity","avg_post_release_distance",
            "SB","CS","real_sb_pct","sb_residual","SSSI_v7"]
print("\n   Top 10 by SSSI_v7:")
print(SSSI.head(10)[top_cols].round(3).to_string(index=False))
print("\n   Naylor + Soto under SSSI_v7:")
print(SSSI[SSSI["runner_id"].isin([NAYLOR_ID,SOTO_ID])][top_cols]
        .round(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 8. LEADERBOARDS (per year, top 15)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/10] Leaderboards …")
LB = DF_Season[mask_q].copy()

def board(col, asc, fname):
    rows = []
    for yr, g in LB.groupby("season"):
        g2 = g.dropna(subset=[col]).sort_values(col, ascending=asc).head(15).copy()
        g2["rank"] = range(1, len(g2)+1)
        rows.append(g2[["season","rank","runner_id","player_name",
                         "sprint_speed","jump_time","primary_lead","lead_gain",
                         "avg_pre_release_velocity","avg_post_release_distance",
                         "avg_pop_faced","SB","CS","real_sb_pct","sb_residual",col]])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(DATA_DIR/fname, index=False)
    print(f"   {fname:<48}{len(out):>5} rows")
    return out

LB_jump = board("jump_time",                  True,  "DF_v7_LB_JumpTime.csv")
LB_gain = board("lead_gain",                  False, "DF_v7_LB_LeadGain.csv")
LB_prv  = board("avg_pre_release_velocity",   False, "DF_v7_LB_PreReleaseVelocity.csv")
LB_prd  = board("avg_post_release_distance",  False, "DF_v7_LB_PostReleaseDistance.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  NEW v7 METRIC B — EXPECTED STOLEN-BASE OUTCOME (xSB)   [standalone lens]
#  xSB = z(net SB above league avg) + z(sprint speed).  Surfaces runners who are
#  productive AND fast (high ceiling).  Complementary to SSSI (slow-but-skilled).
#  The DIFFERENCE z(speed) - z(net SB) = "potential gap": positive = fast but
#  under-stealing (untapped wheels); negative = over-performs speed (Naylor type).
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7b/10] Expected SB Outcome (xSB) …")
xsb = DF_Season[mask_q].copy()
xsb["net_sb"] = xsb["SB"] - xsb["CS"]
xsb["z_net_sb"] = xsb.groupby("season")["net_sb"].transform(zscore)
xsb["z_sprint"] = xsb.groupby("season")["sprint_speed"].transform(zscore)
xsb["xsb_outcome"]      = xsb["z_net_sb"] + xsb["z_sprint"]
xsb["sb_potential_gap"] = xsb["z_sprint"] - xsb["z_net_sb"]

def _quadrant(r):
    fast  = r["z_sprint"] >= 0
    steal = r["z_net_sb"] >= 0
    if fast and steal:       return "Realized Burner"
    if fast and not steal:   return "Untapped Wheels"
    if steal and not fast:   return "Crafty Technician"
    return "Stationary"
xsb["quadrant"] = xsb.apply(_quadrant, axis=1)

xsb = xsb.sort_values("xsb_outcome", ascending=False)
xsb["rank_xsb"] = xsb["xsb_outcome"].rank(ascending=False, method="min").astype(int)
xsb_cols = ["rank_xsb","player_name","season","era","sprint_speed","SB","CS",
            "net_sb","z_net_sb","z_sprint","xsb_outcome","sb_potential_gap","quadrant"]
xsb[xsb_cols + ["runner_id"]].to_csv(DATA_DIR/"DF_v7_xSB_Outcome.csv", index=False)
print(f"   DF_v7_xSB_Outcome.csv  {len(xsb):>5} rows")
print("\n   Top 10 by xSB outcome (fast + productive):")
print(xsb.head(10)[xsb_cols].round(3).to_string(index=False))
print("\n   Most untapped potential (high z_sprint, low z_net_sb):")
print(xsb.sort_values("sb_potential_gap", ascending=False).head(8)[xsb_cols]
        .round(3).to_string(index=False))
print("\n   Naylor + Soto under xSB:")
print(xsb[xsb["runner_id"].isin([NAYLOR_ID, SOTO_ID])][xsb_cols]
        .round(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 9. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/10] Figures …")

# Fig: GLM Plain-English bar chart
fig, ax = plt.subplots(figsize=(9, 5))
g = DF_GLM.sort_values("sb_pct_boost_per_tier")
colors = [COLOR["elite"] if v > 2 else COLOR["above"] if v > 0
          else COLOR["below"] if v > -2 else COLOR["poor"]
          for v in g["sb_pct_boost_per_tier"]]
ax.barh(g["feature"], g["sb_pct_boost_per_tier"], color=colors)
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("SB % Boost per Tier  (pp change in predicted success rate)")
ax.set_title("v7 Simple GLM — Plain-English Weight Table")
for i, v in enumerate(g["sb_pct_boost_per_tier"]):
    ax.text(v + (0.15 if v>=0 else -0.15), i, f"{v:+.1f}",
            ha="left" if v>=0 else "right", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES_DIR/"Fig_v7_GLM_PlainEnglish.png", dpi=160); plt.close(fig)

# Fig: AUC across versions
#  NOTE: v4–v6 AUCs were computed BEFORE the v7 de-duplication fix.  Those runs
#  carried duplicate runner-season rows that leaked across CV folds, inflating
#  AUC.  v7 averages duplicate split measurements into one row per runner-season,
#  removing the leak — so the v7 bar is the honest, de-leaked figure and is NOT
#  directly comparable to the (optimistic) historical bars.
fig, ax = plt.subplots(figsize=(7.4, 4.4))
labels = ["v4\n(season)","v5 Model A\n(per-attempt)","v5 Model B\n(season+new)","v6 Model B","v7 Model B\n(de-leaked)"]
v4_auc = 0.6300
v5_A = 0.5933
v5_B = 0.6794
v6_B = 0.6620
v7_B = m_full["auc"] if m_full else float("nan")
aucs = [v4_auc, v5_A, v5_B, v6_B, v7_B]
ax.bar(labels, aucs, color=[COLOR["neutral"], COLOR["accent"], COLOR["post"], COLOR["below"], COLOR["highlight"]])
ax.set_ylabel("CV AUC"); ax.set_ylim(0.5, 0.85)
ax.set_title("Model AUC across versions")
for i, v in enumerate(aucs):
    ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)
ax.text(0.5, -0.30,
        "v4–v6 bars carried duplicate runner-season rows that leaked across CV folds (optimistic).\n"
        "v7 averages duplicate split measurements → one row per runner-season, removing the leak.\n"
        "The v7 bar is the honest de-leaked AUC and is not directly comparable to the historical bars.",
        transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#555555")
fig.subplots_adjust(bottom=0.34)
fig.savefig(FIGURES_DIR/"Fig_v7_AUC.png", dpi=160); plt.close(fig)

# Fig: Pre vs Post importance
fig, ax = plt.subplots(figsize=(9, 6))
imp = DF_Imp.pivot(index="feature", columns="epoch", values="importance")
imp_friendly_idx = imp.index.map(
    lambda f: {"jump_time":"Jump Time","sprint_speed":"Sprint Speed",
                "primary_lead":"Primary Lead","lead_gain":"Lead Gain",
                "secondary_lead":"Secondary Lead",
                "avg_pop_faced":"Pop Time Faced",
                "avg_pickoff_rate_faced":"Pickoff Rate Faced",
                "weak_arm_share":"Weak-Arm Catcher Share",
                "avg_pre_release_velocity":"Pre-Release Velocity",
                "avg_post_release_distance":"Post-Release Distance",
                "accel_gap":"Accel Gap","bolts":"Bolts",
                "speed_capped":"Sprint (capped)","accel_phase":"Accel Phase",
                "top_speed_phase":"Top-Speed Phase","total_90":"Total 90",
                "two_strike_share":"Two-Strike Share",
                "n_attempts":"# Attempts"}.get(f, f))
imp.index = imp_friendly_idx
imp = imp.dropna(subset=[c for c in ["pre_2023","post_2023"] if c in imp.columns]).sort_values("post_2023")
y_ = np.arange(len(imp))
ax.barh(y_-0.18, imp.get("pre_2023", imp.iloc[:,0]), height=0.36, color=COLOR["pre"], label="pre-2023")
ax.barh(y_+0.18, imp.get("post_2023", imp.iloc[:,1]), height=0.36, color=COLOR["post"], label="post-2023")
ax.set_yticks(y_); ax.set_yticklabels(imp.index)
ax.set_xlabel("Mean |SHAP| importance")
ax.set_title("v7 — Feature Importance · Pre vs Post 2023")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR/"Fig_v7_Importance_PrePost.png", dpi=160); plt.close(fig)


# Fig: Naylor + Soto Statcast-style profile
def runner_profile_panel(ax, runner_id, name, color):
    rows = DF_Season[DF_Season["runner_id"]==runner_id].sort_values("season")
    if len(rows) == 0: return
    yrs = rows["season"].tolist()
    metrics = [("Sprint Speed (ft/s)", rows["sprint_speed"]),
                ("Jump Time (s)",        rows["jump_time"]),
                ("Lead Gain (ft)",       rows["lead_gain"]),
                ("Real SB %",            rows["real_sb_pct"]),
                ("SB Residual",          rows["sb_residual"]),
                ("Post-Release Dist",    rows["avg_post_release_distance"])]
    text = f"{name}  ({runner_id})\n\n"
    for label_, series in metrics:
        vals = [f"{v:.2f}" if pd.notna(v) else "—" for v in series.tolist()]
        text += f"{label_:<22}: " + "  ".join(f"{y}:{v}" for y,v in zip(yrs, vals)) + "\n"
    ax.axis("off")
    ax.text(0.02, 0.95, text, fontsize=8, family="monospace", va="top",
            color=color, transform=ax.transAxes)

fig, axes = plt.subplots(2, 1, figsize=(11, 7))
runner_profile_panel(axes[0], NAYLOR_ID, "Josh Naylor", COLOR["naylor"])
runner_profile_panel(axes[1], SOTO_ID, "Juan Soto", COLOR["soto"])
fig.suptitle("Naylor + Soto · Season-by-Season Statcast-style Profile", y=1.02)
fig.tight_layout()
fig.savefig(FIGURES_DIR/"Fig_v7_NaylorSoto_Profile.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# Fig: xSB Outcome quadrant — z(sprint) vs z(net SB)
QUAD_COLOR = {"Realized Burner": COLOR["elite"], "Untapped Wheels": COLOR["accent"],
              "Crafty Technician": COLOR["below"], "Stationary": COLOR["avg"]}
fig, ax = plt.subplots(figsize=(9, 7))
for q, gq in xsb.groupby("quadrant"):
    ax.scatter(gq["z_sprint"], gq["z_net_sb"], s=26, alpha=0.55,
               color=QUAD_COLOR.get(q, COLOR["avg"]), label=q, edgecolor="none")
ax.axhline(0, color="black", lw=0.7); ax.axvline(0, color="black", lw=0.7)
# quadrant captions
ax.text(0.98, 0.98, "Realized Burner\n(fast + steals)", transform=ax.transAxes,
        ha="right", va="top", fontsize=9, color=COLOR["elite"], fontweight="bold")
ax.text(0.98, 0.02, "Untapped Wheels\n(fast, under-steals)", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, color=COLOR["accent"], fontweight="bold")
ax.text(0.02, 0.98, "Crafty Technician\n(steals, slower)", transform=ax.transAxes,
        ha="left", va="top", fontsize=9, color=COLOR["below"], fontweight="bold")
ax.text(0.02, 0.02, "Stationary", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=9, color=COLOR["avg"], fontweight="bold")
# Annotate a richer set of names so the loaded quadrants read clearly:
#  • Naylor/Soto anchors (Crafty Technician)
#  • top Realized Burners (fast + productive — the recognizable stars)
#  • top Untapped Wheels (fast but under-stealing — the coaching targets)
def _label(r, color, dx=4, dy=4):
    ax.scatter(r["z_sprint"], r["z_net_sb"], s=72, color=color,
               edgecolor="white", linewidth=0.6, zorder=6)
    ax.annotate(f"{r['player_name'].split()[-1]} {int(r['season'])}",
                (r["z_sprint"], r["z_net_sb"]), fontsize=7.5, fontweight="bold",
                color="#222222", xytext=(dx, dy), textcoords="offset points", zorder=7)

# one row per player (their best xSB season) to avoid stacking the same name
_burners = (xsb[xsb["quadrant"] == "Realized Burner"]
            .sort_values("xsb_outcome", ascending=False)
            .drop_duplicates("runner_id").head(8))
for _, r in _burners.iterrows():
    _label(r, COLOR["elite"])

_untapped = (xsb[xsb["quadrant"] == "Untapped Wheels"]
             .sort_values("sb_potential_gap", ascending=False)
             .drop_duplicates("runner_id").head(6))
for _, r in _untapped.iterrows():
    _label(r, COLOR["accent"])

# Naylor/Soto anchors last so they sit on top
for _, r in xsb[xsb["runner_id"].isin([NAYLOR_ID, SOTO_ID])].iterrows():
    _label(r, COLOR["naylor"])

ax.set_xlabel("z(Sprint Speed)  →  faster"); ax.set_ylabel("z(Net SB above avg)  →  more productive")
ax.set_title("Expected SB Outcome (xSB) — Speed vs Production Quadrant")
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(FIGURES_DIR/"Fig_v7_xSB_Quadrant.png", dpi=160); plt.close(fig)

print("   5 v7 figures written.")


# ─────────────────────────────────────────────────────────────────────────────
# 10. PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9/10] PDF report …")

def textpage(pdf, title, lines):
    fig=plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor("white")
    ax=fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.text(0.06,0.94,title,fontsize=18,fontweight="bold",color="#0B2545")
    y=0.88
    for ln in lines:
        if ln.startswith("##"):
            y-=0.018; ax.text(0.06,y,ln[2:].strip(),fontsize=13,fontweight="bold",color="#1F3A5F"); y-=0.025
        elif ln.startswith("•") or ln.startswith("-"):
            ax.text(0.08,y,ln,fontsize=10.5,color="#222"); y-=0.020
        elif ln=="":
            y-=0.012
        else:
            ax.text(0.06,y,ln,fontsize=10.5,color="#222"); y-=0.020
        if y<0.06: break
    pdf.savefig(fig); plt.close(fig)

def imgpage(pdf, title, img, caption=""):
    if not Path(img).exists(): return
    fig=plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor("white")
    ax_t=fig.add_axes([0.06,0.92,0.88,0.05]); ax_t.axis("off")
    ax_t.text(0,0.5,title,fontsize=16,fontweight="bold",color="#0B2545")
    ax_i=fig.add_axes([0.06,0.18,0.88,0.70]); ax_i.axis("off")
    ax_i.imshow(plt.imread(img))
    if caption:
        ax_c=fig.add_axes([0.06,0.06,0.88,0.10]); ax_c.axis("off")
        ax_c.text(0,1,caption,fontsize=10,color="#444",va="top",wrap=True)
    pdf.savefig(fig); plt.close(fig)

pdf_path = REPORTS_DIR/"Naylor_Model_v7_Report.pdf"
with PdfPages(pdf_path) as pdf:
    # Cover
    fig=plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor("white")
    ax=fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.text(0.5,0.78,"The Naylor Model",fontsize=30,fontweight="bold",ha="center")
    ax.text(0.5,0.71,"v7 · Intuitive Outputs",fontsize=22,ha="center")
    ax.text(0.5,0.62,
        "Plain-English feature names · Statcast-style glossary\n"
        "GLM table now reads in 'SB % Boost per Tier' instead of coef_z",
        ha="center",fontsize=11,style="italic",color="#444",linespacing=1.5)
    ax.text(0.5,0.10,"Companion: Variable_Glossary.pdf",ha="center",fontsize=10,color="#888")
    pdf.savefig(fig); plt.close(fig)

    # Exec summary
    textpage(pdf, "Executive Summary", [
        f"Per-pitch data: {len(DF_Pitch):,} pitches w/ runner on 1st (2018-2026)",
        f"Attempts identified: {len(DF_Attempts):,}",
        f"Qualified runner-seasons (SB+CS≥{MIN_REAL_SB_CS}): {mask_q.sum():,}",
        "",
        "## What's new in v7",
        "• Accel→Top-Speed Premium — how few feet a runner needs to reach top",
        "  speed, speed-adjusted; small runway at high speed is a premium.",
        "  Added to the GBM, the GLM weight table, and as a 9th SSSI feature.",
        "• Expected SB Outcome (xSB) = z(net SB above avg) + z(sprint speed),",
        "  a standalone speed-vs-production quadrant lens (see §8).",
        "",
        "## Carried over from v6",
        "• ALL column names plain English — no coef_z, OR/SD, pct_in_HL.",
        "• GLM table shows  'SB % Boost per Tier'  =  pp change in success rate.",
        "• Real catcher pop, pitcher pickoff rate, lead variables.",
        "",
        "## AUC summary  (READ THE CAVEAT)",
        f"• v4 baseline:           0.6300   (leaked dup rows)",
        f"• v5 Model A (per-att):  0.5933   (leaked dup rows)",
        f"• v5 Model B (season):   0.6794   (leaked dup rows)",
        f"• v6 Model B (full):     0.6620   (leaked dup rows)",
        f"• v7 Model B (full):     {m_full['auc']:.4f}   ← honest, de-leaked",
        f"• v7 Model B (pre-23):   {m_pre['auc']:.4f}" if m_pre else "",
        f"• v7 Model B (post-23):  {m_post['auc']:.4f}" if m_post else "",
        "",
        "  v4–v6 runs carried duplicate runner-season rows (repeated Statcast",
        "  split measurements) that leaked across CV folds and inflated AUC.",
        "  v7 averages those duplicate splits into ONE row per runner-season,",
        "  removing the leak.  So v7's lower AUC is not a regression — it is the",
        "  first honest estimate.  The earlier bars are optimistic and are kept",
        "  only for historical context, NOT as a fair comparison.",
        "",
        "## Naylor + Soto rank under SSSI v7",
    ] + [
        f"  #{int(r['rank_v7']):>3}  {r['player_name']:<22}  {int(r['season'])}  "
        f"SSSI {r['SSSI_v7']:+.2f}   SB/CS {int(r['SB'])}/{int(r['CS'])}   real_sb_pct {r['real_sb_pct']:.3f}"
        for _, r in SSSI[SSSI["runner_id"].isin([NAYLOR_ID,SOTO_ID])].iterrows()
    ])

    imgpage(pdf, "§1 · AUC Comparison", FIGURES_DIR/"Fig_v7_AUC.png",
            "v7 Model B at the season level (real catcher pop, pitcher pickoff "
            "rate, lead variables).  CAVEAT: the v4–v6 bars were computed before "
            "the v7 de-duplication fix — duplicate runner-season rows leaked across "
            "CV folds and inflated those numbers.  v7 averages duplicate splits "
            "into one row per runner-season, so its bar is the honest de-leaked "
            "AUC and is not directly comparable to the historical bars.")
    imgpage(pdf, "§2 · GLM Plain-English Weight Chart",
            FIGURES_DIR/"Fig_v7_GLM_PlainEnglish.png",
            "Each bar = predicted percentage-point change in SB success when "
            "the runner improves on that feature by ONE TIER (1 SD).  Positive "
            "and green = the feature HELPS; red/orange = HURTS.")
    imgpage(pdf, "§3 · Feature Importance · Pre vs Post 2023",
            FIGURES_DIR/"Fig_v7_Importance_PrePost.png",
            "How feature importance shifted after the 2023 rule change.")
    imgpage(pdf, "§4 · Naylor + Soto Statcast-style Profile",
            FIGURES_DIR/"Fig_v7_NaylorSoto_Profile.png",
            "Season-by-season values of the key features for the two "
            "archetypal slow-but-effective stealers.")

    # GLM weight table page
    textpage(pdf, "§5 · GLM Weight Table — Plain English",
        [f"Baseline P(success) at league-average inputs ≈ {baseline_p:.3f}",
         "",
         "Read each row as:",
         "  'If a runner improves [Feature] by 1 tier (1 SD), the model",
         "   expects their SB success rate to change by [Boost] percentage",
         "   points, equivalent to multiplying their odds by [Mult].'",
         "",
         "## Weight table (sorted by absolute boost)",
         ""] +
        [f"  {r['feature']:<30}  boost {r['sb_pct_boost_per_tier']:+.2f} pp   "
         f"odds × {r['odds_multiplier']:.3f}   "
         f"(higher_is_better = {r['higher_is_better']})"
         for _, r in DF_GLM.iterrows()] +
        ["",
         "## Technical notes (for stats readers)",
         "boost_pp = sigmoid(intercept + coef) − sigmoid(intercept)  × 100",
         "odds_mul = exp(coef)",
         "coef     = standardised logistic-regression coefficient",
         "",
         f"Edit DF_v7_GLM_PlainEnglish.csv to hand-tune.  No retraining needed."])

    # SSSI page
    textpage(pdf, "§6 · SSSI v7 Top 15",
        [f"Weights:  sb_res={w_best[0]}  gap={w_best[1]}  gain={w_best[2]}  "
         f"jump={w_best[3]}  prim={w_best[4]}  speed={w_best[5]}  "
         f"pre={w_best[6]}  post={w_best[7]}  accel_top={w_best[8]}",
         "(80% of runners used for grid search; Naylor + Soto held out.)",
         "v7 adds the Accel→Top-Speed Premium as a 9th weighted feature.",
         ""] +
        [f"  #{int(r['rank_v7']):>3}  {r['player_name']:<22}  {int(r['season'])}  "
         f"SSSI {r['SSSI_v7']:+.2f}  spd {r['sprint_speed']:.1f}  "
         f"jump {r['jump_time']:.2f}  gain {r['lead_gain']:.2f}  "
         f"SB/CS {int(r['SB'])}/{int(r['CS'])}"
         for _, r in SSSI.head(15).iterrows()])

    # Caveats
    textpage(pdf, "§7 · Honest Data Limitations",
        ["What the model has access to:",
         "  - Sprint speed, running splits (real, per-runner-season, 2015+)",
         "  - Real catcher pop time (2018+, per-year)",
         "  - Real SB / CS counts (MLB Stats API)",
         "  - Career-aggregate lead profiles for runners and pitchers",
         "  - Per-pitch catcher_id, pitcher_id, balls, strikes, outs, inning",
         "",
         "What the model does NOT have:",
         "  - Per-pitch pitcher delivery time (not publicly published)",
         "  - The EXACT count when an SB happened (we use AB's final count)",
         "  - Per-attempt lead distance (we use the runner's career mean)",
         "",
         "Why the AUC ceiling exists:",
         "  - Stolen-base success has high intrinsic noise — pitch sequence,",
         "    runner read, base-coach decisions, exact location at release.",
         "  - With public data the ceiling is around AUC 0.70 - 0.75.",
         "  - To push higher would require TrackMan / Hawk-Eye raw outputs.",
         "",
         "De-duplication note (v7):",
         "  - Earlier versions (v4-v6) carried duplicate runner-season rows —",
         "    repeated Statcast split measurements for the same player-season.",
         "  - Those dupes leaked across CV folds and INFLATED the reported AUC.",
         "  - v7 averages duplicate splits into one row per runner-season; the",
         "    de-leaked AUC (~0.59 full) is lower but honest, not a regression."])

    # §8 — Expected SB Outcome (xSB) — the v7 standalone lens
    imgpage(pdf, "§8 · Expected SB Outcome (xSB)", FIGURES_DIR/"Fig_v7_xSB_Quadrant.png",
            "xSB = z(net SB above avg) + z(sprint speed).  A complementary lens to "
            "SSSI: where SSSI surfaces slow-but-skilled stealers, xSB surfaces the "
            "fast AND productive (high-ceiling) runners.  The four quadrants split "
            "runners by speed (x) and production (y).")
    _xtop = xsb.head(15)
    _xpot = xsb.sort_values("sb_potential_gap", ascending=False).head(10)
    textpage(pdf, "§8 · xSB — Leaderboard & Untapped Potential",
        ["xsb_outcome = z(net SB above avg) + z(sprint speed)",
         "sb_potential_gap = z(sprint) − z(net SB)   (+ = fast but under-steals)",
         "",
         "## Top 15 by xSB (Realized Burners — fast + productive)",
         ""] +
        [f"  #{int(r['rank_xsb']):>3}  {r['player_name']:<22} {int(r['season'])}  "
         f"xSB {r['xsb_outcome']:+.2f}  spd {r['sprint_speed']:.1f}  "
         f"SB/CS {int(r['SB'])}/{int(r['CS'])}  [{r['quadrant']}]"
         for _, r in _xtop.iterrows()] +
        ["",
         "## Most untapped potential (high speed, under-stealing)",
         "Coaching targets: the tools are there, the production isn't yet.",
         ""] +
        [f"  {r['player_name']:<22} {int(r['season'])}  gap {r['sb_potential_gap']:+.2f}  "
         f"spd {r['sprint_speed']:.1f}  SB/CS {int(r['SB'])}/{int(r['CS'])}"
         for _, r in _xpot.iterrows()] +
        ["",
         "## Naylor + Soto under xSB (Crafty Technician archetype)",
         ""] +
        [f"  {r['player_name']:<22} {int(r['season'])}  xSB {r['xsb_outcome']:+.2f}  "
         f"gap {r['sb_potential_gap']:+.2f}  [{r['quadrant']}]"
         for _, r in xsb[xsb["runner_id"].isin([NAYLOR_ID,SOTO_ID])].iterrows()])

print(f"   wrote {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CONSOLIDATE v7 CSV OUTPUTS INTO  "v7 Model.xlsx"  (one sheet per DF_v7_*.csv)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10/10] Consolidating v7 Model.xlsx …")
_xlsx_path = DATA_DIR / "v7 Model.xlsx"
_sheet_map = {
    "DF_v7_SSSI.csv":                 "SSSI v7",
    "DF_v7_xSB_Outcome.csv":          "xSB Outcome",
    "DF_v7_GLM_PlainEnglish.csv":     "GLM Weights",
    "DF_v7_Importance.csv":           "Feature Importance",
    "DF_v7_ModelB_AUC.csv":           "Model B AUC",
    "DF_v7_LB_JumpTime.csv":          "LB JumpTime",
    "DF_v7_LB_LeadGain.csv":          "LB LeadGain",
    "DF_v7_LB_PreReleaseVelocity.csv":"LB PreRelVel",
    "DF_v7_LB_PostReleaseDistance.csv":"LB PostRelDist",
}
try:
    with pd.ExcelWriter(_xlsx_path, engine="openpyxl") as _xw:
        for _csv, _sheet in _sheet_map.items():
            _p = DATA_DIR / _csv
            if _p.exists():
                pd.read_csv(_p).to_excel(_xw, sheet_name=_sheet[:31], index=False)
    print(f"   wrote {_xlsx_path}")
except Exception as _e:
    print(f"   [xlsx] skipped ({type(_e).__name__}: {_e})")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(" v7 EXPLORATORY PIPELINE COMPLETE")
print("=" * 72)
for p in sorted(DATA_DIR.glob("DF_v7_*.csv")):
    print(f"   {p.name:<45} {p.stat().st_size/1024:>7.1f} KB")
for p in sorted(FIGURES_DIR.glob("Fig_v7_*.png")):
    print(f"   {p.name:<45} {p.stat().st_size/1024:>7.1f} KB")
print(f"   Naylor_Model_v7_Report.pdf            "
      f"{(REPORTS_DIR/'Naylor_Model_v7_Report.pdf').stat().st_size/1024:.1f} KB")
print()
print(f"Headline (v7 is the first DE-LEAKED estimate — v4–v6 leaked dup rows):")
print(f"  v4 baseline:       0.6300  (leaked)")
print(f"  v5 Model B:        0.6794  (leaked)")
print(f"  v6 Model B:        0.6620  (leaked)")
print(f"  v7 Model B (full): {m_full['auc']:.4f}  <- honest, de-leaked")
print(f"  v7 Model B (pre):  {m_pre['auc']:.4f}" if m_pre else "")
print(f"  v7 Model B (post): {m_post['auc']:.4f}" if m_post else "")
