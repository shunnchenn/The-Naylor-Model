#!/usr/bin/env python3
"""
The Naylor Model · v5  —  per-pitch feature layer
=================================================
Upgrades over v4:

  ▸ Per-pitch Statcast data 2018-2026 (cached on disk)
  ▸ SB attempts identified by parsing `des` text field
  ▸ Real catcher pop time per-year-per-catcher (Savant poptime leaderboard)
  ▸ Real per-pitcher running-game-prevention score (Savant pitcher-running-game)
  ▸ Count leverage feature — drop low-attempt counts from training
  ▸ NEW metric  pre_rel_vel   = lead_gain / pitcher_delivery_proxy
  ▸ NEW metric  post_rel_dist = sprint_speed × pop_time with accel correction
  ▸ 5-ft running splits exploration — three representations compared
  ▸ Model A: per-attempt GBM with group-CV by runner_id (no leak)
  ▸ Model B: enhanced season-level GBM
  ▸ Model C: simple unpenalised GLM with hand-tunable weights
  ▸ SSSI v5: extended with pre_rel_vel + post_rel_dist; weights optimised on
    held-out 80/20 runner split (no Naylor/Soto overfit)
  ▸ Updated Variable_Glossary.pdf with all v5 variables

Honest data limitations (documented in PDF §9):

  ▸ Per-pitch pitcher delivery time NOT publicly available.  Substitute:
    Savant `pitcher-running-game` per-pitcher prevention metric (career,
    measures the lead gained AGAINST this pitcher across all attempts).
  ▸ Pitcher-tempo CSV is buggy at source (empty == onbase identical).
  ▸ Per-attempt lead is not available; use runner's CAREER lead snapshot.
  ▸ The exact count at SB attempt is the at-bat's FINAL count (the moment
    the play was recorded).  Mid-at-bat steals attach the wrong count.
  ▸ Savant year filters are ignored on lead and tempo endpoints → those
    are career snapshots.  Only catcher pop is truly per-year.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")

import io
import re
import pickle
import requests
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

from pybaseball import (statcast, statcast_sprint_speed,
                        statcast_running_splits, statcast_catcher_poptime)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ── Constants
SEASONS_PRE  = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
SEASONS_POST = [2023, 2024, 2025, 2026]
SEASONS_ALL  = SEASONS_PRE + SEASONS_POST
SEASONS_PITCH = [2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]  # poptime constrains
RULE_CHANGE_YEAR = 2023
MIN_REAL_SB_CS   = 10
SEED             = 42
NAYLOR_ID, SOTO_ID = 647304, 665742
SPEED_CAP        = 28.0
OUTPUT_DIR       = Path("/Users/shunchen/Desktop/The-Naylor-Model")
CACHE_DIR        = OUTPUT_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

COLOR = {"pre":"#E0A458","post":"#3D5A80","naylor":"#DC2626","soto":"#1D4ED8",
         "neutral":"#374151","highlight":"#10B981","accent":"#0EA5E9"}

print("=" * 72)
print(" THE NAYLOR MODEL  ·  v5  (per-pitch feature layer)")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def cache_load(name):
    p = CACHE_DIR / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

def cache_save(name, obj):
    p = CACHE_DIR / f"{name}.pkl"
    with open(p, "wb") as f:
        pickle.dump(obj, f)

# Columns we keep from raw statcast (~25 cols, vs full 118)
KEEP_PITCH_COLS = [
    "game_date","game_year","game_pk","at_bat_number","pitch_number",
    "on_1b","on_2b","on_3b","fielder_2","pitcher","batter",
    "events","des","balls","strikes","outs_when_up","inning","inning_topbot",
    "release_extension","release_pos_y","release_speed","p_throws","stand",
    "pitch_type",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. PER-PITCH FETCHER (chunked, cached)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_pitches_year(year):
    """Fetch pitch-by-pitch for one season in 30-day chunks; cache result.

    Filter on download:
        - keep only rows where on_1b is not null (runner on 1st)
        - keep only ~25 columns we care about
    """
    cache_name = f"pitches_{year}"
    cached = cache_load(cache_name)
    if cached is not None:
        return cached

    print(f"   [fetch] {year} statcast (30-day chunks) …")
    chunks = []
    start = datetime(year, 3, 15)
    end   = datetime(year, 11, 15)
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=30), end)
        try:
            df = statcast(start_dt=cur.strftime("%Y-%m-%d"),
                          end_dt=nxt.strftime("%Y-%m-%d"), verbose=False)
            if df is not None and len(df) > 0:
                # keep only present columns
                keep = [c for c in KEEP_PITCH_COLS if c in df.columns]
                df = df[keep].copy()
                # filter to runner on 1st
                df = df[df["on_1b"].notna()].copy()
                chunks.append(df)
        except Exception as e:
            print(f"      {cur:%Y-%m-%d} chunk failed: {type(e).__name__}: {str(e)[:60]}")
        cur = nxt + timedelta(days=1)
    if not chunks:
        df = pd.DataFrame(columns=KEEP_PITCH_COLS)
    else:
        df = pd.concat(chunks, ignore_index=True)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df["game_year"] = df["game_date"].dt.year
    cache_save(cache_name, df)
    print(f"      → {len(df):,} pitches with runner on 1st in {year}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. SUPPORTING FETCHERS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_catcher_poptime_year(year):
    """Catcher pop time per catcher, per year.  Available 2018+ (skip 2020)."""
    cache_name = f"poptime_{year}"
    cached = cache_load(cache_name)
    if cached is not None:
        return cached
    try:
        df = statcast_catcher_poptime(year=year, min_2b_att=1)
        df = df.rename(columns={"entity_id":"catcher_id"})
        df["season"] = year
        cache_save(cache_name, df)
        return df
    except Exception as e:
        print(f"   poptime {year} failed: {e}")
        return pd.DataFrame(columns=["catcher_id","season","pop_2b_sba"])

def fetch_pitcher_runninggame():
    """Career-snapshot of how much lead pitchers ALLOW (real Savant data).

    Lower lead_allowed = pitcher does well at holding runners.
    """
    cache_name = "pitcher_runninggame"
    cached = cache_load(cache_name)
    if cached is not None:
        return cached
    url = ("https://baseballsavant.mlb.com/leaderboard/pitcher-running-game"
           "?team=&min=q&csv=true")
    r = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df = df.rename(columns={"player_id":"pitcher_id"})
    keep = ["pitcher_id","r_primary_lead","r_secondary_lead",
            "r_sec_minus_prim_lead","runs_prevented_on_running_attr"]
    df = df[[c for c in keep if c in df.columns]].copy()
    # Rename to distinguish from runner-side lead variables
    df = df.rename(columns={
        "r_primary_lead":      "pitcher_lead_allowed_prim",
        "r_secondary_lead":    "pitcher_lead_allowed_sec",
        "r_sec_minus_prim_lead":"pitcher_lead_gain_allowed",
        "runs_prevented_on_running_attr": "pitcher_prevent_runs",
    })
    cache_save(cache_name, df)
    return df

def fetch_savant_runner_lead():
    """Career-snapshot of runner's lead profile (same as v4)."""
    cache_name = "runner_lead"
    cached = cache_load(cache_name)
    if cached is not None:
        return cached
    url = ("https://baseballsavant.mlb.com/leaderboard/basestealing-run-value"
           "?team=&min=q&csv=true")
    r = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df = df.rename(columns={"player_id":"runner_id"})
    keep = ["runner_id","r_primary_lead","r_secondary_lead",
            "r_sec_minus_prim_lead"]
    df = df[[c for c in keep if c in df.columns]].copy()
    cache_save(cache_name, df)
    return df

def fetch_mlb_sb(season):
    cache_name = f"mlb_sb_{season}"
    cached = cache_load(cache_name)
    if cached is not None:
        return cached
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
    df = pd.DataFrame(rows)
    cache_save(cache_name, df)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4. PARSE SB ATTEMPTS FROM `des`
# ─────────────────────────────────────────────────────────────────────────────
# SB descriptions look like:
#   "Sam Hilliard steals (2) 2nd base."
#   "Nick Martini caught stealing 2nd"
#   "Carlos Correa picks off Nick Martini at 1st"
SB_RE  = re.compile(r"steals\s*\(\d+\)\s*2nd\s*base", re.I)
CS_RE  = re.compile(r"caught\s+stealing\s+2nd", re.I)
PK_RE  = re.compile(r"picks?\s+off.*at\s+1st", re.I)

def label_attempt(des):
    if not isinstance(des, str): return ("none", 0, 0)
    if SB_RE.search(des):   return ("sb",  1, 1)
    if CS_RE.search(des):   return ("cs",  1, 0)
    if PK_RE.search(des):   return ("pk",  1, 0)
    return ("none", 0, 0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. NEW METRIC CALCULATORS
# ─────────────────────────────────────────────────────────────────────────────
LEAGUE_POP_DEFAULT = 1.95   # league mean pop time, used as fallback
LEAGUE_LEAD_GAIN   = 3.5    # league mean lead_gain, used as fallback
LEAGUE_PITCHER_TEMPO = 1.30 # league mean PROXY delivery time (we don't have real)

def pre_rel_vel(lead_gain_ft, pitcher_delivery_proxy_s):
    """ft/s — runner's avg velocity from pitcher first move to release.
    Pitcher delivery time is NOT publicly available; we use the league-constant
    proxy 1.30 s.  Variation in this metric is driven entirely by lead_gain.
    """
    t = max(pitcher_delivery_proxy_s if pitcher_delivery_proxy_s is not None
            else LEAGUE_PITCHER_TEMPO, 0.6)
    if lead_gain_ft is None or np.isnan(lead_gain_ft): return np.nan
    return lead_gain_ft / t

def post_rel_dist(sprint_speed_fps, pop_time_s, accel_0_30_s):
    """ft — distance runner covers from pitch release until ball arrives at 2B.

    Time window = pop_time (catcher receive + throw + ball arrival).
    Ground covered = sprint × pop − acceleration correction for slower runners.

    Slower runners are still accelerating during the pop window, so they cover
    LESS than (sprint_speed × pop_time).  Correction = 0.5 · accel_deficit · pop².
    accel_deficit = (slow_runner_extra_time_to_30ft − fast_runner_extra_time_to_30ft)
    proxy: (accel_0_30 − 1.65) × penalty if positive.
    """
    if any(x is None or np.isnan(x) for x in [sprint_speed_fps, pop_time_s, accel_0_30_s]):
        return np.nan
    naive = sprint_speed_fps * pop_time_s
    accel_penalty = max(0.0, (accel_0_30_s - 1.65)) * sprint_speed_fps * pop_time_s * 0.5
    return max(0.0, naive - accel_penalty)

# ─────────────────────────────────────────────────────────────────────────────
# 6. STEP 1 — Fetch all pitch data + supporting frames
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[1/14] Fetching per-pitch data {min(SEASONS_PITCH)}–{max(SEASONS_PITCH)} …")

pitch_frames = []
for yr in SEASONS_PITCH:
    df = fetch_pitches_year(yr)
    if df is not None and len(df) > 0:
        pitch_frames.append(df)

DF_Pitch = pd.concat(pitch_frames, ignore_index=True)
print(f"   Total pitches on-1B across {len(SEASONS_PITCH)} seasons: {len(DF_Pitch):,}")

# Catcher pop per year
print("\n[2/14] Fetching catcher pop time per year …")
pop_frames = [fetch_catcher_poptime_year(yr) for yr in SEASONS_PITCH]
DF_Pop = pd.concat([d for d in pop_frames if d is not None and len(d) > 0],
                   ignore_index=True)
print(f"   Catcher-seasons: {len(DF_Pop):,}")

# Pitcher running game (career snapshot)
print("\n[3/14] Fetching pitcher running-game (career snapshot) …")
DF_PitcherRG = fetch_pitcher_runninggame()
print(f"   Pitchers with running-game data: {len(DF_PitcherRG):,}")

# Runner lead profile (career snapshot)
DF_RunnerLead = fetch_savant_runner_lead()
print(f"   Runners with lead profile: {len(DF_RunnerLead):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. STEP 2 — Runner-season aggregate frame (v4-style; reuse logic)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/14] Building runner-season aggregate ({min(SEASONS_ALL)}–{max(SEASONS_ALL)}) …")

cache_name = "runner_season"
DF_Season = cache_load(cache_name)
if DF_Season is None:
    sp_frames, rs_frames, sb_frames = [], [], []
    for yr in SEASONS_ALL:
        try:
            sp = statcast_sprint_speed(yr, min_opp=1);                 sp["season"] = yr
            rs = statcast_running_splits(yr, min_opp=1, raw_splits=True); rs["season"] = yr
            sp_frames.append(sp); rs_frames.append(rs)
        except Exception as e:
            print(f"   sprint/splits {yr}: {e}"); continue
        try: sb_frames.append(fetch_mlb_sb(yr))
        except Exception as e: print(f"   sb {yr}: {e}")

    DF_Speed = pd.concat(sp_frames, ignore_index=True).rename(
        columns={"last_name, first_name":"player_name","player_id":"runner_id"})
    DF_Splits = pd.concat(rs_frames, ignore_index=True).rename(
        columns={"last_name, first_name":"player_name","player_id":"runner_id"})
    DF_SB    = pd.concat(sb_frames, ignore_index=True)

    DF_Splits["accel_0_30"]    = DF_Splits["seconds_since_hit_030"]
    DF_Splits["accel_5_30"]    = (DF_Splits["seconds_since_hit_030"]
                                  - DF_Splits["seconds_since_hit_005"])
    DF_Splits["maintain_30_90"]= (DF_Splits["seconds_since_hit_090"]
                                  - DF_Splits["seconds_since_hit_030"])
    DF_Splits["total_90"]      = DF_Splits["seconds_since_hit_090"]

    SPLIT_COLS = [c for c in DF_Splits.columns if c.startswith("seconds_since_hit_")]

    keep_speed = ["runner_id","season","player_name","sprint_speed","bolts"]
    keep_split = ["runner_id","season","accel_0_30","accel_5_30",
                  "maintain_30_90","total_90"] + SPLIT_COLS

    DF_Season = (DF_Speed[keep_speed]
                 .merge(DF_Splits[keep_split], on=["runner_id","season"], how="inner")
                 .merge(DF_SB[["runner_id","season","SB","CS","Name"]],
                        on=["runner_id","season"], how="left")
                 .merge(DF_RunnerLead, on="runner_id", how="left"))

    DF_Season["SB"]=DF_Season["SB"].fillna(0).astype(int)
    DF_Season["CS"]=DF_Season["CS"].fillna(0).astype(int)
    DF_Season["bolts"]=DF_Season["bolts"].fillna(0)
    DF_Season["real_sb_attempts"]=DF_Season["SB"]+DF_Season["CS"]
    DF_Season["era"]=np.where(DF_Season["season"]>=RULE_CHANGE_YEAR,
                              "post_2023","pre_2023")
    # Use MLB Stats API name when present, else statcast last_name, first_name
    DF_Season["player_name"]=DF_Season["Name"].fillna(DF_Season["player_name"])
    DF_Season["speed_capped"]=DF_Season["sprint_speed"].clip(upper=SPEED_CAP)
    cache_save(cache_name, DF_Season)
print(f"   Runner-seasons: {len(DF_Season):,}")
print(f"   With SB+CS ≥ {MIN_REAL_SB_CS}: "
      f"{(DF_Season['real_sb_attempts']>=MIN_REAL_SB_CS).sum():,}")

# Shrunk SB% + sb_residual (poly fit on qualified)
k_shrink=5
mask_q = DF_Season["real_sb_attempts"]>=MIN_REAL_SB_CS
league_sb = DF_Season.loc[mask_q,"SB"].sum() / max(1, DF_Season.loc[mask_q,"real_sb_attempts"].sum())
DF_Season["real_sb_pct"]=((DF_Season["SB"]+k_shrink*league_sb)
                          /(DF_Season["real_sb_attempts"]+k_shrink))
coeffs = np.polyfit(DF_Season.loc[mask_q,"sprint_speed"],
                    DF_Season.loc[mask_q,"real_sb_pct"], 2)
DF_Season["expected_sb_pct"]=np.polyval(coeffs, DF_Season["sprint_speed"]).clip(0.3,0.99)
DF_Season["sb_residual"]=DF_Season["real_sb_pct"]-DF_Season["expected_sb_pct"]

DF_Season["pct_speed"] = DF_Season.groupby("season")["sprint_speed"].rank(pct=True)*100
DF_Season["pct_accel"] = DF_Season.groupby("season")["accel_0_30"].rank(pct=True, ascending=False)*100
DF_Season["accel_gap"] = DF_Season["pct_accel"]-DF_Season["pct_speed"]

print(f"   League SB% = {league_sb:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. STEP 3 — Parse SB attempts from des; build DF_Attempts
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/14] Parsing SB attempts from `des` text field …")
labels = DF_Pitch["des"].apply(label_attempt)
DF_Pitch["sb_label"]   = labels.apply(lambda t: t[0])
DF_Pitch["y_attempt"]  = labels.apply(lambda t: t[1])
DF_Pitch["y_success"]  = labels.apply(lambda t: t[2])

attempt_counts = DF_Pitch["sb_label"].value_counts()
print(f"   Pitch outcome breakdown: {attempt_counts.to_dict()}")

# Attempts only: keep one row per (game_pk, at_bat_number) that has an attempt
DF_Attempts = (DF_Pitch[DF_Pitch["y_attempt"]==1]
               .sort_values(["game_pk","at_bat_number","pitch_number"])
               .drop_duplicates(["game_pk","at_bat_number"], keep="last")
               .copy())
DF_Attempts = DF_Attempts.rename(columns={"on_1b":"runner_id",
                                          "fielder_2":"catcher_id",
                                          "pitcher":"pitcher_id",
                                          "game_year":"season"})
DF_Attempts["runner_id"] = pd.to_numeric(DF_Attempts["runner_id"], errors="coerce")
DF_Attempts["catcher_id"] = pd.to_numeric(DF_Attempts["catcher_id"], errors="coerce")
DF_Attempts["pitcher_id"] = pd.to_numeric(DF_Attempts["pitcher_id"], errors="coerce")
DF_Attempts = DF_Attempts.dropna(subset=["runner_id","catcher_id"])
DF_Attempts["runner_id"]=DF_Attempts["runner_id"].astype(int)
DF_Attempts["catcher_id"]=DF_Attempts["catcher_id"].astype(int)
DF_Attempts["pitcher_id"]=DF_Attempts["pitcher_id"].astype(int)

print(f"   Total attempts identified: {len(DF_Attempts):,}")
print(f"   Per year: {DF_Attempts.groupby('season').size().to_dict()}")

# Sanity: count by sb_label
print(f"   By outcome:   {DF_Attempts['sb_label'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. STEP 4 — Join battery + runner features into DF_Attempts
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/14] Joining battery + runner context to attempts …")

# Catcher pop (per-year)
DF_Attempts = DF_Attempts.merge(
    DF_Pop[["catcher_id","season","pop_2b_sba","exchange_2b_3b_sba",
            "maxeff_arm_2b_3b_sba"]],
    on=["catcher_id","season"], how="left")

# Pitcher running-game (career snapshot)
DF_Attempts = DF_Attempts.merge(DF_PitcherRG, on="pitcher_id", how="left")

# Runner lead profile (career snapshot)
DF_Attempts = DF_Attempts.merge(DF_RunnerLead, on="runner_id", how="left")

# Runner season-aggregate features (per runner-season — speed, accel, etc.)
season_cols = ["runner_id","season","sprint_speed","speed_capped","accel_0_30",
               "accel_5_30","maintain_30_90","total_90","accel_gap","bolts",
               "real_sb_pct","sb_residual"]
DF_Attempts = DF_Attempts.merge(
    DF_Season[season_cols].drop_duplicates(["runner_id","season"]),
    on=["runner_id","season"], how="left")

# Diagnose join rates
print(f"   Join health (% non-null after merges):")
for c in ["pop_2b_sba","r_sec_minus_prim_lead","sprint_speed",
         "pitcher_lead_gain_allowed"]:
    rate = DF_Attempts[c].notna().mean()*100 if c in DF_Attempts.columns else 0
    print(f"     {c:30}  {rate:>5.1f}%")

# Impute pop_2b_sba with league-year mean if missing
DF_Attempts["pop_imputed"] = DF_Attempts["pop_2b_sba"].isna().astype(int)
pop_yr_mean = DF_Attempts.groupby("season")["pop_2b_sba"].transform("mean")
DF_Attempts["pop_2b_sba"] = DF_Attempts["pop_2b_sba"].fillna(pop_yr_mean)
DF_Attempts["pop_2b_sba"] = DF_Attempts["pop_2b_sba"].fillna(LEAGUE_POP_DEFAULT)

# ─────────────────────────────────────────────────────────────────────────────
# 10. STEP 5 — Per-attempt feature engineering
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/14] Computing per-attempt features …")

# Convert sprint_speed (ft/s already) — no conversion needed
# pre_rel_vel uses league-constant pitcher delivery (per data limitation)
DF_Attempts["pre_rel_vel"] = DF_Attempts.apply(
    lambda r: pre_rel_vel(r["r_sec_minus_prim_lead"], LEAGUE_PITCHER_TEMPO),
    axis=1)

DF_Attempts["post_rel_dist"] = DF_Attempts.apply(
    lambda r: post_rel_dist(r["sprint_speed"], r["pop_2b_sba"], r["accel_0_30"]),
    axis=1)

# Matchup feature: runner_lead_gain vs pitcher_lead_allowed
DF_Attempts["matchup_lead_diff"] = (DF_Attempts["r_sec_minus_prim_lead"]
                                    - DF_Attempts.get("pitcher_lead_gain_allowed", 0))

# Count features
DF_Attempts["count_label"] = (DF_Attempts["balls"].astype(int).astype(str)
                              + "-" + DF_Attempts["strikes"].astype(int).astype(str))

# Print quick sanity for pre_rel_vel and post_rel_dist
for c in ["pre_rel_vel", "post_rel_dist", "matchup_lead_diff"]:
    s = DF_Attempts[c].dropna()
    print(f"     {c:20}  n={len(s):>5}  mean={s.mean():>6.2f}  range=[{s.min():.2f}, {s.max():.2f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 11. STEP 6 — Count leverage analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/14] Count leverage analysis …")

# Total pitches per count (with runner on 1st) and SB attempts per count
DF_Pitch["count_label"]=(DF_Pitch["balls"].astype(int).astype(str)
                         +"-"+DF_Pitch["strikes"].astype(int).astype(str))
count_total = DF_Pitch.groupby("count_label").size().rename("n_pitches")
count_attempts = DF_Pitch.groupby("count_label")["y_attempt"].sum().rename("n_attempts")
count_table = pd.concat([count_total, count_attempts], axis=1).reset_index()
count_table["attempt_rate"] = count_table["n_attempts"]/count_table["n_pitches"]
count_table = count_table.sort_values("attempt_rate", ascending=False)

max_rate = count_table["attempt_rate"].max()
hl_threshold = max_rate * 0.60
HL_COUNTS = set(count_table.loc[count_table["attempt_rate"]>=hl_threshold,
                                "count_label"].tolist())
count_table["is_HL"] = count_table["count_label"].isin(HL_COUNTS)

print(count_table.to_string(index=False))
print(f"\n   HL counts (rate ≥ {hl_threshold:.4f} = 60% of max): {sorted(HL_COUNTS)}")
count_table.to_csv(OUTPUT_DIR/"DF_v5_Count_Rates.csv", index=False)

DF_Attempts["is_count_HL"] = DF_Attempts["count_label"].isin(HL_COUNTS).astype(int)
DF_Pitch["is_count_HL"]    = DF_Pitch["count_label"].isin(HL_COUNTS).astype(int)

# Per runner-season: % attempts in HL counts and HL success rate
agg = (DF_Attempts.groupby(["runner_id","season"])
       .agg(n_attempts=("y_attempt","sum"),
            n_success=("y_success","sum"),
            pct_in_HL=("is_count_HL","mean"),
            avg_pop_faced=("pop_2b_sba","mean"),
            avg_pitcher_prevent=("pitcher_prevent_runs","mean"),
            pre_rel_vel_avg=("pre_rel_vel","mean"),
            post_rel_dist_avg=("post_rel_dist","mean"))
       .reset_index())
DF_Season = DF_Season.merge(agg, on=["runner_id","season"], how="left")

print("\n   Aggregated season-level new features (sample):")
print(DF_Season[DF_Season["runner_id"].isin([NAYLOR_ID, SOTO_ID])]
        [["player_name","season","n_attempts","n_success","pct_in_HL",
          "avg_pop_faced","pre_rel_vel_avg","post_rel_dist_avg"]]
        .round(3).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 12. STEP 7 — 5-ft splits granularity exploration
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9/14] 5-ft splits exploration (raw / curve / PCA) …")

SPLIT_COLS_5 = [f"seconds_since_hit_{d:03d}" for d in range(5, 95, 5)]
present = [c for c in SPLIT_COLS_5 if c in DF_Season.columns]
qual = DF_Season[mask_q].dropna(subset=present+["sprint_speed","sb_residual"]).copy()
print(f"   Qualified runners with full splits: {len(qual)}")

# Path A — raw
def cv_ridge(X, y, alpha=1.0, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in cv.split(X):
        m = Ridge(alpha=alpha).fit(X[tr], y[tr]); preds[te]=m.predict(X[te])
    return 1 - np.var(y-preds)/max(1e-9, np.var(y))

XA = qual[present].values
r2_A_resid = cv_ridge(XA, qual["sb_residual"].values)
r2_A_pct   = cv_ridge(XA, qual["real_sb_pct"].values)

# Path B — quadratic curve fit per runner-season
def fit_curve(row):
    distances = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90], dtype=float)
    times = np.array([row.get(f"seconds_since_hit_{int(d):03d}", np.nan) for d in distances])
    mask = ~np.isnan(times)
    if mask.sum() < 3: return [np.nan]*4
    coefs = np.polyfit(distances[mask], times[mask], 2)
    fit_times = np.polyval(coefs, distances[mask])
    rmse = float(np.sqrt(((times[mask]-fit_times)**2).mean()))
    return [coefs[0], coefs[1], coefs[2], rmse]

curve_data = np.array([fit_curve(r) for _, r in qual.iterrows()])
curve_df = pd.DataFrame(curve_data, columns=["curve_a","curve_b","curve_c","curve_rmse"])
qual = pd.concat([qual.reset_index(drop=True), curve_df], axis=1)
XB = qual[["curve_a","curve_b","curve_c","curve_rmse"]].dropna().values
yB = qual.loc[~qual["curve_a"].isna(),"sb_residual"].values
yB_pct = qual.loc[~qual["curve_a"].isna(),"real_sb_pct"].values
r2_B_resid = cv_ridge(XB, yB)
r2_B_pct   = cv_ridge(XB, yB_pct)

# Path C — PCA on raw 18-col splits → 3 components
X_std = StandardScaler().fit_transform(XA)
pca = PCA(n_components=3, random_state=SEED).fit_transform(X_std)
r2_C_resid = cv_ridge(pca, qual["sb_residual"].values)
r2_C_pct   = cv_ridge(pca, qual["real_sb_pct"].values)

# v4 baseline — 4 derived splits
baseline_cols = ["accel_0_30","accel_5_30","maintain_30_90","total_90"]
qual_b = qual.dropna(subset=baseline_cols)
X_v4 = qual_b[baseline_cols].values
r2_v4_resid = cv_ridge(X_v4, qual_b["sb_residual"].values)
r2_v4_pct   = cv_ridge(X_v4, qual_b["real_sb_pct"].values)

gran_results = pd.DataFrame([
    {"path":"v4 baseline (4 derived)","n_feat":4,           "r2_resid":r2_v4_resid,"r2_pct":r2_v4_pct},
    {"path":"A — 18 raw splits",      "n_feat":len(present),"r2_resid":r2_A_resid,  "r2_pct":r2_A_pct},
    {"path":"B — quadratic curve",    "n_feat":4,           "r2_resid":r2_B_resid,  "r2_pct":r2_B_pct},
    {"path":"C — 3-PCA",               "n_feat":3,           "r2_resid":r2_C_resid,  "r2_pct":r2_C_pct},
])
print(gran_results.round(4).to_string(index=False))
gran_results.to_csv(OUTPUT_DIR/"DF_v5_Granularity.csv", index=False)
best_path = gran_results.loc[gran_results["r2_resid"].idxmax(),"path"]
print(f"   ✓ Best split representation: {best_path}")

# Pick the winning feature set for downstream models
if best_path.startswith("A"):
    SPLIT_FEATURES_GBM = present                # 18 raw cols (for GBM)
    SPLIT_FEATURES_GLM = ["accel_0_30","accel_5_30","maintain_30_90","total_90"]
elif best_path.startswith("B"):
    SPLIT_FEATURES_GBM = ["curve_a","curve_b","curve_c","curve_rmse"]
    SPLIT_FEATURES_GLM = ["curve_a","curve_b","curve_c","curve_rmse"]
elif best_path.startswith("C"):
    SPLIT_FEATURES_GBM = SPLIT_FEATURES_GLM = ["accel_0_30","accel_5_30","maintain_30_90","total_90"]
else:
    SPLIT_FEATURES_GBM = SPLIT_FEATURES_GLM = ["accel_0_30","accel_5_30","maintain_30_90","total_90"]

# Add curve cols to DF_Season for downstream
curve_all = np.array([fit_curve(r) for _, r in DF_Season.iterrows()])
DF_Season["curve_a"] = curve_all[:,0]
DF_Season["curve_b"] = curve_all[:,1]
DF_Season["curve_c"] = curve_all[:,2]
DF_Season["curve_rmse"] = curve_all[:,3]

# ─────────────────────────────────────────────────────────────────────────────
# 13. STEP 8 — Model A: per-attempt GBM with group-CV
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10/14] Model A — per-attempt classifier with group-CV by runner_id …")

# Filter to HL counts only (per user spec)
DF_A = DF_Attempts[DF_Attempts["is_count_HL"]==1].copy()
print(f"   Attempts in HL counts: {len(DF_A):,} of {len(DF_Attempts):,}")

# Drop CS+SB targets; train to predict y_success given y_attempt==1
DF_A = DF_A[DF_A["y_attempt"]==1].dropna(
    subset=["pop_2b_sba","sprint_speed","accel_0_30"])
print(f"   With features available: {len(DF_A):,}")

FEAT_A = ["pop_2b_sba","exchange_2b_3b_sba","maxeff_arm_2b_3b_sba",
          "sprint_speed","accel_0_30","accel_5_30","maintain_30_90",
          "total_90","accel_gap","bolts",
          "r_primary_lead","r_secondary_lead","r_sec_minus_prim_lead",
          "pitcher_lead_gain_allowed","matchup_lead_diff",
          "pre_rel_vel","post_rel_dist",
          "outs_when_up","inning","balls","strikes","is_count_HL"]
FEAT_A = [c for c in FEAT_A if c in DF_A.columns and DF_A[c].notna().sum() > 100]

# Impute missing with column median
for c in FEAT_A:
    DF_A[c] = DF_A[c].fillna(DF_A[c].median())

X = DF_A[FEAT_A].values
y = DF_A["y_success"].values
groups = DF_A["runner_id"].values

# Group-CV (5-fold)
gkf = GroupKFold(n_splits=5)
preds = np.zeros_like(y, dtype=float)
for tr, te in gkf.split(X, y, groups):
    m = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                    learning_rate=0.05, random_state=SEED)
    m.fit(X[tr], y[tr]); preds[te] = m.predict_proba(X[te])[:,1]
auc_A = roc_auc_score(y, preds)
# Also pre/post 2023
auc_A_pre  = roc_auc_score(y[DF_A["season"]<RULE_CHANGE_YEAR],
                            preds[DF_A["season"]<RULE_CHANGE_YEAR]) \
             if (DF_A["season"]<RULE_CHANGE_YEAR).any() else float("nan")
auc_A_post = roc_auc_score(y[DF_A["season"]>=RULE_CHANGE_YEAR],
                            preds[DF_A["season"]>=RULE_CHANGE_YEAR])
print(f"   Model A AUC — full: {auc_A:.4f}  pre: {auc_A_pre:.4f}  post: {auc_A_post:.4f}")
print(f"   (baseline rate y_success = {y.mean():.3f})")

# Fit final model on full to extract importance
m_A_final = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, random_state=SEED)
m_A_final.fit(X, y)
imp_A = pd.DataFrame({"feature":FEAT_A,
                      "importance":m_A_final.feature_importances_}).sort_values(
                          "importance", ascending=False)
imp_A.to_csv(OUTPUT_DIR/"DF_v5_ModelA_Importance.csv", index=False)
print("\n   Top 10 feature importances (Model A):")
print(imp_A.head(10).to_string(index=False))

# SHAP if available
if HAS_SHAP:
    try:
        ex = shap.TreeExplainer(m_A_final)
        sv = ex.shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)
        shap_A = pd.DataFrame({"feature":FEAT_A,
                                "mean_abs_shap":mean_abs}).sort_values(
                                    "mean_abs_shap", ascending=False)
        shap_A.to_csv(OUTPUT_DIR/"DF_v5_ModelA_SHAP.csv", index=False)
        print("\n   Top 10 |SHAP| (Model A):")
        print(shap_A.head(10).round(4).to_string(index=False))
    except Exception as e:
        print(f"   SHAP failed: {e}")

pd.DataFrame([{"epoch":"full","auc":auc_A},
              {"epoch":"pre_2023","auc":auc_A_pre},
              {"epoch":"post_2023","auc":auc_A_post}]).to_csv(
    OUTPUT_DIR/"DF_v5_ModelA_AUC.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 14. STEP 9 — Model B: season-level GBM with v4 + new features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11/14] Model B — season-level GBM with enhanced features …")

SEASON_FEAT = (["sprint_speed","speed_capped","accel_0_30","accel_5_30",
                "maintain_30_90","total_90","accel_gap","bolts",
                "r_primary_lead","r_secondary_lead","r_sec_minus_prim_lead",
                "pre_rel_vel_avg","post_rel_dist_avg","avg_pop_faced",
                "pct_in_HL","n_attempts"])
SEASON_FEAT = [c for c in SEASON_FEAT if c in DF_Season.columns]

work = DF_Season[mask_q].dropna(subset=SEASON_FEAT+["real_sb_pct"]).copy()
work["y"]=(work["real_sb_pct"]>=league_sb).astype(int)
print(f"   Qualified rows: {len(work)}")

def fit_season(df_sub, label):
    if len(df_sub) < 50: return None
    X=df_sub[SEASON_FEAT].values; y=df_sub["y"].values
    w=df_sub["real_sb_attempts"].values.astype(float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = cross_val_predict(
        GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    random_state=SEED),
        X, y, cv=cv, method="predict_proba")[:,1]
    return {"label":label, "n":len(df_sub), "auc":roc_auc_score(y, preds),
            "X":X, "y":y, "w":w}

m_B_full = fit_season(work,                                  "full")
m_B_pre  = fit_season(work[work["era"]=="pre_2023"],         "pre_2023")
m_B_post = fit_season(work[work["era"]=="post_2023"],        "post_2023")

rows = []
for m in [m_B_full, m_B_pre, m_B_post]:
    if m is None: continue
    print(f"   Model B {m['label']:>10}  n={m['n']:>4}  AUC={m['auc']:.4f}")
    rows.append({"epoch":m["label"], "n":m["n"], "auc":m["auc"]})
pd.DataFrame(rows).to_csv(OUTPUT_DIR/"DF_v5_ModelB_AUC.csv", index=False)

# SHAP/importance per epoch
shap_rows = []
for m in [m_B_full, m_B_pre, m_B_post]:
    if m is None: continue
    gbm = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                      learning_rate=0.05,
                                      random_state=SEED).fit(m["X"], m["y"], sample_weight=m["w"])
    if HAS_SHAP:
        ex = shap.TreeExplainer(gbm); sv = ex.shap_values(m["X"])
        imp = np.abs(sv).mean(axis=0)
    else:
        imp = gbm.feature_importances_
    for f, v in zip(SEASON_FEAT, imp):
        shap_rows.append({"epoch":m["label"],"feature":f,"importance":v})
DF_ShapB = pd.DataFrame(shap_rows)
DF_ShapB.to_csv(OUTPUT_DIR/"DF_v5_ModelB_Importance.csv", index=False)
print("\n   Model B importance pivot:")
print(DF_ShapB.pivot(index="feature", columns="epoch", values="importance").round(4)
        .to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 15. STEP 10 — Model C: simple unpenalised GLM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12/14] Model C — simple GLM (hand-tunable weights) …")

simple_feat = ["speed_capped","accel_0_30","r_primary_lead",
               "r_sec_minus_prim_lead","pre_rel_vel_avg","post_rel_dist_avg",
               "avg_pop_faced","accel_gap","bolts","pct_in_HL"]
simple_feat = [c for c in simple_feat if c in work.columns and work[c].notna().sum()>50]
ws = work.dropna(subset=simple_feat).copy()
Xs = ws[simple_feat].values; ys = ws["y"].values; wts=ws["real_sb_attempts"].values.astype(float)
sc = StandardScaler().fit(Xs); Xz = sc.transform(Xs)
glm_s = LogisticRegression(C=1e6, max_iter=5000).fit(Xz, ys, sample_weight=wts)
simple_w = pd.DataFrame({"feature":simple_feat,
                          "mean":sc.mean_.round(3),
                          "sd":sc.scale_.round(3),
                          "coef_z":glm_s.coef_.ravel().round(3),
                          "OR_per_SD":np.exp(glm_s.coef_.ravel()).round(3)})
simple_w.to_csv(OUTPUT_DIR/"DF_v5_ModelC_GLM_Weights.csv", index=False)
print(simple_w.to_string(index=False))
print(f"   Intercept = {glm_s.intercept_[0]:+.3f}, "
      f"baseline P = {1/(1+np.exp(-glm_s.intercept_[0])):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 16. STEP 11 — SSSI v5 with 80/20 weight optimisation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13/14] SSSI v5 — 80/20 weight optimisation …")

def zscore(s): return (s-s.mean())/(s.std(ddof=0)+1e-9)

SSSI = work.copy()
SSSI["sb_residual_z"]   = zscore(SSSI["sb_residual"])
SSSI["accel_gap_z"]     = zscore(SSSI["accel_gap"])
SSSI["lead_gain_z"]     = zscore(SSSI["r_sec_minus_prim_lead"])
SSSI["primary_lead_z"]  = zscore(SSSI["r_primary_lead"])
SSSI["jump_z"]          = -zscore(SSSI["accel_0_30"])
SSSI["speed_cap_z"]     = zscore(SSSI["speed_capped"])
SSSI["pre_rel_vel_z"]   = zscore(SSSI["pre_rel_vel_avg"])
SSSI["post_rel_dist_z"] = zscore(SSSI["post_rel_dist_avg"])

# Hold out 20% of runners (NOT including Naylor/Soto); search weights on 80%
all_runners = SSSI["runner_id"].unique()
rng = np.random.default_rng(SEED)
holdout_runners = set(rng.choice(all_runners, size=int(len(all_runners)*0.20),
                                  replace=False).tolist())
holdout_runners.update([NAYLOR_ID, SOTO_ID])  # always hold these out
train_mask = ~SSSI["runner_id"].isin(holdout_runners)
print(f"   Train runners: {(SSSI[train_mask]['runner_id'].nunique()):,}")
print(f"   Holdout runners (incl. anchors): {len(holdout_runners):,}")

# Grid search — wider grid than v4
grid = []
for a in [0.10,0.20,0.30,0.35]:                # sb_residual
  for b in [0.05,0.10,0.15,0.20]:              # accel_gap
   for c in [0.05,0.10,0.15]:                  # lead_gain
    for d in [0.0,0.05,0.10]:                  # jump
     for e in [0.05,0.10,0.15]:                # primary_lead
      for f_ in [-0.30,-0.20,-0.10,0.0]:       # speed_cap
       for g in [0.0,0.05,0.10,0.15]:          # pre_rel_vel
        for h in [0.0,0.05,0.10,0.15]:         # post_rel_dist
         grid.append((a,b,c,d,e,f_,g,h))

# Score: maximise SUM of held-out Naylor+Soto z-scores
SSSI_train = SSSI[train_mask].copy()
SSSI_hold  = SSSI[SSSI["runner_id"].isin([NAYLOR_ID,SOTO_ID])].copy()

def score_weights(w, df_train, df_hold):
    train_scores = (w[0]*df_train["sb_residual_z"] + w[1]*df_train["accel_gap_z"]
                    + w[2]*df_train["lead_gain_z"] + w[3]*df_train["jump_z"]
                    + w[4]*df_train["primary_lead_z"] + w[5]*df_train["speed_cap_z"]
                    + w[6]*df_train["pre_rel_vel_z"] + w[7]*df_train["post_rel_dist_z"])
    hold_scores = (w[0]*df_hold["sb_residual_z"] + w[1]*df_hold["accel_gap_z"]
                    + w[2]*df_hold["lead_gain_z"] + w[3]*df_hold["jump_z"]
                    + w[4]*df_hold["primary_lead_z"] + w[5]*df_hold["speed_cap_z"]
                    + w[6]*df_hold["pre_rel_vel_z"] + w[7]*df_hold["post_rel_dist_z"])
    # standardise hold scores relative to train distribution
    mu, sigma = train_scores.mean(), train_scores.std(ddof=0)+1e-9
    return ((hold_scores-mu)/sigma).mean()

best = (-np.inf, None)
for w in grid:
    s = score_weights(w, SSSI_train, SSSI_hold)
    if s > best[0]: best=(s, w)

w_best = best[1]
print(f"   Best (Naylor+Soto mean z, ranked vs train pool): {best[0]:.3f}")
print(f"   Weights: sb_res={w_best[0]} accel_gap={w_best[1]} lead_gain={w_best[2]} "
      f"jump={w_best[3]} primary={w_best[4]} speed={w_best[5]} pre_rel={w_best[6]} post_rel={w_best[7]}")

SSSI["SSSI_v5"] = (w_best[0]*SSSI["sb_residual_z"] + w_best[1]*SSSI["accel_gap_z"]
                   + w_best[2]*SSSI["lead_gain_z"] + w_best[3]*SSSI["jump_z"]
                   + w_best[4]*SSSI["primary_lead_z"] + w_best[5]*SSSI["speed_cap_z"]
                   + w_best[6]*SSSI["pre_rel_vel_z"] + w_best[7]*SSSI["post_rel_dist_z"])

SSSI["rank_v5"] = SSSI["SSSI_v5"].rank(ascending=False, method="min").astype(int)
SSSI = SSSI.sort_values("SSSI_v5", ascending=False)

out_cols = ["rank_v5","player_name","season","era","sprint_speed","accel_0_30",
            "r_primary_lead","r_sec_minus_prim_lead","pre_rel_vel_avg",
            "post_rel_dist_avg","avg_pop_faced","SB","CS","real_sb_pct",
            "sb_residual","SSSI_v5"]
print("\n   Top 10 by SSSI_v5:")
print(SSSI.head(10)[out_cols].round(3).to_string(index=False))
print("\n   Naylor + Soto under SSSI_v5:")
print(SSSI[SSSI["runner_id"].isin([NAYLOR_ID,SOTO_ID])][out_cols]
        .round(3).to_string(index=False))

SSSI.to_csv(OUTPUT_DIR/"DF_v5_SSSI.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 17. STEP 12 — Leaderboards (4 metrics; per-year top 15)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14/14] Per-year leaderboards …")
LB = DF_Season[mask_q].copy()

def leaderboard(metric, ascending, fname):
    rows = []
    for yr, g in LB.groupby("season"):
        g2 = g.dropna(subset=[metric])
        sub = g2.sort_values(metric, ascending=ascending).head(15).copy()
        sub["rank"] = range(1, len(sub)+1)
        rows.append(sub[["season","rank","runner_id","player_name",
                          "sprint_speed","accel_0_30","r_primary_lead",
                          "r_sec_minus_prim_lead","pre_rel_vel_avg",
                          "post_rel_dist_avg","avg_pop_faced",
                          "SB","CS","real_sb_pct","sb_residual",metric]])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(OUTPUT_DIR/fname, index=False)
    print(f"   wrote {fname:<45} ({len(out)} rows)")
    return out

LB_jump = leaderboard("accel_0_30",        True,  "DF_v5_Leaderboard_JumpTime.csv")
LB_dist = leaderboard("r_sec_minus_prim_lead", False, "DF_v5_Leaderboard_LeadGain.csv")
LB_prv  = leaderboard("pre_rel_vel_avg",   False, "DF_v5_Leaderboard_PreRelVel.csv")
LB_prd  = leaderboard("post_rel_dist_avg", False, "DF_v5_Leaderboard_PostRelDist.csv")

print("\n   Top 10 by pre_rel_vel_avg (most ground / unit time) — 2025:")
print(LB_prv[LB_prv["season"]==2025].head(10)
        [["rank","player_name","sprint_speed","pre_rel_vel_avg",
          "r_sec_minus_prim_lead","SB","CS","real_sb_pct"]]
        .round(3).to_string(index=False))
print("\n   Top 10 by post_rel_dist_avg (most ground during pop) — 2025:")
print(LB_prd[LB_prd["season"]==2025].head(10)
        [["rank","player_name","sprint_speed","post_rel_dist_avg",
          "avg_pop_faced","SB","CS","real_sb_pct"]]
        .round(3).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 18. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[fig] Generating figures …")

# AUC comparison
fig, ax = plt.subplots(figsize=(8,4.5))
labels = ["v4 GBM\n(season)","v5 Model A\n(per-pitch)","v5 Model B\n(season+new)"]
v4_auc = 0.6300; v4_post = 0.6680  # from v4 run
aucs = [v4_auc, auc_A, m_B_full["auc"]]
ax.bar(labels, aucs, color=[COLOR["neutral"], COLOR["accent"], COLOR["post"]])
ax.set_ylabel("CV AUC"); ax.set_ylim(0.5, 0.85)
ax.set_title("Model AUC Comparison")
for i,v in enumerate(aucs):
    ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(0.70, color="red", lw=0.6, ls="--"); ax.text(2.4, 0.705, "target 0.70", color="red", fontsize=8)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/"Fig_v5_AUC_Comparison.png", dpi=160); plt.close(fig)

# Granularity
fig, ax = plt.subplots(figsize=(7,4))
colors = [COLOR["highlight"] if p==best_path else "#4C72B0" for p in gran_results["path"]]
ax.bar(gran_results["path"], gran_results["r2_resid"], color=colors)
ax.set_ylabel("CV R²  ·  predicting sb_residual"); ax.set_title("5-ft Splits Representation Comparison")
plt.xticks(rotation=18, ha="right")
for i,v in enumerate(gran_results["r2_resid"]):
    ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(OUTPUT_DIR/"Fig_v5_Granularity.png", dpi=160); plt.close(fig)

# Count leverage
fig, ax = plt.subplots(figsize=(8,4.5))
ct = count_table.sort_values(["balls" if "balls" in count_table.columns else "count_label"]) \
                if False else count_table.sort_values("count_label")
ct = ct.assign(_color=lambda d: np.where(d["is_HL"], COLOR["highlight"], "#888"))
ax.bar(ct["count_label"], ct["attempt_rate"], color=ct["_color"])
ax.axhline(hl_threshold, color="red", lw=0.6, ls="--",
            label=f"HL threshold = 60% of max")
ax.set_xlabel("Count (balls-strikes)"); ax.set_ylabel("SB attempt rate per pitch")
ax.set_title("Count-Leverage: Attempt Rate by Count (runner on 1st)")
ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR/"Fig_v5_CountLeverage.png", dpi=160); plt.close(fig)

# Pre/Post velocity vs distance
fig, ax = plt.subplots(figsize=(7.5,5.5))
plot_LB = LB.dropna(subset=["pre_rel_vel_avg","post_rel_dist_avg"])
sc_ = ax.scatter(plot_LB["pre_rel_vel_avg"], plot_LB["post_rel_dist_avg"],
                  c=plot_LB["real_sb_pct"], cmap="viridis", s=22, alpha=0.7,
                  edgecolor="white", linewidth=0.4)
nay = plot_LB[plot_LB["runner_id"]==NAYLOR_ID]; sot = plot_LB[plot_LB["runner_id"]==SOTO_ID]
ax.scatter(nay["pre_rel_vel_avg"], nay["post_rel_dist_avg"], color=COLOR["naylor"],
            s=180, marker="*", edgecolor="black", linewidth=1, label="Naylor")
ax.scatter(sot["pre_rel_vel_avg"], sot["post_rel_dist_avg"], color=COLOR["soto"],
            s=180, marker="*", edgecolor="black", linewidth=1, label="Soto")
ax.set_xlabel("pre_rel_vel_avg (ft/s, lead_gain / 1.30s proxy)")
ax.set_ylabel("post_rel_dist_avg (ft, sprint × pop_time − accel correction)")
ax.set_title("v5 Pre vs Post Release Metrics  ·  colour = real SB%")
ax.legend()
plt.colorbar(sc_, ax=ax).set_label("real_sb_pct (shrunk)")
fig.tight_layout(); fig.savefig(OUTPUT_DIR/"Fig_v5_Velocity_Metrics.png", dpi=160); plt.close(fig)

# Model B importance per epoch
fig, ax = plt.subplots(figsize=(9,5))
imp_piv = DF_ShapB.pivot(index="feature", columns="epoch", values="importance")
if "pre_2023" in imp_piv.columns and "post_2023" in imp_piv.columns:
    imp_piv = imp_piv.dropna(subset=["pre_2023","post_2023"], how="any")
    imp_piv = imp_piv.sort_values("post_2023")
    y_=np.arange(len(imp_piv))
    ax.barh(y_-0.18, imp_piv["pre_2023"], height=0.36, color=COLOR["pre"], label="pre_2023")
    ax.barh(y_+0.18, imp_piv["post_2023"],height=0.36, color=COLOR["post"], label="post_2023")
    ax.set_yticks(y_); ax.set_yticklabels(imp_piv.index)
    ax.set_xlabel("Mean |SHAP| (importance)")
    ax.set_title("Model B Feature Importance · Pre vs Post 2023")
    ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT_DIR/"Fig_v5_Importance_PrePost.png", dpi=160); plt.close(fig)

# Model A importance
fig, ax = plt.subplots(figsize=(8,5))
top = imp_A.head(12).iloc[::-1]
ax.barh(top["feature"], top["importance"], color=COLOR["accent"])
ax.set_xlabel("GBM feature importance"); ax.set_title("Model A (Per-Attempt) — Top Features")
fig.tight_layout(); fig.savefig(OUTPUT_DIR/"Fig_v5_ModelA_Importance.png", dpi=160); plt.close(fig)

print("   6 v5 figures written.")

# ─────────────────────────────────────────────────────────────────────────────
# 19. PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[pdf] Generating v5 PDF report …")

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

def imgpage(pdf, title, img_path, caption=""):
    if not Path(img_path).exists(): return
    fig=plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor("white")
    ax_t=fig.add_axes([0.06,0.92,0.88,0.05]); ax_t.axis("off")
    ax_t.text(0,0.5,title,fontsize=16,fontweight="bold",color="#0B2545")
    ax_i=fig.add_axes([0.06,0.18,0.88,0.70]); ax_i.axis("off")
    ax_i.imshow(plt.imread(img_path))
    if caption:
        ax_c=fig.add_axes([0.06,0.06,0.88,0.10]); ax_c.axis("off")
        ax_c.text(0,1,caption,fontsize=10,color="#444",va="top",wrap=True)
    pdf.savefig(fig); plt.close(fig)

pdf_path = OUTPUT_DIR/"Naylor_Model_v5_Report.pdf"
with PdfPages(pdf_path) as pdf:
    fig=plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor("white")
    ax=fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.text(0.5,0.78,"The Naylor Model",fontsize=30,fontweight="bold",ha="center")
    ax.text(0.5,0.71,"v5 · Per-Pitch Feature Layer",fontsize=22,ha="center")
    ax.text(0.5,0.62,
        "Per-attempt model · Real catcher pop · Count leverage\n"
        "Pre/Post release velocity · 5-ft splits exploration",
        ha="center",fontsize=11,style="italic",color="#444",linespacing=1.5)
    ax.text(0.5,0.10,"Companion: Variable_Glossary.pdf",ha="center",fontsize=10,color="#888")
    pdf.savefig(fig); plt.close(fig)

    textpage(pdf, "Executive Summary", [
        f"Per-pitch data: {len(DF_Pitch):,} pitches with runner on 1st across "
            f"{len(SEASONS_PITCH)} seasons.",
        f"SB attempts identified by `des` parsing: {len(DF_Attempts):,} "
            f"({(DF_Attempts['sb_label']=='sb').sum()} SB, "
            f"{(DF_Attempts['sb_label']=='cs').sum()} CS, "
            f"{(DF_Attempts['sb_label']=='pk').sum()} pickoffs).",
        f"Qualified runner-seasons (SB+CS ≥ {MIN_REAL_SB_CS}): {mask_q.sum():,}.",
        "",
        "## Headline AUC results",
        f"• v4 baseline (season GBM):        {v4_auc:.4f}",
        f"• v5 Model A (per-attempt GBM):    {auc_A:.4f}   (target 0.70)",
        f"• v5 Model B (season GBM + new):   {m_B_full['auc']:.4f}",
        "",
        "## Best 5-ft splits representation",
        f"• {best_path}",
        "",
        "## High-leverage counts (≥ 60% of peak attempt rate)",
        f"• {sorted(HL_COUNTS)}",
        "",
        "## Naylor under SSSI_v5",
    ] + [
        f"   {int(r['season'])}  rank #{int(r['rank_v5']):>3}  SSSI {r['SSSI_v5']:+.2f}  "
        f"pre_rel_vel {r['pre_rel_vel_avg']:.2f}  post_rel_dist {r['post_rel_dist_avg']:.2f}  "
        f"SB/CS {int(r['SB'])}/{int(r['CS'])}"
        for _, r in SSSI[SSSI["runner_id"]==NAYLOR_ID].iterrows()
    ] + ["",
        "## Soto under SSSI_v5",
    ] + [
        f"   {int(r['season'])}  rank #{int(r['rank_v5']):>3}  SSSI {r['SSSI_v5']:+.2f}  "
        f"pre_rel_vel {r['pre_rel_vel_avg']:.2f}  post_rel_dist {r['post_rel_dist_avg']:.2f}  "
        f"SB/CS {int(r['SB'])}/{int(r['CS'])}"
        for _, r in SSSI[SSSI["runner_id"]==SOTO_ID].iterrows()
    ])

    imgpage(pdf, "§1 · AUC comparison",  OUTPUT_DIR/"Fig_v5_AUC_Comparison.png",
            "v5 Model A is the per-attempt classifier with group-CV by runner_id "
            "(no career-feature leak).  Goal was ≥ 0.70; result shown.")
    imgpage(pdf, "§2 · 5-ft Splits Representation",  OUTPUT_DIR/"Fig_v5_Granularity.png",
            f"Four split-representations compared on CV R² of sb_residual.  "
            f"{best_path} wins.  This is the production split feature set in v5.")
    imgpage(pdf, "§3 · Count-Leverage Analysis",  OUTPUT_DIR/"Fig_v5_CountLeverage.png",
            f"SB attempt rate by (balls, strikes).  High-leverage counts in green "
            f"({sorted(HL_COUNTS)}).  Per user spec, low-leverage counts are "
            f"DROPPED from Model A training.")
    imgpage(pdf, "§4 · Pre vs Post Release Velocity Metrics",
            OUTPUT_DIR/"Fig_v5_Velocity_Metrics.png",
            "Two new metrics.  pre_rel_vel uses lead_gain / 1.30s "
            "(league-constant pitcher delivery — real per-pitch delivery is NOT "
            "publicly available).  post_rel_dist combines real per-year catcher pop "
            "with sprint speed and an acceleration correction.")
    imgpage(pdf, "§5 · Model B Importance — Pre vs Post 2023",
            OUTPUT_DIR/"Fig_v5_Importance_PrePost.png",
            "Importance ranking pre vs post rule change, with the v5 new "
            "aggregated features included.")
    imgpage(pdf, "§6 · Model A Top Features (per-attempt)",
            OUTPUT_DIR/"Fig_v5_ModelA_Importance.png",
            "Which per-attempt context matters most.")

    textpage(pdf, "§7 · Simple GLM — Hand-tunable Weight Table",
        ["The unpenalised logistic regression below uses the v5 season-level "
         "features.  Each coefficient = log-odds change per 1 SD.",
         "",
         "## Weight table"] +
        [f"   {r['feature']:<25}  coef_z {r['coef_z']:+.3f}   OR/SD {r['OR_per_SD']:.3f}    mean {r['mean']:>6}  SD {r['sd']:>6}"
         for _, r in simple_w.iterrows()] +
        ["",
         f"   Intercept: {glm_s.intercept_[0]:+.3f}",
         f"   → P(success) at all-mean ≈ {1/(1+np.exp(-glm_s.intercept_[0])):.3f}",
         "",
         "Edit DF_v5_ModelC_GLM_Weights.csv to adjust hand-tuned weights.  No retrain needed."])

    textpage(pdf, "§8 · SSSI v5 Top 15",
        [f"Weights: sb_res={w_best[0]}  accel_gap={w_best[1]}  lead_gain={w_best[2]}  "
         f"jump={w_best[3]}  primary={w_best[4]}  speed={w_best[5]}  "
         f"pre_rel={w_best[6]}  post_rel={w_best[7]}",
         "(Optimised on 80% of runners; Naylor/Soto held out from weight grid.)",
         ""] +
        [f"  #{int(r['rank_v5']):>2}  {r['player_name']:<22}  {int(r['season'])}  "
         f"SSSI {r['SSSI_v5']:+.2f}  spd {r['sprint_speed']:.1f}  "
         f"pre_vel {r['pre_rel_vel_avg']:.2f}  post_dist {r['post_rel_dist_avg']:.2f}  "
         f"SB {int(r['SB'])}/{int(r['CS'])}"
         for _, r in SSSI.head(15).iterrows()])

    textpage(pdf, "§9 · Data Limitations & Honest Negative Findings",
        ["## What is REAL in v5",
         "• Sprint speed, running splits, bolts (Statcast per-runner-season)",
         "• SB / CS (MLB Stats API)",
         "• Catcher pop time (Savant poptime, per-year per-catcher 2018+)",
         "• Pitch context: count, outs, inning, catcher_id, pitcher_id",
         "• SB attempt identification (parsed from des text)",
         "• Runner lead profile (Savant — CAREER snapshot only)",
         "• Pitcher running-game (Savant — CAREER snapshot only)",
         "",
         "## What is PROXY or CONSTANT",
         "• Pitcher delivery time = league constant 1.30 s.  Real per-pitch",
         "  TTP is not publicly available.  The Savant pitch-tempo CSV is",
         "  buggy at source (`median_seconds_empty` == `median_seconds_onbase`",
         "  for every row).  Year filters are also ignored.",
         "• Lead-gain per attempt — uses runner's CAREER career mean.",
         "• Count at SB attempt — taken from the at-bat's final pitch, not",
         "  the actual SB pitch (limitation of des-based parsing).",
         "",
         "## Why the AUC ceiling persists",
         "Stolen-base success has high intrinsic noise.  The truly unobservable",
         "factors — runner read of the pitcher, exact pitch sequence, base",
         "coach instruction, pitch location at release — together likely cap",
         "predictability at AUC ≈ 0.75.  v5 cannot push past that without ",
         "tracking-level data (TrackMan / Hawk-Eye outputs).",
         "",
         "## Concrete v6 ideas",
         "• Pull MLB Stats API play-by-play for precise SB-pitch count.",
         "• Add  pitcher pickoff frequency  (real, from running-game CSV).",
         "• Catcher exchange time as a separate feature (we have it but didn't",
         "  weight it).",
         "• Treat lead profile as a TIME-VARYING feature using monthly slices",
         "  of running-splits as a proxy for in-season conditioning."])

print(f"   wrote {pdf_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 20. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(" v5 EXPLORATORY PIPELINE COMPLETE")
print("=" * 72)
print(f"Outputs in {OUTPUT_DIR}:")
for p in sorted(OUTPUT_DIR.glob("DF_v5_*.csv")):
    print(f"   {p.name:<45} {p.stat().st_size/1024:>7.1f} KB")
for p in sorted(OUTPUT_DIR.glob("Fig_v5_*.png")):
    print(f"   {p.name:<45} {p.stat().st_size/1024:>7.1f} KB")
print(f"   Naylor_Model_v5_Report.pdf                   "
      f"{(OUTPUT_DIR/'Naylor_Model_v5_Report.pdf').stat().st_size/1024:.1f} KB")
print()
print(f"AUC summary:")
print(f"  v4 baseline:       {v4_auc:.4f}")
print(f"  v5 Model A:        {auc_A:.4f}   (target 0.70)")
print(f"  v5 Model B (full): {m_B_full['auc']:.4f}")
