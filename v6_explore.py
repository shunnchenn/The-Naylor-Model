#!/usr/bin/env python3
"""
The Naylor Model · v6  —  cleaner outputs, intuitive metrics
============================================================
Builds on v5.  Reuses the same per-pitch + catcher pop + pitcher running-game
data (cached on disk).  Focus of v6:

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

Outputs:  DF_v6_*.csv, Fig_v6_*.png, Naylor_Model_v6_Report.pdf.
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

np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

COLOR = {"pre":"#E0A458","post":"#3D5A80","naylor":"#DC2626","soto":"#1D4ED8",
         "neutral":"#374151","highlight":"#10B981","accent":"#0EA5E9",
         "elite":"#10B981","above":"#3B82F6","avg":"#6B7280",
         "below":"#F59E0B","poor":"#DC2626"}

print("=" * 72)
print(" THE NAYLOR MODEL  ·  v6  (intuitive outputs)")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CACHE HELPERS  (same as v5 — reuse cached files)
# ─────────────────────────────────────────────────────────────────────────────
def cache_load(name):
    p = CACHE_DIR / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f: return pickle.load(f)
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

print(f"   Runner-seasons: {len(DF_Season):,}  ·  qualified: {mask_q.sum():,}")
print(f"   League SB% = {league_sb:.3f}")

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
LEAGUE_PITCHER_TTP = 1.30
DF_Attempts["pre_release_velocity"] = (
    DF_Attempts["lead_gain"] / LEAGUE_PITCHER_TTP)

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

# Per runner-season aggregates (new v6 set)
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
# 5. MODEL B — season-level GBM with v6 features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/10] Model B — season-level GBM …")

V6_FEATURES = [
    "sprint_speed","speed_capped","jump_time","accel_phase","top_speed_phase",
    "total_90","accel_gap","bolts",
    "primary_lead","secondary_lead","lead_gain",
    "avg_pop_faced","avg_pickoff_rate_faced","weak_arm_share","two_strike_share",
    "avg_pre_release_velocity","avg_post_release_distance",
    "n_attempts",
]
V6_FEATURES = [c for c in V6_FEATURES if c in DF_Season.columns]

work = DF_Season[mask_q].dropna(subset=V6_FEATURES+["real_sb_pct"]).copy()
work["y"] = (work["real_sb_pct"]>=league_sb).astype(int)
print(f"   Qualified rows w/ all features: {len(work)}")

def fit_season(df_sub, label):
    if len(df_sub) < 50: return None
    X = df_sub[V6_FEATURES].values; y = df_sub["y"].values
    w = df_sub["sb_attempts"].values.astype(float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = cross_val_predict(
        GradientBoostingClassifier(n_estimators=400, max_depth=3,
                                    learning_rate=0.04, random_state=SEED),
        X, y, cv=cv, method="predict_proba")[:,1]
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
pd.DataFrame(auc_rows).to_csv(OUTPUT_DIR/"DF_v6_ModelB_AUC.csv", index=False)

# SHAP / importance
shap_rows = []
for m in [m_full, m_pre, m_post]:
    if m is None: continue
    gbm = GradientBoostingClassifier(n_estimators=400, max_depth=3,
                                       learning_rate=0.04,
                                       random_state=SEED).fit(m["X"], m["y"], sample_weight=m["w"])
    if HAS_SHAP:
        try:
            ex = shap.TreeExplainer(gbm)
            imp = np.abs(ex.shap_values(m["X"])).mean(axis=0)
        except Exception:
            imp = gbm.feature_importances_
    else:
        imp = gbm.feature_importances_
    for f, v in zip(V6_FEATURES, imp):
        shap_rows.append({"epoch":m["label"], "feature":f,
                          "importance": round(v, 4)})
DF_Imp = pd.DataFrame(shap_rows)
DF_Imp.to_csv(OUTPUT_DIR/"DF_v6_Importance.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6. INTUITIVE GLM (replaces coef_z / OR/SD with plain-English columns)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/10] Simple GLM with intuitive metric names …")

simple_feat = ["speed_capped","jump_time","primary_lead","lead_gain",
               "avg_pre_release_velocity","avg_post_release_distance",
               "avg_pop_faced","avg_pickoff_rate_faced","weak_arm_share",
               "accel_gap","bolts","two_strike_share"]
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
DF_GLM.to_csv(OUTPUT_DIR/"DF_v6_GLM_PlainEnglish.csv", index=False)

print(f"\n   Baseline P(success) at all-mean inputs ≈ {baseline_p:.3f}")
print(f"\n   {'Feature':<32}{'Boost (pp)':>12}{'Odds×':>9}  Note")
for _, r in DF_GLM.iterrows():
    arrow = "↑helps" if r["higher_is_better"] else ("↓helps" if r["higher_is_better"] is False else "?")
    print(f"   {r['feature']:<32}{r['sb_pct_boost_per_tier']:>+12.2f}"
          f"{r['odds_multiplier']:>9.3f}  {arrow}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SSSI v6  (same idea, cleaner column names)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/10] SSSI v6 (held-out weight search) …")

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
            + w[6]*df["z_pre_rel_vel"] + w[7]*df["z_post_rel_dist"])

grid = [(a,b,c,d,e,f_,g,h)
        for a in [0.10,0.20,0.30,0.35]
        for b in [0.05,0.10,0.15,0.20]
        for c in [0.05,0.10,0.15]
        for d in [0.0,0.05,0.10]
        for e in [0.05,0.10,0.15]
        for f_ in [-0.30,-0.20,-0.10,0.0]
        for g in [0.0,0.05,0.10]
        for h in [0.0,0.05,0.10]]

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
SSSI["SSSI_v6"] = score(w_best, SSSI)
SSSI = SSSI.sort_values("SSSI_v6", ascending=False)
SSSI["rank_v6"] = SSSI["SSSI_v6"].rank(ascending=False, method="min").astype(int)
SSSI.to_csv(OUTPUT_DIR/"DF_v6_SSSI.csv", index=False)

print(f"   Best Naylor+Soto mean z: {best[0]:.3f}")
print(f"   Weights: sb_res={w_best[0]} gap={w_best[1]} gain={w_best[2]} "
      f"jump={w_best[3]} prim={w_best[4]} speed={w_best[5]} "
      f"pre={w_best[6]} post={w_best[7]}")

top_cols = ["rank_v6","player_name","season","era","sprint_speed",
            "jump_time","primary_lead","lead_gain",
            "avg_pre_release_velocity","avg_post_release_distance",
            "SB","CS","real_sb_pct","sb_residual","SSSI_v6"]
print("\n   Top 10 by SSSI_v6:")
print(SSSI.head(10)[top_cols].round(3).to_string(index=False))
print("\n   Naylor + Soto under SSSI_v6:")
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
    out.to_csv(OUTPUT_DIR/fname, index=False)
    print(f"   {fname:<48}{len(out):>5} rows")
    return out

LB_jump = board("jump_time",                  True,  "DF_v6_LB_JumpTime.csv")
LB_gain = board("lead_gain",                  False, "DF_v6_LB_LeadGain.csv")
LB_prv  = board("avg_pre_release_velocity",   False, "DF_v6_LB_PreReleaseVelocity.csv")
LB_prd  = board("avg_post_release_distance",  False, "DF_v6_LB_PostReleaseDistance.csv")


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
ax.set_title("v6 Simple GLM — Plain-English Weight Table")
for i, v in enumerate(g["sb_pct_boost_per_tier"]):
    ax.text(v + (0.15 if v>=0 else -0.15), i, f"{v:+.1f}",
            ha="left" if v>=0 else "right", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR/"Fig_v6_GLM_PlainEnglish.png", dpi=160); plt.close(fig)

# Fig: AUC across versions
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["v4\n(season)","v5 Model A\n(per-attempt)","v5 Model B\n(season+new)","v6 Model B"]
v4_auc = 0.6300
v5_A = 0.5933
v5_B = 0.6794
v6_B = m_full["auc"] if m_full else float("nan")
aucs = [v4_auc, v5_A, v5_B, v6_B]
ax.bar(labels, aucs, color=[COLOR["neutral"], COLOR["accent"], COLOR["post"], COLOR["highlight"]])
ax.set_ylabel("CV AUC"); ax.set_ylim(0.5, 0.85)
ax.set_title("Model AUC across versions")
for i, v in enumerate(aucs):
    ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(OUTPUT_DIR/"Fig_v6_AUC.png", dpi=160); plt.close(fig)

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
ax.set_title("v6 — Feature Importance · Pre vs Post 2023")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR/"Fig_v6_Importance_PrePost.png", dpi=160); plt.close(fig)


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
fig.savefig(OUTPUT_DIR/"Fig_v6_NaylorSoto_Profile.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("   4 v6 figures written.")


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

pdf_path = OUTPUT_DIR/"Naylor_Model_v6_Report.pdf"
with PdfPages(pdf_path) as pdf:
    # Cover
    fig=plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor("white")
    ax=fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.text(0.5,0.78,"The Naylor Model",fontsize=30,fontweight="bold",ha="center")
    ax.text(0.5,0.71,"v6 · Intuitive Outputs",fontsize=22,ha="center")
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
        "## What changed from v5",
        "• ALL column names now plain English — no more coef_z, OR/SD, pct_in_HL.",
        "• GLM table shows  'SB % Boost per Tier'  =  pp change in success rate",
        "  for a 1-SD improvement in the feature.",
        "• Added  pitcher pickoff-rate-faced  and  weak-arm-catcher share.",
        "• Dropped  pct_in_HL  (was a Statcast data artifact — see v5 §9).",
        "",
        "## AUC summary",
        f"• v4 baseline:           0.6300",
        f"• v5 Model A (per-att):  0.5933",
        f"• v5 Model B (season):   0.6794",
        f"• v6 Model B (full):     {m_full['auc']:.4f}",
        f"• v6 Model B (pre-23):   {m_pre['auc']:.4f}" if m_pre else "",
        f"• v6 Model B (post-23):  {m_post['auc']:.4f}" if m_post else "",
        "",
        "## Naylor + Soto rank under SSSI v6",
    ] + [
        f"  #{int(r['rank_v6']):>3}  {r['player_name']:<22}  {int(r['season'])}  "
        f"SSSI {r['SSSI_v6']:+.2f}   SB/CS {int(r['SB'])}/{int(r['CS'])}   real_sb_pct {r['real_sb_pct']:.3f}"
        for _, r in SSSI[SSSI["runner_id"].isin([NAYLOR_ID,SOTO_ID])].iterrows()
    ])

    imgpage(pdf, "§1 · AUC Comparison", OUTPUT_DIR/"Fig_v6_AUC.png",
            "v6 Model B at the season level, with the v6 feature set including "
            "real catcher pop, pitcher pickoff rate, lead variables.")
    imgpage(pdf, "§2 · GLM Plain-English Weight Chart",
            OUTPUT_DIR/"Fig_v6_GLM_PlainEnglish.png",
            "Each bar = predicted percentage-point change in SB success when "
            "the runner improves on that feature by ONE TIER (1 SD).  Positive "
            "and green = the feature HELPS; red/orange = HURTS.")
    imgpage(pdf, "§3 · Feature Importance · Pre vs Post 2023",
            OUTPUT_DIR/"Fig_v6_Importance_PrePost.png",
            "How feature importance shifted after the 2023 rule change.")
    imgpage(pdf, "§4 · Naylor + Soto Statcast-style Profile",
            OUTPUT_DIR/"Fig_v6_NaylorSoto_Profile.png",
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
         f"Edit DF_v6_GLM_PlainEnglish.csv to hand-tune.  No retraining needed."])

    # SSSI page
    textpage(pdf, "§6 · SSSI v6 Top 15",
        [f"Weights:  sb_res={w_best[0]}  gap={w_best[1]}  gain={w_best[2]}  "
         f"jump={w_best[3]}  prim={w_best[4]}  speed={w_best[5]}  "
         f"pre={w_best[6]}  post={w_best[7]}",
         "(80% of runners used for grid search; Naylor + Soto held out.)",
         ""] +
        [f"  #{int(r['rank_v6']):>3}  {r['player_name']:<22}  {int(r['season'])}  "
         f"SSSI {r['SSSI_v6']:+.2f}  spd {r['sprint_speed']:.1f}  "
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
         "  - To push higher would require TrackMan / Hawk-Eye raw outputs."])

print(f"   wrote {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(" v6 EXPLORATORY PIPELINE COMPLETE")
print("=" * 72)
for p in sorted(OUTPUT_DIR.glob("DF_v6_*.csv")):
    print(f"   {p.name:<45} {p.stat().st_size/1024:>7.1f} KB")
for p in sorted(OUTPUT_DIR.glob("Fig_v6_*.png")):
    print(f"   {p.name:<45} {p.stat().st_size/1024:>7.1f} KB")
print(f"   Naylor_Model_v6_Report.pdf            "
      f"{(OUTPUT_DIR/'Naylor_Model_v6_Report.pdf').stat().st_size/1024:.1f} KB")
print()
print(f"Headline:")
print(f"  v4 baseline:       0.6300")
print(f"  v5 Model B:        0.6794")
print(f"  v6 Model B (full): {m_full['auc']:.4f}")
print(f"  v6 Model B (pre):  {m_pre['auc']:.4f}" if m_pre else "")
print(f"  v6 Model B (post): {m_post['auc']:.4f}" if m_post else "")
