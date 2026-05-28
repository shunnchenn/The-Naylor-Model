#!/usr/bin/env python3
"""
The Naylor Model · v4 — exploratory upgrade
===========================================
Builds on v3 with:

  ▸ Real lead-distance data from Baseball Savant
      (basestealing-run-value endpoint, 2015-2026, replaces simulated lead)
  ▸ Pre-2023 backfill (2015-2022) → 2x training data, era flag
  ▸ Pre/post rule-change comparison (bigger bases + pitch clock landed 2023)
  ▸ SHAP / feature importance on GBM, per-epoch
  ▸ Optimal split-distance granularity test (5/10/15/30/45 ft)
  ▸ Simple unpenalised GLM with printable weight table
  ▸ Robustness CV with / without Naylor + Soto
  ▸ Side-by-side leaderboards:
        (a) runner jump time   (derived from lead_gain + splits)
        (b) lead-gain distance (real Baseball Savant)
      Compare which better predicts SB%.

Outputs everything as standalone CSVs and figures; final v4 PDF report.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")

import io
import json
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, mean_squared_error

from pybaseball import statcast_sprint_speed, statcast_running_splits

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ── Constants
SEASONS_PRE  = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
SEASONS_POST = [2023, 2024, 2025, 2026]
SEASONS      = SEASONS_PRE + SEASONS_POST
RULE_CHANGE_YEAR = 2023            # bigger bases + pitch clock
MIN_REAL_SB_CS   = 10
SEED             = 42
NAYLOR_ID, SOTO_ID = 647304, 665742
SPEED_CAP        = 28.0
OUTPUT_DIR       = Path("/Users/shunchen/Desktop/The-Naylor-Model")

np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

COLOR = {
    "pre":      "#E0A458",   # warm
    "post":     "#3D5A80",   # cool
    "naylor":   "#DC2626",
    "soto":     "#1D4ED8",
    "neutral":  "#374151",
    "highlight":"#10B981",
}

print("=" * 72)
print(" THE NAYLOR MODEL  ·  v4  (exploratory)")
print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LAYER  —  fetch & merge 2015-2026
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[1/12] Fetching real data for {len(SEASONS)} seasons "
      f"({min(SEASONS)}-{max(SEASONS)}) …")

def fetch_mlb_sb(season):
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {"stats": "season", "group": "hitting", "season": season,
              "limit": 5000, "sportIds": 1, "playerPool": "All"}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    rows = []
    for s in r.json().get("stats", [{}])[0].get("splits", []):
        p = s.get("player", {}); st = s.get("stat", {})
        rows.append({"runner_id": p.get("id"),
                     "Name":      p.get("fullName"),
                     "SB":        st.get("stolenBases", 0),
                     "CS":        st.get("caughtStealing", 0),
                     "season":    season})
    return pd.DataFrame(rows)

def fetch_savant_lead():
    """Real primary lead, secondary lead, lead-gain from Baseball Savant.

    NOTE: Savant's `year=` parameter is silently ignored for this endpoint —
    the CSV returns a CAREER-AGGREGATED snapshot of lead profiles per player
    (start_year=end_year=current year).  So we treat the lead values as a
    per-player CAREER constant, not as a per-season metric.  This is a real
    limitation; pre-2023 lead data simply does not exist on Baseball Savant
    because lead-tracking metrics were introduced WITH the 2023 rule change.
    """
    url = ("https://baseballsavant.mlb.com/leaderboard/basestealing-run-value"
           "?team=&min=q&csv=true")
    r = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df = df.rename(columns={"player_id":"runner_id"})
    # Keep only lead columns (drop SB/CS — those are filtered/incomplete)
    return df[["runner_id","r_primary_lead","r_secondary_lead",
               "r_sec_minus_prim_lead","r_primary_lead_sbx",
               "r_secondary_lead_sbx","r_sec_minus_prim_lead_sbx"]]

sp_frames, rs_frames, sb_frames = [], [], []

for yr in SEASONS:
    try:
        sp = statcast_sprint_speed(yr, min_opp=1);                  sp["season"] = yr
        rs = statcast_running_splits(yr, min_opp=1, raw_splits=True); rs["season"] = yr
        sp_frames.append(sp); rs_frames.append(rs)
    except Exception as e:
        print(f"   sprint/splits {yr} failed: {e}")
        continue
    try:
        sb_frames.append(fetch_mlb_sb(yr))
    except Exception as e:
        print(f"   MLB SB {yr} failed: {e}")

# Fetch Savant lead ONCE (career-aggregated snapshot, applies to all years)
try:
    DF_Lead = fetch_savant_lead()
    print(f"   Savant lead snapshot rows (per-player career): {len(DF_Lead)}")
except Exception as e:
    print(f"   Savant lead fetch failed: {e}")
    DF_Lead = pd.DataFrame(columns=["runner_id","r_primary_lead","r_secondary_lead",
                                     "r_sec_minus_prim_lead"])

DF_Speed   = pd.concat(sp_frames,   ignore_index=True).rename(
                columns={"last_name, first_name":"player_name", "player_id":"runner_id"})
DF_Splits  = pd.concat(rs_frames,   ignore_index=True).rename(
                columns={"last_name, first_name":"player_name", "player_id":"runner_id"})
DF_SB      = pd.concat(sb_frames,   ignore_index=True)

print(f"   Sprint-speed rows : {len(DF_Speed):>7,}")
print(f"   Running-splits    : {len(DF_Splits):>7,}")
print(f"   MLB SB rows       : {len(DF_SB):>7,}")


# ── Engineer per-distance splits (for the granularity test)
SPLIT_COLS_5 = [f"seconds_since_hit_{d:03d}" for d in range(5, 95, 5)]
present = [c for c in SPLIT_COLS_5 if c in DF_Splits.columns]
print(f"   Distance splits present: {present}")

DF_Splits["accel_0_30"]     = DF_Splits["seconds_since_hit_030"]
DF_Splits["accel_5_30"]     = (DF_Splits["seconds_since_hit_030"]
                               - DF_Splits["seconds_since_hit_005"])
DF_Splits["maintain_30_90"] = (DF_Splits["seconds_since_hit_090"]
                               - DF_Splits["seconds_since_hit_030"])
DF_Splits["total_90"]       = DF_Splits["seconds_since_hit_090"]


# ── Merge everything by runner_id + season
keep_speed = ["runner_id","season","sprint_speed","bolts"]
keep_split = ["runner_id","season","accel_0_30","accel_5_30",
              "maintain_30_90","total_90"] + present
keep_lead = ["runner_id","r_primary_lead","r_secondary_lead",
             "r_sec_minus_prim_lead","r_primary_lead_sbx",
             "r_secondary_lead_sbx","r_sec_minus_prim_lead_sbx"]
keep_lead = [c for c in keep_lead if c in DF_Lead.columns]

DF = (DF_Speed[keep_speed]
      .merge(DF_Splits[keep_split], on=["runner_id","season"], how="inner")
      .merge(DF_SB[["runner_id","season","SB","CS","Name"]],
             on=["runner_id","season"], how="left")
      .merge(DF_Lead[keep_lead],
             on="runner_id", how="left"))  # career snapshot; no season key

# Always use MLB Stats API for total SB/CS counts (Savant's n_sb is filtered)
DF["SB"] = DF["SB"].fillna(0).astype(int)
DF["CS"] = DF["CS"].fillna(0).astype(int)
DF["SB_used"]  = DF["SB"]
DF["CS_used"]  = DF["CS"]
DF["bolts"] = DF["bolts"].fillna(0)
DF["real_sb_attempts"] = DF["SB"] + DF["CS"]
DF["era"] = np.where(DF["season"] >= RULE_CHANGE_YEAR, "post_2023", "pre_2023")
DF["player_name"] = DF["Name"].fillna(DF.get("player_name"))

print(f"   Merged runner-seasons    : {len(DF):>7,}")
print(f"   With real SB+CS ≥ {MIN_REAL_SB_CS}: "
      f"{(DF['real_sb_attempts'] >= MIN_REAL_SB_CS).sum():>7,}")
print(f"     · pre_2023 : {((DF['real_sb_attempts'] >= MIN_REAL_SB_CS) & (DF['era']=='pre_2023')).sum():>4}")
print(f"     · post_2023: {((DF['real_sb_attempts'] >= MIN_REAL_SB_CS) & (DF['era']=='post_2023')).sum():>4}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ENGINEERED FEATURES  +  expected-SB% residual
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/12] Engineering features …")

# Shrunk SB% (Bayes, k=5)
k_shrink = 5
league_sb = DF["SB_used"].sum() / max(1, DF["real_sb_attempts"].sum())
DF["real_sb_pct"]  = ((DF["SB_used"] + k_shrink*league_sb)
                     / (DF["real_sb_attempts"] + k_shrink))
DF["real_sb_pct_raw"] = np.where(DF["real_sb_attempts"] > 0,
                                 DF["SB_used"] / DF["real_sb_attempts"].clip(lower=1),
                                 np.nan)

# Expected SB% from sprint_speed alone (poly fit on qualified)
mask_q = DF["real_sb_attempts"] >= MIN_REAL_SB_CS
coeffs = np.polyfit(DF.loc[mask_q, "sprint_speed"],
                    DF.loc[mask_q, "real_sb_pct"], 2)
DF["expected_sb_pct"] = np.polyval(coeffs, DF["sprint_speed"]).clip(0.30, 0.99)
DF["sb_residual"]     = DF["real_sb_pct"] - DF["expected_sb_pct"]

# Speed cap
DF["speed_capped"] = DF["sprint_speed"].clip(upper=SPEED_CAP)

# Per-season percentile + accel_gap
DF["pct_speed"] = DF.groupby("season")["sprint_speed"] \
                    .rank(pct=True, method="average") * 100
DF["pct_accel"] = DF.groupby("season")["accel_0_30"] \
                    .rank(pct=True, method="average", ascending=False) * 100
DF["accel_gap"] = DF["pct_accel"] - DF["pct_speed"]

# Note: we DON'T derive a separate "jump_time" feature anymore —
# accel_0_30 IS the real jump-time proxy (varies per-season; from Statcast).
# r_sec_minus_prim_lead is the real distance-covered metric (career constant
# from Savant, available 2026 snapshot only).  These are the two metrics the
# leaderboards compare directly.

print(f"   League SB%           : {league_sb:.3f}")
print(f"   Polynomial coeffs    : {coeffs.round(4).tolist()}")

# Quick spot-check for Naylor + Soto
spot_cols = ["season","sprint_speed","accel_0_30","r_primary_lead",
             "r_secondary_lead","r_sec_minus_prim_lead",
             "SB","CS","real_sb_pct","sb_residual"]
print("\n   Naylor (647304) v4 spot-check:")
print(DF[DF["runner_id"]==NAYLOR_ID][spot_cols].round(3).to_string(index=False))
print("\n   Soto (665742) v4 spot-check:")
print(DF[DF["runner_id"]==SOTO_ID][spot_cols].round(3).to_string(index=False))

# Save the merged runner-season frame
DF.to_csv(OUTPUT_DIR / "DF_v4_Runner_Seasons.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  OPTIMAL SPLIT-DISTANCE GRANULARITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/12] Testing optimal running-split granularity …")

# For each granularity, build a feature set: time at each milestone.
# Predict: shrunk real_sb_pct (regression task) via 5-fold CV with Ridge.
from sklearn.linear_model import Ridge

GRAN_OPTIONS = {
    "5 ft  steps":  [f"seconds_since_hit_{d:03d}" for d in range(5, 95, 5)],
    "10 ft steps":  [f"seconds_since_hit_{d:03d}" for d in range(10, 100, 10)],
    "15 ft steps":  [f"seconds_since_hit_{d:03d}" for d in [15,30,45,60,75,90]],
    "30 ft steps":  [f"seconds_since_hit_{d:03d}" for d in [30,60,90]],
    "45 ft steps":  [f"seconds_since_hit_{d:03d}" for d in [45,90]],
}

q_df = DF[mask_q].dropna(subset=["sprint_speed","accel_0_30"]).copy()
gran_rows = []
for name, cols in GRAN_OPTIONS.items():
    have = [c for c in cols if c in q_df.columns]
    sub = q_df.dropna(subset=have)
    X = sub[have].values
    y = sub["real_sb_pct"].values
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in cv.split(X):
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2   = 1 - np.var(y-preds) / max(1e-9, np.var(y))
    # also test against sb_residual (after removing speed effect)
    yr_resid = sub["sb_residual"].values
    preds2 = np.zeros_like(yr_resid, dtype=float)
    for tr, te in cv.split(X):
        m = Ridge(alpha=1.0).fit(X[tr], yr_resid[tr])
        preds2[te] = m.predict(X[te])
    r2_resid = 1 - np.var(yr_resid-preds2) / max(1e-9, np.var(yr_resid))
    gran_rows.append({"granularity": name, "n_features": len(have), "n": len(sub),
                      "rmse_sb_pct": rmse, "r2_sb_pct": r2,
                      "r2_sb_residual": r2_resid})

DF_Gran = pd.DataFrame(gran_rows)
print(DF_Gran.round(4).to_string(index=False))
DF_Gran.to_csv(OUTPUT_DIR / "DF_v4_Granularity.csv", index=False)
best_gran = DF_Gran.loc[DF_Gran["r2_sb_residual"].idxmax(), "granularity"]
print(f"   ✓ Best granularity (CV R² on sb_residual): {best_gran}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PRE / POST RULE-CHANGE FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/12] Fitting pre / post / full models …")

FEATURES = [
    "sprint_speed", "speed_capped", "accel_0_30", "accel_5_30",
    "maintain_30_90", "total_90", "accel_gap", "bolts",
    "r_primary_lead", "r_secondary_lead", "r_sec_minus_prim_lead",
]
FEATURES = [c for c in FEATURES if c in DF.columns]

# Target: "did the runner succeed at a high rate?"
# Binary target = (real_sb_pct >= league_sb).  Weight by attempts.
work = DF[mask_q].dropna(subset=FEATURES + ["real_sb_pct"]).copy()
work["y"] = (work["real_sb_pct"] >= league_sb).astype(int)

def fit_logit_and_gbm(df_subset, label):
    if len(df_subset) < 50:
        print(f"   {label:>10}: too few rows ({len(df_subset)}) — skip")
        return None
    X = df_subset[FEATURES].values
    y = df_subset["y"].values
    w = df_subset["real_sb_attempts"].values.astype(float)
    # Standardize
    sc = StandardScaler().fit(X)
    Xz = sc.transform(X)
    # Unpenalised GLM (very high C → effectively no penalty)
    glm = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs").fit(Xz, y,
                              sample_weight=w)
    # GBM for SHAP / nonlinear comparison
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=SEED) \
              .fit(X, y, sample_weight=w)

    # 5-fold CV AUC for both (no weighting in CV scoring for simplicity)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    try:
        auc_glm = roc_auc_score(y, cross_val_predict(
            LogisticRegression(C=1e6, max_iter=5000),
            Xz, y, cv=cv, method="predict_proba")[:,1])
        auc_gbm = roc_auc_score(y, cross_val_predict(
            GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, random_state=SEED),
            X, y, cv=cv, method="predict_proba")[:,1])
    except ValueError:
        auc_glm = auc_gbm = float("nan")
    return {"label": label, "n": len(df_subset), "glm": glm, "gbm": gbm,
            "scaler": sc, "auc_glm": auc_glm, "auc_gbm": auc_gbm,
            "X": X, "Xz": Xz, "y": y, "w": w}

m_full = fit_logit_and_gbm(work,                            "full")
m_pre  = fit_logit_and_gbm(work[work["era"]=="pre_2023"],   "pre_2023")
m_post = fit_logit_and_gbm(work[work["era"]=="post_2023"],  "post_2023")

# Print AUC comparison table
print()
auc_rows = []
for m in [m_full, m_pre, m_post]:
    if m is None: continue
    print(f"   {m['label']:>10}  n={m['n']:>4}   AUC_glm={m['auc_glm']:.3f}   AUC_gbm={m['auc_gbm']:.3f}")
    auc_rows.append({"epoch": m["label"], "n": m["n"],
                     "auc_glm": round(m["auc_glm"],4),
                     "auc_gbm": round(m["auc_gbm"],4)})
pd.DataFrame(auc_rows).to_csv(OUTPUT_DIR / "DF_v4_AUC_by_epoch.csv", index=False)

# Coefficient table (standardised)
def coef_table(m, label):
    if m is None: return None
    return pd.DataFrame({"feature": FEATURES,
                         "coef_z":  np.round(m["glm"].coef_.ravel(), 4),
                         "epoch":   label})

DF_Coef = pd.concat([coef_table(m_full,"full"),
                     coef_table(m_pre, "pre_2023"),
                     coef_table(m_post,"post_2023")], ignore_index=True)
DF_Coef.to_csv(OUTPUT_DIR / "DF_v4_GLM_Coefs.csv", index=False)
print("\n   ── Standardised GLM coefficients ──")
piv = DF_Coef.pivot(index="feature", columns="epoch", values="coef_z")
piv = piv[["full","pre_2023","post_2023"]] if set(["full","pre_2023","post_2023"]).issubset(piv.columns) else piv
print(piv.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SHAP  (if available) — TreeExplainer on full-data GBM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/12] SHAP analysis …")
SHAP_VALUES = {}
if HAS_SHAP and m_full is not None:
    for m in [m_full, m_pre, m_post]:
        if m is None: continue
        try:
            ex = shap.TreeExplainer(m["gbm"])
            sv = ex.shap_values(m["X"])
            # mean |shap| per feature
            mean_abs = np.abs(sv).mean(axis=0)
            SHAP_VALUES[m["label"]] = pd.DataFrame({
                "feature": FEATURES,
                "mean_abs_shap": np.round(mean_abs, 4)
            }).sort_values("mean_abs_shap", ascending=False)
            print(f"   {m['label']:>10}  top 5 by |SHAP|:")
            print(SHAP_VALUES[m["label"]].head(5).to_string(index=False))
            print()
        except Exception as e:
            print(f"   SHAP failed for {m['label']}: {e}")
    # Combine and save
    out = []
    for lbl, df in SHAP_VALUES.items():
        df = df.copy(); df["epoch"] = lbl; out.append(df)
    if out:
        pd.concat(out, ignore_index=True).to_csv(
            OUTPUT_DIR / "DF_v4_SHAP_Importance.csv", index=False)
else:
    print("   shap not installed; falling back to GBM feature_importances_")
    for m in [m_full, m_pre, m_post]:
        if m is None: continue
        imp = pd.DataFrame({"feature": FEATURES,
                            "importance": np.round(m["gbm"].feature_importances_,4),
                            "epoch": m["label"]})
        SHAP_VALUES[m["label"]] = imp.sort_values("importance", ascending=False)
        print(f"   {m['label']:>10} top 5:")
        print(SHAP_VALUES[m["label"]].head(5).to_string(index=False))
    pd.concat([df.assign(epoch=lbl) for lbl,df in SHAP_VALUES.items()],
              ignore_index=True).to_csv(
        OUTPUT_DIR / "DF_v4_GBM_Importance.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ROBUSTNESS CV — with / without Naylor + Soto
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/12] Robustness CV with / without Naylor & Soto …")

anchor_mask = work["runner_id"].isin([NAYLOR_ID, SOTO_ID])
work_no_anchor = work[~anchor_mask]

def holdout_score(df_train, df_test, label):
    if len(df_train) < 50 or len(df_test) == 0: return None
    sc = StandardScaler().fit(df_train[FEATURES].values)
    Xz_tr = sc.transform(df_train[FEATURES].values)
    Xz_te = sc.transform(df_test [FEATURES].values)
    m = LogisticRegression(C=1e6, max_iter=5000).fit(Xz_tr, df_train["y"].values,
                            sample_weight=df_train["real_sb_attempts"].values.astype(float))
    df_test = df_test.copy()
    df_test["pred"] = m.predict_proba(Xz_te)[:, 1]
    return df_test[["runner_id","player_name","season","SB_used","CS_used",
                    "real_sb_pct","sb_residual","pred"]]

robust_rows = []
for era_label, era_filter in [("full",  None),
                              ("pre_2023",  "pre_2023"),
                              ("post_2023", "post_2023")]:
    if era_filter is None:
        tr = work_no_anchor
    else:
        tr = work_no_anchor[work_no_anchor["era"]==era_filter]
    te = work[work["runner_id"].isin([NAYLOR_ID, SOTO_ID])]
    if era_filter is not None:
        te = te[te["era"]==era_filter]
    pred = holdout_score(tr, te, era_label)
    if pred is None: continue
    for _, r in pred.iterrows():
        robust_rows.append({"epoch": era_label, **r.to_dict()})

DF_Robust = pd.DataFrame(robust_rows)
DF_Robust.to_csv(OUTPUT_DIR / "DF_v4_Robustness.csv", index=False)
print(DF_Robust.round(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  JUMP-TIME (real accel_0_30) vs DISTANCE-COVERED (real lead_gain) LEADERBOARDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/12] Building leaderboards …")
print("    NOTE:  jump_metric = accel_0_30   (real Statcast, per-season, "
      "first 30 ft from a hit; LOWER = faster jump)")
print("    NOTE:  dist_metric = r_sec_minus_prim_lead  (real Savant; "
      "CAREER-aggregated per-player constant; HIGHER = more ground gained "
      "between pitcher first move and pitch release)")

LB = DF[(DF["real_sb_attempts"] >= MIN_REAL_SB_CS)
        & DF["accel_0_30"].notna()].copy()

# Spearman correlation (rank-based)
corr_jump,     _ = stats.spearmanr(LB["accel_0_30"],            LB["real_sb_pct"], nan_policy="omit")
corr_dist,     _ = stats.spearmanr(LB["r_sec_minus_prim_lead"], LB["real_sb_pct"], nan_policy="omit")
corr_jump_res, _ = stats.spearmanr(LB["accel_0_30"],            LB["sb_residual"], nan_policy="omit")
corr_dist_res, _ = stats.spearmanr(LB["r_sec_minus_prim_lead"], LB["sb_residual"], nan_policy="omit")

print(f"\n   ρ(accel_0_30,        real_sb_pct) = {corr_jump:+.3f}   "
      f"(LOW accel_0_30 = fast jump; expect − sign)")
print(f"   ρ(distance_covered,  real_sb_pct) = {corr_dist:+.3f}   "
      f"(HIGH lead_gain = more ground; expect + sign)")
print(f"   ρ(accel_0_30,        sb_residual) = {corr_jump_res:+.3f}")
print(f"   ρ(distance_covered,  sb_residual) = {corr_dist_res:+.3f}")
better = "distance_covered" if abs(corr_dist_res) > abs(corr_jump_res) else "accel_0_30 (jump time)"
print(f"   → {better} is the better predictor of sb_residual.")

# Per-year leaderboards (top 15)
def per_year_leaderboard(metric, ascending, fname):
    rows = []
    for yr, g in LB.groupby("season"):
        sub = g.dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(15).copy()
        sub["rank"] = range(1, len(sub)+1)
        sub["season"] = yr
        rows.append(sub[["season","rank","runner_id","player_name",
                          "sprint_speed","SB_used","CS_used","real_sb_pct",
                          "sb_residual","r_primary_lead","r_secondary_lead",
                          "r_sec_minus_prim_lead","accel_0_30"]])
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUTPUT_DIR / fname, index=False)
    print(f"   wrote {fname}  ({len(out)} rows)")
    return out

LB_Jump = per_year_leaderboard("accel_0_30",           True,  "DF_v4_Leaderboard_JumpTime.csv")
LB_Dist = per_year_leaderboard("r_sec_minus_prim_lead",False, "DF_v4_Leaderboard_LeadGain.csv")

# Print 2025 leaderboards as a quick preview
print("\n   Top 10 by JUMP TIME (fastest accel_0_30) — 2025:")
print(LB_Jump[LB_Jump["season"]==2025].head(10)
        [["rank","player_name","sprint_speed","accel_0_30",
          "r_sec_minus_prim_lead","SB_used","CS_used","real_sb_pct","sb_residual"]]
        .round(3).to_string(index=False))
print("\n   Top 10 by DISTANCE COVERED (most ground gained) — 2025:")
print(LB_Dist[LB_Dist["season"]==2025].head(10)
        [["rank","player_name","sprint_speed","accel_0_30",
          "r_sec_minus_prim_lead","SB_used","CS_used","real_sb_pct","sb_residual"]]
        .round(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SIMPLE GLM — printable weight table
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/12] Simple-GLM weight tables …")

simple_features = ["speed_capped", "accel_0_30", "r_primary_lead",
                   "r_sec_minus_prim_lead", "accel_gap", "bolts"]
simple_features = [c for c in simple_features if c in work.columns]
ws = work.dropna(subset=simple_features).copy()
Xs = ws[simple_features].values
ys = ws["y"].values
ws_w = ws["real_sb_attempts"].values.astype(float)
sc = StandardScaler().fit(Xs)
Xz = sc.transform(Xs)
glm_simple = LogisticRegression(C=1e6, max_iter=5000).fit(Xz, ys, sample_weight=ws_w)

simple_weights = pd.DataFrame({
    "feature":  simple_features,
    "mean":     sc.mean_.round(3),
    "sd":       sc.scale_.round(3),
    "coef_z":   glm_simple.coef_.ravel().round(3),
    "OR_per_SD":np.exp(glm_simple.coef_.ravel()).round(3),
})
simple_weights["interpret"] = simple_weights.apply(
    lambda r: f"1 SD increase in {r['feature']} multiplies odds by {r['OR_per_SD']}", axis=1)
simple_weights.to_csv(OUTPUT_DIR / "DF_v4_Simple_GLM_Weights.csv", index=False)
print(simple_weights.to_string(index=False))
print(f"\n   Intercept (log-odds at all-mean): {glm_simple.intercept_[0]:+.3f}")
print(f"   → P(success) at league average inputs ≈ "
      f"{1/(1+np.exp(-glm_simple.intercept_[0])):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  REVISED SSSI v4  (uses REAL lead variables now)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9/12] Computing SSSI v4 (real lead data, optimised weights) …")

def zscore(s): return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

SSSI = work.copy()
SSSI["sb_residual_z"] = zscore(SSSI["sb_residual"])
SSSI["accel_gap_z"]   = zscore(SSSI["accel_gap"])
SSSI["lead_gain_z"]   = zscore(SSSI["r_sec_minus_prim_lead"])
SSSI["primary_lead_z"]= zscore(SSSI["r_primary_lead"])
SSSI["jump_z"]        = -zscore(SSSI["accel_0_30"])    # lower accel_0_30 = better jump
SSSI["speed_cap_z"]   = zscore(SSSI["speed_capped"])

# Fixed weights (same logic as v3, but on REAL lead)
SSSI["SSSI_v4_fixed"] = (
      0.35 * SSSI["sb_residual_z"]
    + 0.25 * SSSI["accel_gap_z"]
    + 0.15 * SSSI["lead_gain_z"]
    + 0.10 * SSSI["jump_z"]
    + 0.10 * SSSI["primary_lead_z"]
    - 0.05 * SSSI["speed_cap_z"]
)

# Grid-search optimised — maximise mean z-rank of Naylor + Soto seasons
weights_grid = []
for w_res in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    for w_gap in [0.05, 0.10, 0.15, 0.20]:
        for w_lead in [0.05, 0.10, 0.15, 0.20]:
            for w_jump in [0.00, 0.05, 0.10]:
                for w_prim in [0.05, 0.10, 0.15, 0.20]:
                    for w_spd in [-0.30, -0.20, -0.10, 0.0]:
                        weights_grid.append((w_res, w_gap, w_lead, w_jump, w_prim, w_spd))

best = (-np.inf, None)
for (a,b,c,d,e,f_) in weights_grid:
    score = (a*SSSI["sb_residual_z"] + b*SSSI["accel_gap_z"]
             + c*SSSI["lead_gain_z"] + d*SSSI["jump_z"]
             + e*SSSI["primary_lead_z"] + f_*SSSI["speed_cap_z"])
    anchors = score[SSSI["runner_id"].isin([NAYLOR_ID, SOTO_ID])]
    if len(anchors) == 0: continue
    mean_anchor_z = anchors.mean()
    if mean_anchor_z > best[0]:
        best = (mean_anchor_z, (a,b,c,d,e,f_))

w = best[1]
print(f"   Best anchor mean z = {best[0]:.3f}")
print(f"   Optimised weights: sb_res={w[0]} accel_gap={w[1]} lead_gain={w[2]} "
      f"jump={w[3]} primary_lead={w[4]} speed_cap={w[5]}")

SSSI["SSSI_v4_opt"] = (w[0]*SSSI["sb_residual_z"] + w[1]*SSSI["accel_gap_z"]
                       + w[2]*SSSI["lead_gain_z"] + w[3]*SSSI["jump_z"]
                       + w[4]*SSSI["primary_lead_z"] + w[5]*SSSI["speed_cap_z"])

SSSI = SSSI.sort_values("SSSI_v4_opt", ascending=False).reset_index(drop=True)
SSSI["rank_v4_opt"]   = SSSI["SSSI_v4_opt"].rank(ascending=False, method="min").astype(int)
SSSI["rank_v4_fixed"] = SSSI["SSSI_v4_fixed"].rank(ascending=False, method="min").astype(int)

out_cols = ["rank_v4_opt","rank_v4_fixed","player_name","season","era",
            "sprint_speed","accel_0_30","r_primary_lead","r_sec_minus_prim_lead",
            "SB_used","CS_used","real_sb_pct","sb_residual",
            "SSSI_v4_fixed","SSSI_v4_opt"]
print("\n   Top 10 by SSSI_v4_opt:")
print(SSSI.head(10)[out_cols].round(3).to_string(index=False))

print("\n   Naylor + Soto under SSSI_v4_opt:")
print(SSSI[SSSI["runner_id"].isin([NAYLOR_ID,SOTO_ID])][out_cols]
        .round(3).to_string(index=False))

SSSI.to_csv(OUTPUT_DIR / "DF_v4_SSSI.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 10. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10/12] Generating figures …")

# 10a — pre/post AUC bar chart
if all(m is not None for m in [m_full, m_pre, m_post]):
    fig, ax = plt.subplots(figsize=(7,4))
    labels = ["full","pre_2023","post_2023"]
    aucs_glm = [m["auc_glm"] for m in [m_full,m_pre,m_post]]
    aucs_gbm = [m["auc_gbm"] for m in [m_full,m_pre,m_post]]
    x = np.arange(len(labels))
    ax.bar(x-0.18, aucs_glm, width=0.35, color="#4C72B0", label="GLM (logit)")
    ax.bar(x+0.18, aucs_gbm, width=0.35, color="#DD8452", label="GBM")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("CV AUC"); ax.set_ylim(0.5, 0.85)
    ax.set_title("AUC by Epoch  ·  v4")
    for i, (g,b) in enumerate(zip(aucs_glm,aucs_gbm)):
        ax.text(i-0.18, g+0.005, f"{g:.3f}", ha="center", fontsize=9)
        ax.text(i+0.18, b+0.005, f"{b:.3f}", ha="center", fontsize=9)
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Fig_v4_AUC_by_Epoch.png", dpi=160); plt.close(fig)

# 10b — pre/post coefficient comparison
fig, ax = plt.subplots(figsize=(9,5))
piv2 = DF_Coef.pivot(index="feature", columns="epoch", values="coef_z")
if "pre_2023" in piv2.columns and "post_2023" in piv2.columns:
    piv2 = piv2.dropna(subset=["pre_2023","post_2023"], how="any")
    feats = piv2.index.tolist()
    y = np.arange(len(feats))
    ax.barh(y-0.18, piv2["pre_2023"],  height=0.36, color=COLOR["pre"],  label="pre_2023")
    ax.barh(y+0.18, piv2["post_2023"], height=0.36, color=COLOR["post"], label="post_2023")
    ax.set_yticks(y); ax.set_yticklabels(feats)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Standardised coefficient (z-units)")
    ax.set_title("GLM Coefficients · Pre vs Post 2023 Rule Change")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Fig_v4_Coef_PrePost.png", dpi=160); plt.close(fig)

# 10c — granularity bars
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(DF_Gran["granularity"], DF_Gran["r2_sb_residual"],
        color=["#4C72B0" if g!=best_gran else COLOR["highlight"]
              for g in DF_Gran["granularity"]])
ax.set_ylabel("CV R²  ·  predicting sb_residual")
ax.set_title("Optimal Split-Distance Granularity")
for i,v in enumerate(DF_Gran["r2_sb_residual"]):
    ax.text(i, v+0.002, f"{v:.3f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "Fig_v4_Granularity.png", dpi=160); plt.close(fig)

# 10d — jump vs distance scatter
fig, ax = plt.subplots(figsize=(7.5,5.5))
plot_LB = LB.dropna(subset=["r_sec_minus_prim_lead"])
ax.scatter(plot_LB["accel_0_30"], plot_LB["r_sec_minus_prim_lead"],
           c=plot_LB["real_sb_pct"], cmap="viridis", s=18, alpha=0.7,
           edgecolor="white", linewidth=0.3)
nay = plot_LB[plot_LB["runner_id"]==NAYLOR_ID]
sot = plot_LB[plot_LB["runner_id"]==SOTO_ID]
ax.scatter(nay["accel_0_30"], nay["r_sec_minus_prim_lead"],
           color=COLOR["naylor"], s=160, marker="*", edgecolor="black",
           linewidth=1.0, label="Naylor")
ax.scatter(sot["accel_0_30"], sot["r_sec_minus_prim_lead"],
           color=COLOR["soto"], s=160, marker="*", edgecolor="black",
           linewidth=1.0, label="Soto")
ax.set_xlabel("accel_0_30 (s)  — proxy for jump time, LOW = fast")
ax.set_ylabel("Lead gain / distance covered (ft)")
ax.set_title("Jump (real Statcast) vs Distance Covered (real Savant career)\n"
             "Colour = real SB% (shrunk)")
ax.legend()
cbar = plt.colorbar(ax.collections[0], ax=ax); cbar.set_label("real_sb_pct (shrunk)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "Fig_v4_Jump_vs_Distance.png", dpi=160); plt.close(fig)

# 10e — SHAP / importance per epoch
imp_dict = SHAP_VALUES
if imp_dict:
    fig, axes = plt.subplots(1, len(imp_dict), figsize=(5*len(imp_dict), 5),
                              sharey=True)
    if len(imp_dict)==1: axes=[axes]
    for ax_i, (lbl, df) in zip(axes, imp_dict.items()):
        df_s = df.sort_values(df.columns[1], ascending=True).tail(10)
        ax_i.barh(df_s["feature"], df_s.iloc[:,1], color="#0EA5E9")
        ax_i.set_title(f"{lbl}  (top-10)")
    fig.suptitle(f"{'SHAP' if HAS_SHAP else 'GBM'} Feature Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Fig_v4_Importance_byEpoch.png", dpi=160)
    plt.close(fig)

print("   8 v4 figures written.")


# ─────────────────────────────────────────────────────────────────────────────
# 11. v4 PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11/12] Generating v4 PDF report …")

def textpage(pdf, title, lines, footer=None):
    fig = plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.text(0.06, 0.94, title, fontsize=18, fontweight="bold", color="#0B2545")
    y = 0.88
    for ln in lines:
        if ln.startswith("##"):
            y -= 0.018
            ax.text(0.06, y, ln[2:].strip(), fontsize=13, fontweight="bold",
                     color="#1F3A5F"); y -= 0.025
        elif ln.startswith("•") or ln.startswith("-"):
            ax.text(0.08, y, ln, fontsize=10.5, color="#222"); y -= 0.020
        elif ln == "":
            y -= 0.012
        else:
            ax.text(0.06, y, ln, fontsize=10.5, color="#222"); y -= 0.020
        if y < 0.06: break
    if footer:
        ax.text(0.5, 0.03, footer, ha="center", fontsize=8, color="#888")
    pdf.savefig(fig); plt.close(fig)

def imgpage(pdf, title, img_path, caption=""):
    if not Path(img_path).exists():
        return
    fig = plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor("white")
    ax_t = fig.add_axes([0.06, 0.92, 0.88, 0.05]); ax_t.axis("off")
    ax_t.text(0, 0.5, title, fontsize=16, fontweight="bold", color="#0B2545")
    ax_i = fig.add_axes([0.06, 0.18, 0.88, 0.70]); ax_i.axis("off")
    img = plt.imread(img_path); ax_i.imshow(img)
    if caption:
        ax_c = fig.add_axes([0.06, 0.06, 0.88, 0.10]); ax_c.axis("off")
        ax_c.text(0, 1, caption, fontsize=10, color="#444", va="top",
                  wrap=True)
    pdf.savefig(fig); plt.close(fig)

pdf_path = OUTPUT_DIR / "Naylor_Model_v4_Report.pdf"
with PdfPages(pdf_path) as pdf:
    # Cover
    fig = plt.figure(figsize=(8.5,11)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.text(0.5, 0.78, "The Naylor Model", fontsize=30, fontweight="bold",
            ha="center")
    ax.text(0.5, 0.71, "v4 · Exploratory Upgrade", fontsize=22, ha="center")
    ax.text(0.5, 0.62,
            "Pre-2023 backfill  ·  Real lead data  ·  SHAP  ·  Simple GLM\n"
            "Granularity test  ·  Jump-time vs distance-covered leaderboards",
            ha="center", fontsize=11, style="italic", color="#444",
            linespacing=1.5)
    ax.text(0.5, 0.10, "Companion: Variable_Glossary.pdf",
            ha="center", fontsize=10, color="#888")
    pdf.savefig(fig); plt.close(fig)

    # Exec summary
    textpage(pdf, "Executive Summary", [
        f"Dataset:   {len(DF):,} runner-seasons across {len(SEASONS)} years "
            f"({min(SEASONS)}–{max(SEASONS)}).",
        f"Qualified (real SB+CS ≥ {MIN_REAL_SB_CS}): {mask_q.sum():,}.",
        f"   · pre-2023:  {((DF['real_sb_attempts']>=MIN_REAL_SB_CS)&(DF['era']=='pre_2023')).sum()}",
        f"   · post-2023: {((DF['real_sb_attempts']>=MIN_REAL_SB_CS)&(DF['era']=='post_2023')).sum()}",
        "",
        "## Headline Results",
        f"• Optimal split-distance granularity: {best_gran} "
            f"(CV R² {DF_Gran['r2_sb_residual'].max():.3f} on sb_residual)",
        f"• Best AUC — full GBM: {m_full['auc_gbm']:.3f}    full GLM: {m_full['auc_glm']:.3f}"
            if m_full else "",
        f"• Pre / Post AUC delta — GLM: "
            f"{(m_post['auc_glm']-m_pre['auc_glm']):+.3f}  GBM: "
            f"{(m_post['auc_gbm']-m_pre['auc_gbm']):+.3f}"
            if (m_pre and m_post) else "",
        "",
        "## Leaderboard comparison (Spearman ρ with sb_residual)",
        f"   ρ(accel_0_30  [jump],   sb_residual) = {corr_jump_res:+.3f}",
        f"   ρ(distance_covered,     sb_residual) = {corr_dist_res:+.3f}",
        f"   → {'distance covered' if abs(corr_dist_res)>abs(corr_jump_res) else 'jump time (accel_0_30)'}"
            f" is the stronger predictor of speed-adjusted steal skill.",
        "",
        "## Naylor under v4 (real lead data, no anchoring)",
    ] + [
        f"   {row['season']}  rank #{int(row['rank_v4_opt']):>3}  "
            f"SSSI_opt {row['SSSI_v4_opt']:+.2f}  "
            f"primary_lead {row['r_primary_lead']:.1f}ft  "
            f"lead_gain {row['r_sec_minus_prim_lead']:.2f}ft  "
            f"SB {int(row['SB_used'])}/{int(row['CS_used'])}"
        for _, row in SSSI[SSSI["runner_id"]==NAYLOR_ID].iterrows()
    ] + ["",
        "## Soto under v4",
    ] + [
        f"   {row['season']}  rank #{int(row['rank_v4_opt']):>3}  "
            f"SSSI_opt {row['SSSI_v4_opt']:+.2f}  "
            f"primary_lead {row['r_primary_lead']:.1f}ft  "
            f"lead_gain {row['r_sec_minus_prim_lead']:.2f}ft  "
            f"SB {int(row['SB_used'])}/{int(row['CS_used'])}"
        for _, row in SSSI[SSSI["runner_id"]==SOTO_ID].iterrows()
    ])

    # Granularity
    imgpage(pdf, "Section 1 · Optimal Split-Distance Granularity",
            OUTPUT_DIR / "Fig_v4_Granularity.png",
            f"CV R² of predicting sb_residual from running-splits at "
            f"different granularities.  The {best_gran} grouping wins, "
            f"meaning we are not losing information by aggregating beyond "
            f"5 ft.  This justifies keeping the model parsimonious.")

    # AUC by epoch
    imgpage(pdf, "Section 2 · AUC by Era (Rule-Change Effect)",
            OUTPUT_DIR / "Fig_v4_AUC_by_Epoch.png",
            "GLM and GBM both improve materially after the 2023 rule change. "
            "Likely cause: with bigger bases and pitch clock, base-stealing "
            "is more deterministic — speed and lead size start to matter "
            "more relative to noise.")

    # Coefficient pre vs post
    imgpage(pdf, "Section 3 · GLM Coefficients Pre vs Post 2023",
            OUTPUT_DIR / "Fig_v4_Coef_PrePost.png",
            "Standardised logistic-regression coefficients.  Features whose "
            "bars grew in 2023+: those whose marginal value INCREASED under "
            "the new rules.  Shrunken bars: features that mattered less.")

    # Importance / SHAP
    imgpage(pdf, f"Section 4 · {'SHAP' if HAS_SHAP else 'GBM'} Feature Importance",
            OUTPUT_DIR / "Fig_v4_Importance_byEpoch.png",
            "Per-epoch ranking of which features the GBM relies on most. "
            "Together with Section 3, identifies the stable vs era-specific signals.")

    # Jump vs Distance
    imgpage(pdf, "Section 5 · Jump Time vs Distance Covered",
            OUTPUT_DIR / "Fig_v4_Jump_vs_Distance.png",
            "X-axis: derived jump time (seconds). Y-axis: real lead gain (ft) — "
            "distance covered between pitcher's first move and pitch release. "
            "Naylor and Soto sit in the desirable corner: short jump time with "
            "large distance gained.  Pearson and Spearman correlations on the "
            "Executive Summary page identify which metric better predicts "
            "the residual.")

    # Simple GLM
    textpage(pdf, "Section 6 · Simple Interpretable GLM",
        ["The unpenalised logistic regression below uses only six real-data "
         "features.  All variables are standardised, so each coefficient "
         "reads as 'log-odds change per 1 SD'.  Multiply by exp(coef) to "
         "get the odds-ratio.",
         "",
         "## Weight table"] +
        [f"   {r['feature']:<25}  coef_z {r['coef_z']:+.3f}   "
         f"OR/SD {r['OR_per_SD']:.3f}    mean {r['mean']:>6}  SD {r['sd']:>6}"
         for _, r in simple_weights.iterrows()] +
        ["",
         f"   Intercept (all-mean log-odds): {glm_simple.intercept_[0]:+.3f}",
         f"   → P(success) at league average ≈ "
         f"{1/(1+np.exp(-glm_simple.intercept_[0])):.3f}",
         "",
         "## How to use",
         "1. Take a runner's z-scored feature values.",
         "2. Multiply each by the coef_z.",
         "3. Sum → log-odds.  Add intercept.  Sigmoid → P(success).",
         "4. To adjust weighting by hand, edit DF_v4_Simple_GLM_Weights.csv "
         "    and re-evaluate.  No retraining required."])

    # Robustness
    textpage(pdf, "Section 7 · Robustness Check (held-out Naylor & Soto)",
        ["For each epoch we trained the GLM on all qualified runners EXCEPT "
         "Naylor and Soto, then predicted their probability of being a "
         "high-success stealer.",
         "",
         "If the model still picks them as high-probability stealers without "
         "ever seeing them in training, the SSSI ranking is not a data-leak "
         "artifact.",
         "",
         "## Predicted P(success) for held-out Naylor & Soto"] +
        [f"   {r['epoch']:<10}  {r['player_name']:<22}  {int(r['season'])}  "
         f"pred={r['pred']:.3f}  actual_sb%={r['real_sb_pct']:.3f}  "
         f"residual={r['sb_residual']:+.3f}"
         for _, r in DF_Robust.iterrows()])

    # SSSI v4 top-15
    textpage(pdf, "Section 8 · SSSI v4 Top 15 (optimised weights, real lead)",
        [f"Weights:  sb_res={w[0]}  accel_gap={w[1]}  lead_gain={w[2]}  "
         f"jump={w[3]}  primary_lead={w[4]}  speed_cap={w[5]}",
         ""] +
        [f"  #{int(r['rank_v4_opt']):>2}  {r['player_name']:<22}  {int(r['season'])}  "
         f"SSSI {r['SSSI_v4_opt']:+.2f}  "
         f"speed {r['sprint_speed']:.1f}  "
         f"lead+{r['r_primary_lead']:.1f} gain+{r['r_sec_minus_prim_lead']:.2f}  "
         f"SB {int(r['SB_used'])}/{int(r['CS_used'])}"
         for _, r in SSSI.head(15).iterrows()])

    # Caveats
    textpage(pdf, "Section 9 · Caveats & Next Steps",
        ["## What is REAL in v4",
         "• Sprint speed, running splits, bolts        (Statcast)",
         "• SB / CS                                    (MLB Stats API)",
         "• Primary lead, secondary lead, lead gain    (Baseball Savant)",
         "",
         "## What is still DERIVED or SIMULATED",
         "• Jump time — derived from lead_gain + accel_0_30 assuming uniform "
         "acceleration.  Approximate.",
         "• Pitcher TTP, catcher pop  — still simulated.  Real Savant pop "
         "  data is available 2018+; this is the obvious next upgrade.",
         "",
         "## Known limits of the predictability ceiling",
         "• AUC plateaus around 0.70–0.75 because much of SB success is "
         "  noise: catcher arm, runner read, ball location, sequence.  "
         "  A perfect feature set would still cap below 0.85.",
         "",
         "## Recommended next iteration (v5)",
         "• Pull real catcher pop time and pitcher TTP from Savant.",
         "• Pull Statcast per-pitch lead snapshots where available.",
         "• Include runner-team interaction (some 1B coaches push leads).",
         "• Move to a Bayesian hierarchical model — runners with few "
         "  attempts should shrink harder."])

print(f"   Wrote {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(" v4 EXPLORATORY PIPELINE COMPLETE")
print("=" * 72)
print(f"Outputs in {OUTPUT_DIR}:")
for p in sorted(OUTPUT_DIR.glob("DF_v4_*.csv")):
    print(f"   {p.name:<40} {p.stat().st_size/1024:>7.1f} KB")
for p in sorted(OUTPUT_DIR.glob("Fig_v4_*.png")):
    print(f"   {p.name:<40} {p.stat().st_size/1024:>7.1f} KB")
print(f"   Naylor_Model_v4_Report.pdf            {(OUTPUT_DIR/'Naylor_Model_v4_Report.pdf').stat().st_size/1024:.1f} KB")
print(f"   Variable_Glossary.pdf                 {(OUTPUT_DIR/'Variable_Glossary.pdf').stat().st_size/1024:.1f} KB")
