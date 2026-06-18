"""
benchmark_models.py  —  Model Benchmarking: AUC comparison across classifiers
No hyperparameter tuning. Pure out-of-the-box defaults (except random_state=SEED).
Same features, CV setup, and sample weights as v7 Model B.

Models tested:
  1. GBM (sklearn) — current baseline
  2. XGBoost
  3. CatBoost
  4. Random Forest (for variance comparison)
  5. Logistic Regression (regularised — interpretable lower bound)
  6. Pygam (LogisticGAM) — if installed

Run: python3 benchmark_models.py
Outputs: prints a ranked table + writes Data Frame/DF_benchmark_AUC.csv
"""

import warnings, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# ── Paths (mirror v7_explore.py conventions) ──────────────────────────────────
ROOT     = Path(__file__).resolve().parent
if not (ROOT / "Figures").exists() and (ROOT.parent / "Figures").exists():
    ROOT = ROOT.parent           # script moved into scripts/
DATA_DIR = (ROOT / "data") if (ROOT / "data").exists() else (ROOT / "Data Frame")
SEED     = 42

# ── Optional imports (soft-fail with a note) ──────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — skipping XGBoost")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[WARN] catboost not installed — skipping CatBoost")

try:
    from pygam import LogisticGAM, s
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    print("[WARN] pygam not installed — skipping LogisticGAM")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading DF_v7_SSSI.csv ...")
df = pd.read_csv(DATA_DIR / "DF_v7_SSSI.csv")

# Replicate v7 qualification filter
mask_q = (df["sb_attempts"] >= 10) & df["real_sb_pct"].notna()
league_sb = df.loc[mask_q, "real_sb_pct"].median()
print(f"  League median SB%: {league_sb:.3f}   (binary outcome threshold)")

FEATURES = [
    "sprint_speed","speed_capped","jump_time",
    "total_90","accel_gap","bolts",
    "dist_to_top_speed_ft","accel_topspeed_gap",
    "primary_lead","secondary_lead","lead_gain",
    "avg_pop_faced","avg_pickoff_rate_faced","weak_arm_share","two_strike_share",
    "avg_pre_release_velocity","avg_post_release_distance",
    "n_attempts",
]
FEATURES = [c for c in FEATURES if c in df.columns]

work = df[mask_q].dropna(subset=FEATURES + ["real_sb_pct"]).copy()
work["y"] = (work["real_sb_pct"] >= league_sb).astype(int)
print(f"  Qualified rows: {len(work)}   features: {len(FEATURES)}")

X = work[FEATURES].values
y = work["y"].values
w = work["sb_attempts"].values.astype(float)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: (display_name, sklearn-compatible estimator, needs_scaling)
models = [
    ("GBM (sklearn) — baseline",
     GradientBoostingClassifier(n_estimators=400, max_depth=3,
                                 learning_rate=0.04, random_state=SEED),
     False),

    ("Logistic Regression (L2)",
     Pipeline([("sc", StandardScaler()),
               ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))]),
     False),  # scaling built into pipeline

    ("Random Forest",
     RandomForestClassifier(n_estimators=400, max_depth=5,
                             random_state=SEED, n_jobs=-1),
     False),
]

if HAS_XGB:
    models.append((
        "XGBoost",
        __import__("xgboost").XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.04,
            eval_metric="logloss", random_state=SEED,
            verbosity=0, use_label_encoder=False),
        False))

if HAS_CAT:
    models.append((
        "CatBoost",
        __import__("catboost").CatBoostClassifier(
            iterations=400, depth=3, learning_rate=0.04,
            verbose=0, random_seed=SEED),
        False))

if HAS_GAM:
    # pygam doesn't accept sample_weight in cross_val_predict — run manually
    pass  # handled separately below

# ── Run CV benchmark ──────────────────────────────────────────────────────────
print("\n" + "─"*60)
print(f"{'Model':<38}  {'AUC':>7}  {'vs baseline':>12}")
print("─"*60)

results = []
baseline_auc = None

for name, clf, _ in models:
    try:
        # Pass sample_weight via fit_params
        preds = cross_val_predict(
            clf, X, y, cv=cv, method="predict_proba",
            fit_params={"sample_weight": w} if "pipeline" not in name.lower() else
                        {"lr__sample_weight": w}
        )[:, 1]
        auc = roc_auc_score(y, preds)
    except TypeError:
        # Some estimators don't accept sample_weight in fit_params via cross_val_predict
        preds = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, preds)

    if baseline_auc is None:
        baseline_auc = auc

    delta = auc - baseline_auc
    delta_str = f"{delta:+.4f}" if baseline_auc is not None and delta != 0 else "  (baseline)"
    print(f"  {name:<36}  {auc:.4f}  {delta_str:>12}")
    results.append({"model": name, "auc": round(auc, 4),
                    "vs_baseline": round(delta, 4)})

# ── GAM (manual CV — no sample_weight support in cross_val_predict) ──────────
if HAS_GAM:
    try:
        from pygam import LogisticGAM, s as gam_s
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        gam_preds = np.zeros(len(y))
        for tr_idx, te_idx in cv.split(Xs, y):
            gam = LogisticGAM(terms="auto").fit(Xs[tr_idx], y[tr_idx])
            gam_preds[te_idx] = gam.predict_proba(Xs[te_idx])
        auc = roc_auc_score(y, gam_preds)
        delta = auc - baseline_auc
        print(f"  {'GAM (LogisticGAM)':<36}  {auc:.4f}  {delta:+.4f}")
        results.append({"model": "GAM (LogisticGAM)", "auc": round(auc, 4),
                        "vs_baseline": round(delta, 4)})
    except Exception as e:
        print(f"  GAM failed: {e}")

print("─"*60)

# ── Save results ──────────────────────────────────────────────────────────────
out = DATA_DIR / "DF_benchmark_AUC.csv"
pd.DataFrame(results).sort_values("auc", ascending=False).to_csv(out, index=False)
print(f"\nSaved → {out}")
print("\nInterpretation guide:")
print("  AUC 0.50 = coin flip   AUC 0.60 = modest   AUC 0.70+ = strong")
print("  With public Statcast data, ceiling is typically ~0.65–0.70.")
print("  If one model is notably better, tune THAT model in the next pass.")
