"""
tune_xgboost.py  —  Bayesian Hyperparameter Optimisation for XGBoost
Uses Optuna (TPE sampler) to maximise 5-fold stratified CV AUC.
Same data / features / weights as v7 Model B + benchmark_models.py.

Run:  python3 tune_xgboost.py
Outputs:
  - prints best params + AUC at each improvement
  - Data Frame/DF_xgb_tuned_params.csv   (best trial)
  - Data Frame/DF_xgb_optuna_trials.csv  (all 100 trials)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
if not (ROOT / "Figures").exists() and (ROOT.parent / "Figures").exists():
    ROOT = ROOT.parent           # script moved into scripts/
DATA_DIR = (ROOT / "data") if (ROOT / "data").exists() else (ROOT / "Data Frame")
SEED     = 42
N_TRIALS = 100   # ~2-3 min on this dataset size

# ── Load & prep (mirror v7 / benchmark_models.py exactly) ─────────────────────
print("Loading data …")
df = pd.read_csv(DATA_DIR / "DF_v7_SSSI.csv")

mask_q    = (df["sb_attempts"] >= 10) & df["real_sb_pct"].notna()
league_sb = df.loc[mask_q, "real_sb_pct"].median()

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

X = work[FEATURES].values
y = work["y"].values
w = work["sb_attempts"].values.astype(float)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print(f"  n={len(work)}  features={len(FEATURES)}")
print(f"  Baseline (default XGB, no tuning): ", end="", flush=True)
baseline = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.04,
                          eval_metric="logloss", verbosity=0,
                          random_state=SEED, use_label_encoder=False)
res = cross_validate(baseline, X, y, cv=cv, scoring="roc_auc",
                     params={"sample_weight": w})
scores = res["test_score"]
print(f"AUC = {scores.mean():.4f}  (±{scores.std():.4f})")

best_so_far = {"auc": 0.0}

# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial):
    params = {
        # Tree structure
        "max_depth":        trial.suggest_int("max_depth", 2, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),

        # Boosting
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),

        # Stochastic regularisation
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.5, 1.0),

        # L1/L2 regularisation
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),

        # Fixed
        "eval_metric": "logloss",
        "verbosity":   0,
        "random_state": SEED,
        "use_label_encoder": False,
    }

    clf    = XGBClassifier(**params)
    res    = cross_validate(clf, X, y, cv=cv, scoring="roc_auc",
                            params={"sample_weight": w})
    auc    = res["test_score"].mean()

    if auc > best_so_far["auc"]:
        best_so_far["auc"]    = auc
        best_so_far["params"] = params
        print(f"  ★ Trial {trial.number:>3}  AUC={auc:.4f}  "
              f"depth={params['max_depth']}  lr={params['learning_rate']:.4f}  "
              f"n={params['n_estimators']}  sub={params['subsample']:.2f}  "
              f"col={params['colsample_bytree']:.2f}")

    return auc


# ── Run study ─────────────────────────────────────────────────────────────────
print(f"\nRunning Bayesian HPO — {N_TRIALS} trials (TPE sampler) …")
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best = study.best_trial
print(f"\n{'─'*60}")
print(f"Best AUC: {best.value:.4f}  (vs untuned XGB: {scores.mean():.4f}, "
      f"vs GBM baseline: 0.6004)")
print(f"{'─'*60}")
print("Best hyperparameters:")
for k, v in best.params.items():
    print(f"  {k:<24} {v}")

# ── Save results ──────────────────────────────────────────────────────────────
params_row = {**best.params, "best_auc": round(best.value, 4)}
pd.DataFrame([params_row]).to_csv(DATA_DIR / "DF_xgb_tuned_params.csv", index=False)

trials_df = study.trials_dataframe()[["number","value","duration"]].rename(
    columns={"number":"trial","value":"auc"})
trials_df.to_csv(DATA_DIR / "DF_xgb_optuna_trials.csv", index=False)

print(f"\nSaved:")
print(f"  {DATA_DIR / 'DF_xgb_tuned_params.csv'}")
print(f"  {DATA_DIR / 'DF_xgb_optuna_trials.csv'}")
print(f"\nNext step: plug best params into v7_explore.py Model B to replace the GBM.")
