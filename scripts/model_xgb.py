"""
model_xgb.py  —  Model B as tuned XGBoost (v8.3 promotion)
==========================================================

Re-fits the season-level steal-success model (Model B) using the Bayesian-tuned
XGBoost hyperparameters (Data Frame/DF_xgb_tuned_params.csv, CV AUC ≈ 0.629) in
place of the old GradientBoostingClassifier (AUC ≈ 0.589 full / 0.608 pre-23).

Runs WITHOUT network — reads the already-computed feature CSV (DF_v7_SSSI.csv),
so it refreshes the model artifacts without re-running the full Statcast pipeline:
  - DF_v7_ModelB_AUC.csv      (epoch, n, auc)  — full / pre_2023 / post_2023
  - DF_v7_Importance.csv      (epoch, feature, importance)  — XGBoost gain
  - Fig_v7_AUC.png            (v7 bar updated to the XGBoost AUC)
  - Fig_v7_Importance_PrePost.png

Exposes fit_model_b() / load_best_params() so v7_explore.py shares one source of truth.

Usage:  python3 model_xgb.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

SEED = 42

# Same feature list + filter as benchmark_models.py / tune_xgboost.py (kept in sync
# so the AUC matches the tuning run that produced the best params).
FEATURES = [
    "sprint_speed", "speed_capped", "jump_time",
    "total_90", "accel_gap", "bolts",
    "dist_to_top_speed_ft", "accel_topspeed_gap",
    "primary_lead", "secondary_lead", "lead_gain",
    "avg_pop_faced", "avg_pickoff_rate_faced", "weak_arm_share", "two_strike_share",
    "avg_pre_release_velocity", "avg_post_release_distance",
    "n_attempts",
]

COLOR = {"pre": "#E0A458", "post": "#3D5A80", "neutral": "#374151",
         "highlight": "#10B981", "accent": "#0EA5E9", "below": "#F59E0B"}

FRIENDLY = {
    "jump_time": "Jump Time", "sprint_speed": "Sprint Speed",
    "speed_capped": "Sprint (capped)", "primary_lead": "Primary Lead",
    "lead_gain": "Lead Gain", "secondary_lead": "Secondary Lead",
    "avg_pop_faced": "Pop Time Faced", "avg_pickoff_rate_faced": "Pickoff Rate Faced",
    "weak_arm_share": "Weak-Arm Catcher Share", "avg_pre_release_velocity": "Pre-Release Velocity",
    "avg_post_release_distance": "Post-Release Distance", "accel_gap": "Accel Gap",
    "bolts": "Bolts", "total_90": "Total 90", "dist_to_top_speed_ft": "Dist to Top Speed",
    "accel_topspeed_gap": "Accel→Top-Speed Gap", "two_strike_share": "Two-Strike Share",
    "n_attempts": "# Attempts",
}


def find_dirs(root: Path | None = None):
    """Resolve (data_dir, figures_dir) whether data lives in 'data/' or 'Data Frame/'."""
    root = root or Path(__file__).resolve().parent
    # If we live in scripts/, the repo root is the parent.
    if not (root / "Figures").exists() and (root.parent / "Figures").exists():
        root = root.parent
    data_dir = (root / "data") if (root / "data").exists() else (root / "Data Frame")
    return data_dir, root / "Figures"


def load_best_params(data_dir: Path) -> dict:
    p = data_dir / "DF_xgb_tuned_params.csv"
    row = pd.read_csv(p).iloc[0].to_dict()
    row.pop("best_auc", None)
    params = {
        "max_depth": int(row["max_depth"]),
        "min_child_weight": float(row["min_child_weight"]),
        "gamma": float(row["gamma"]),
        "n_estimators": int(row["n_estimators"]),
        "learning_rate": float(row["learning_rate"]),
        "subsample": float(row["subsample"]),
        "colsample_bytree": float(row["colsample_bytree"]),
        "colsample_bylevel": float(row["colsample_bylevel"]),
        "reg_alpha": float(row["reg_alpha"]),
        "reg_lambda": float(row["reg_lambda"]),
        "eval_metric": "logloss", "verbosity": 0,
        "random_state": SEED, "use_label_encoder": False,
    }
    return params


def make_model(params: dict) -> XGBClassifier:
    return XGBClassifier(**params)


def prep(df: pd.DataFrame):
    feats = [c for c in FEATURES if c in df.columns]
    mask = (df["sb_attempts"] >= 10) & df["real_sb_pct"].notna()
    league_sb = df.loc[mask, "real_sb_pct"].median()
    work = df[mask].dropna(subset=feats + ["real_sb_pct"]).copy()
    work["y"] = (work["real_sb_pct"] >= league_sb).astype(int)
    return work, feats, league_sb


def fit_model_b(df_sub: pd.DataFrame, feats: list[str], params: dict, label: str):
    """5-fold stratified CV AUC, sample_weight = sb_attempts (matches v7 Model B)."""
    if len(df_sub) < 50:
        return None
    X = df_sub[feats].values
    y = df_sub["y"].values
    w = df_sub["sb_attempts"].values.astype(float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = cross_val_predict(make_model(params), X, y, cv=cv,
                              method="predict_proba",
                              params={"sample_weight": w})[:, 1]
    return {"label": label, "n": len(df_sub), "auc": roc_auc_score(y, preds),
            "X": X, "y": y, "w": w, "feats": feats}


def main():
    data_dir, fig_dir = find_dirs()
    print(f"data_dir = {data_dir}")
    df = pd.read_csv(data_dir / "DF_v7_SSSI.csv")
    params = load_best_params(data_dir)
    work, feats, league_sb = prep(df)
    print(f"  rows={len(work)}  features={len(feats)}  league_sb={league_sb:.3f}")

    eras = {"full": work,
            "pre_2023": work[work["era"] == "pre_2023"],
            "post_2023": work[work["era"] == "post_2023"]}

    # ── AUC by era ───────────────────────────────────────────────────────────
    auc_rows, models = [], {}
    for label, sub in eras.items():
        m = fit_model_b(sub, feats, params, label)
        if m is None:
            continue
        models[label] = m
        auc_rows.append({"epoch": label, "n": m["n"], "auc": round(m["auc"], 4)})
        print(f"  {label:>10}  n={m['n']:>4}  AUC={m['auc']:.4f}")
    pd.DataFrame(auc_rows).to_csv(data_dir / "DF_v7_ModelB_AUC.csv", index=False)

    # ── Feature importance (XGBoost gain) by era ─────────────────────────────
    imp_rows = []
    for label, m in models.items():
        clf = make_model(params).fit(m["X"], m["y"], sample_weight=m["w"])
        for f, v in zip(m["feats"], clf.feature_importances_):
            imp_rows.append({"epoch": label, "feature": f, "importance": round(float(v), 4)})
    DF_Imp = pd.DataFrame(imp_rows)
    DF_Imp.to_csv(data_dir / "DF_v7_Importance.csv", index=False)

    # ── Fig: AUC across versions (v7 bar = tuned XGBoost) ────────────────────
    full_auc = models["full"]["auc"] if "full" in models else float("nan")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels = ["v4\n(season)", "v5 Model A\n(per-attempt)", "v5 Model B\n(season+new)",
              "v6 Model B", "v8 Model B\n(XGBoost, de-leaked)"]
    aucs = [0.6300, 0.5933, 0.6794, 0.6620, full_auc]
    ax.bar(labels, aucs, color=[COLOR["neutral"], COLOR["accent"], COLOR["post"],
                                COLOR["below"], COLOR["highlight"]])
    ax.set_ylabel("CV AUC"); ax.set_ylim(0.5, 0.85)
    ax.set_title("Model AUC across versions  (v8 = Bayesian-tuned XGBoost)")
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)
    ax.text(0.5, -0.30,
            "v4–v6 bars carried duplicate runner-season rows that leaked across CV folds (optimistic).\n"
            "v7 de-leaking → one row per runner-season; v8 swaps GBM for a tuned XGBoost on the same de-leaked data.\n"
            "The v8 bar is the honest tuned AUC and is not directly comparable to the historical bars.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#555555")
    fig.subplots_adjust(bottom=0.34)
    fig.savefig(fig_dir / "Fig_v7_AUC.png", dpi=160); plt.close(fig)

    # ── Fig: Pre vs Post importance ──────────────────────────────────────────
    if not DF_Imp.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        piv = DF_Imp.pivot(index="feature", columns="epoch", values="importance")
        piv.index = piv.index.map(lambda f: FRIENDLY.get(f, f))
        keep = [c for c in ["pre_2023", "post_2023"] if c in piv.columns]
        if keep:
            piv = piv.dropna(subset=keep).sort_values(keep[-1])
        yy = np.arange(len(piv))
        ax.barh(yy - 0.18, piv.get("pre_2023", piv.iloc[:, 0]), height=0.36,
                color=COLOR["pre"], label="pre-2023")
        ax.barh(yy + 0.18, piv.get("post_2023", piv.iloc[:, -1]), height=0.36,
                color=COLOR["post"], label="post-2023")
        ax.set_yticks(yy); ax.set_yticklabels(piv.index)
        ax.set_xlabel("XGBoost feature importance (gain)")
        ax.set_title("v8 — Feature Importance · Pre vs Post 2023  (tuned XGBoost)")
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "Fig_v7_Importance_PrePost.png", dpi=160); plt.close(fig)

    write_curated_root_csvs(data_dir, fig_dir.parent)
    print("model_xgb: wrote AUC + importance CSVs, refreshed figures, and curated root CSVs.")
    return auc_rows


def write_curated_root_csvs(data_dir: Path, root: Path):
    """Emit the 2 human-facing root CSVs the clean repo surfaces.

    Naylor_Model_Data.csv    — the runner-season master (= DF_v7_SSSI.csv).
    Naylor_Model_Results.csv — stacked model results (GLM weights, AUC by era,
                               benchmark AUCs, tuned XGBoost params), with a `section`.
    """
    sssi = data_dir / "DF_v7_SSSI.csv"
    if sssi.exists():
        pd.read_csv(sssi).to_csv(root / "Naylor_Model_Data.csv", index=False)

    blocks = []
    def add(fname, section, rename=None):
        p = data_dir / fname
        if not p.exists():
            return
        d = pd.read_csv(p)
        if rename:
            d = d.rename(columns=rename)
        d.insert(0, "section", section)
        blocks.append(d)

    add("DF_v7_GLM_PlainEnglish.csv", "glm_weights")
    add("DF_v7_ModelB_AUC.csv",       "modelB_auc_by_era")
    add("DF_benchmark_AUC.csv",       "benchmark_auc")
    # tuned params: melt the single row to long key/value for tidy stacking
    tp = data_dir / "DF_xgb_tuned_params.csv"
    if tp.exists():
        row = pd.read_csv(tp).iloc[0]
        mp = pd.DataFrame({"section": "xgb_tuned_params",
                           "param": row.index, "value": row.values})
        blocks.append(mp)
    if blocks:
        pd.concat(blocks, ignore_index=True).to_csv(
            root / "Naylor_Model_Results.csv", index=False)


if __name__ == "__main__":
    main()
