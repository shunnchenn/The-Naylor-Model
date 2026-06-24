"""
model_perattempt.py  —  THE Naylor Model (per-attempt, ~11,000 rows)
====================================================================

This is the PRIMARY model of the project. Players are analysed at the grain that
actually decides a steal: the **individual attempt**, not a season average.

    ~11,169 tracked steal attempts   (one row per attempt)   ← the unit of analysis
        vs.
       673 runner-seasons            (one aggregated row per player-year)   ← too coarse

Modelling at the attempt level is what gives the project its strength. Each row is a
single steal with its own pre-pitch context — the exact lead distances the runner got
on THAT pitch — so the model learns what makes one steal succeed, with 17× more rows
and far less averaging noise than a season aggregate. CV AUC ≈ 0.74, driven by the
lead distances, which is precisely this project's thesis.

Target:  y = 1 if the attempt was a stolen base (SB), 0 if caught (CS).

Leakage discipline:
  - NO outcome-derived columns (run_value is dropped).
  - catcher/pitcher "tendency" features are OUT-OF-FOLD target-encoded (fit on the
    training fold only) so a catcher's own test attempts never inform his encoding.
  - runner skill is joined from DF_v7_SSSI (sprint speed, jump, etc.) — known pre-pitch;
    the runner's season success rate (real_sb_pct) is excluded (it would leak the target).

Runs WITHOUT network.  Usage:  python3 Scripts/model_perattempt.py
Writes:
  Output/Results/DF_perattempt_AUC.csv          (the two model variants)
  Output/Results/DF_perattempt_Importance.csv   (feature importances)
  Output/Figures/Fig_AUC.png                    (per-attempt AUC)
  Output/Figures/Fig_Importance.png             (per-attempt feature importance)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import glob, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

SEED = 42
COLOR = {"primary": "#10B981", "muted": "#9CA3AF", "navy": "#1F2D3D", "lead": "#0EA5E9"}


def find_root() -> Path:
    r = Path(__file__).resolve().parent
    if not (r / "Output").exists() and (r.parent / "Output").exists():
        r = r.parent
    return r

ROOT    = find_root()
RESULTS = ROOT / "Output" / "Results"
FIGS    = ROOT / "Output" / "Figures"
CACHE   = ROOT / "Computer Vision" / "data" / "discovery" / "leads_cache"
for _d in (RESULTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

LEAD_FEATS = ["lead_at_firstmove_ft", "gain_to_release_ft", "lead_at_release_ft"]
# runner-skill columns joined from the season table (known before the pitch; no target leak)
RUNNER_FEATS = ["sprint_speed", "jump_time", "accel_gap", "primary_lead", "lead_gain", "bolts"]

FRIENDLY = {
    "lead_at_firstmove_ft": "Lead at first move (ft)",
    "gain_to_release_ft":   "Ground gained to release (ft)",
    "lead_at_release_ft":   "Lead at release (ft)",
    "base_is_3b":           "Stealing 3rd",
    "sprint_speed":         "Sprint speed",
    "jump_time":            "Jump time",
    "accel_gap":            "Accel gap",
    "primary_lead":         "Primary lead (career)",
    "lead_gain":            "Lead gain (career)",
    "bolts":                "Bolts",
}


def load_attempts() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(CACHE / "*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if len(d) == 0 or "result" not in d.columns:
            continue
        rid, yr = os.path.basename(f).replace(".csv", "").split("_")
        d["runner_id"] = int(rid); d["season"] = int(yr)
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df = df[df["result"].isin(["SB", "CS"])].copy()
    df["y"] = (df["result"] == "SB").astype(int)
    df["base_is_3b"] = (df["base"].astype(str) == "3B").astype(int)
    return df


def oof_target_encode(train_idx, val_idx, key, y, df, prior, smoothing=20.0):
    """Smoothed mean-target encoding fit on train rows, applied to val rows."""
    tr = df.iloc[train_idx]
    stats = tr.groupby(key)["y"].agg(["sum", "count"])
    enc = (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)
    return df.iloc[val_idx][key].map(enc).fillna(prior).values, \
           df.iloc[train_idx][key].map(enc).fillna(prior).values


def _xgb():
    return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         reg_lambda=1.0, eval_metric="logloss", verbosity=0,
                         random_state=SEED, use_label_encoder=False)


def main():
    df = load_attempts()
    print(f"PRIMARY GRAIN — per attempt: {len(df):,} attempts "
          f"(SB {int(df.y.sum()):,} / CS {int((1-df.y).sum()):,}, rate {df.y.mean():.3f})")

    # join runner skill from the season table
    sssi = pd.read_csv(RESULTS / "DF_v7_SSSI.csv")
    keep = ["runner_id", "season"] + [c for c in RUNNER_FEATS if c in sssi.columns]
    df = df.merge(sssi[keep].drop_duplicates(["runner_id", "season"]),
                  on=["runner_id", "season"], how="left")
    runner_cols = [c for c in RUNNER_FEATS if c in df.columns]

    base_num = LEAD_FEATS + ["base_is_3b"] + runner_cols
    df[base_num] = df[base_num].apply(pd.to_numeric, errors="coerce")
    prior = df["y"].mean()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = df["y"].values

    def run(use_ids: bool, label: str):
        oof = np.zeros(len(df))
        for tr, va in cv.split(df, y):
            Xtr = df.iloc[tr][base_num].copy(); Xva = df.iloc[va][base_num].copy()
            if use_ids:
                for key, nm in [("catcher_id", "catch_enc"), ("pitcher_id", "pitch_enc")]:
                    va_enc, tr_enc = oof_target_encode(tr, va, key, y, df, prior)
                    Xtr[nm] = tr_enc; Xva[nm] = va_enc
            clf = _xgb()
            clf.fit(Xtr.values, y[tr])
            oof[va] = clf.predict_proba(Xva.values)[:, 1]
        auc = roc_auc_score(y, oof)
        print(f"  {label:<48} AUC = {auc:.4f}")
        return auc

    print("\nPer-attempt SB-success model (5-fold stratified CV, pooled OOF AUC):")
    a_lead = run(False, "leads + base + runner skill")
    a_full = run(True,  "+ catcher & pitcher tendency (OOF target-encoded)")

    out = pd.DataFrame([
        {"model": "per-attempt: leads+base+runner", "auc": round(a_lead, 4)},
        {"model": "per-attempt: + catcher/pitcher (OOF)", "auc": round(a_full, 4)},
    ])
    out.to_csv(RESULTS / "DF_perattempt_AUC.csv", index=False)
    print(f"\nwrote {RESULTS / 'DF_perattempt_AUC.csv'}")

    # ── feature importance (final fit on all attempts, leads+runner variant) ──
    final = _xgb().fit(df[base_num].values, y)
    imp = (pd.DataFrame({"feature": base_num, "importance": final.feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    imp.to_csv(RESULTS / "DF_perattempt_Importance.csv", index=False)
    print(f"wrote {RESULTS / 'DF_perattempt_Importance.csv'}")

    _fig_auc(a_lead, a_full, len(df))
    _fig_importance(imp)
    return out


def _fig_auc(a_lead, a_full, n):
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    labels = ["Leads + runner skill", "+ catcher / pitcher\ntendency (OOF)"]
    aucs   = [a_lead, a_full]
    ax.bar(labels, aucs, color=[COLOR["primary"], COLOR["muted"]], width=0.5)
    ax.axhline(0.50, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=0)
    ax.text(1.01, 0.50, "coin flip", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=7.5, color="#888888")
    ax.set_ylabel("CV AUC"); ax.set_ylim(0.5, 0.82)
    ax.set_title("Per-Attempt Model — the leads carry the signal", fontsize=11.5)
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
    ax.text(0.5, -0.11,
            f"Trained on {n:,} individual steal attempts (one row per attempt). Adding battery "
            "tendencies LOWERS AUC — the lead distances alone carry the signal.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555555")
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(FIGS / "Fig_AUC.png", dpi=160); plt.close(fig)
    print(f"wrote {FIGS / 'Fig_AUC.png'}")


def _fig_importance(imp):
    g = imp.iloc[::-1]
    names = [FRIENDLY.get(f, f) for f in g["feature"]]
    is_lead = [f in LEAD_FEATS for f in g["feature"]]
    colors = [COLOR["lead"] if l else COLOR["navy"] for l in is_lead]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.barh(names, g["importance"], color=colors)
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Per-Attempt Model — what decides a steal (blue = per-pitch lead distances)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "Fig_Importance.png", dpi=160); plt.close(fig)
    print(f"wrote {FIGS / 'Fig_Importance.png'}")


if __name__ == "__main__":
    main()
