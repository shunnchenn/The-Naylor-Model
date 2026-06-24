"""
model_perattempt.py  —  Per-attempt steal-success model (v9 experiment)
========================================================================

The season-level Model B tops out near AUC 0.62 because it has only 673 rows and a
noisy season-average target. This builds a PER-ATTEMPT model from the cached Savant
leads (Computer Vision/data/discovery/leads_cache/, ~11k attempts) — far more rows
and the per-attempt context (exact leads, base stolen, catcher/pitcher faced) that
actually drives whether one steal succeeds.

Target: y = 1 if the attempt was a stolen base (SB), 0 if caught (CS).

Leakage discipline:
  - NO outcome-derived columns (run_value is dropped).
  - catcher/pitcher "tendency" features are OUT-OF-FOLD target-encoded (computed on the
    training fold only, applied to the held-out fold) so a catcher's own test attempts
    never inform his encoding.
  - runner skill is joined from DF_v7_SSSI (sprint speed, jump, etc.) — known pre-pitch;
    the runner's season success rate (real_sb_pct) is excluded (it would leak the target).

Runs WITHOUT network.  Usage:  python3 scripts/model_perattempt.py
Writes:  data/DF_perattempt_AUC.csv
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import glob, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

SEED = 42

def find_root() -> Path:
    r = Path(__file__).resolve().parent
    if not (r / "Figures").exists() and (r.parent / "Figures").exists():
        r = r.parent
    return r

ROOT = find_root()
DATA = (ROOT / "data") if (ROOT / "data").exists() else (ROOT / "Data Frame")
CACHE = ROOT / "Computer Vision" / "data" / "discovery" / "leads_cache"

LEAD_FEATS = ["lead_at_firstmove_ft", "gain_to_release_ft", "lead_at_release_ft"]
# runner-skill columns to join from the season table (known before the pitch; no target leak)
RUNNER_FEATS = ["sprint_speed", "jump_time", "accel_gap", "primary_lead", "lead_gain", "bolts"]


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


def main():
    df = load_attempts()
    print(f"attempts: {len(df)}  (SB {df.y.sum()} / CS {(1-df.y).sum()}, rate {df.y.mean():.3f})")

    # join runner skill from the season table
    sssi = pd.read_csv(DATA / "DF_v7_SSSI.csv")
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
            clf = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                reg_lambda=1.0, eval_metric="logloss", verbosity=0,
                                random_state=SEED, use_label_encoder=False)
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
        {"model": "season Model B (XGBoost, for reference)", "auc": 0.6244},
    ])
    out.to_csv(DATA / "DF_perattempt_AUC.csv", index=False)
    print(f"\nwrote {DATA / 'DF_perattempt_AUC.csv'}")
    return out


if __name__ == "__main__":
    main()
