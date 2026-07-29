#!/usr/bin/env python3
"""
model_v13.py — the DECISION model (2023–2026): P(attempt), not P(success).

Scope note: this is a 2023–2026 model on purpose. The opportunity denominator is built from the
post-rule-change environment (disengagement limit, bigger bases, pitch clock), where the decision
to run is a different game than it was pre-2023. The classic 2015–2022 era has its own separate
Steal+/Burst + per-attempt fit (see model_classic.py); no classic decision model is built.

Every other model in this repo conditions on a steal attempt having happened. That is why sprint
speed alone scores AUROC 0.53 on the success model: runners self-select, only going when they
already like their odds. This file models the other half — given a runner on 1st with 2nd open,
does he go? — using the opportunity denominator that never existed before
(`scrape_statcast.py opportunities`).

WHAT THIS MODEL DELIBERATELY CANNOT USE. The lead distances that carry essentially all of
v11/v12's signal (lead_at_firstmove_ft, gain_to_release_ft, lead_at_release_ft) come from Savant's
basestealing drawer, which only publishes TRACKED STEAL ATTEMPTS. They do not exist on the pitches
where the runner stayed. The paper this design borrows from (Team 191, SMT, "Winning the Signaling
Game") treats the lead as the runner's signal of intent — here that signal is unobservable on the
non-attempt side, so the decision must be modelled from situation + personnel alone. That is a data
limitation, not a modelling choice, and it belongs in any writeup of this model.

WHY AUPRC LEADS HERE. The success model is 81% positive, which makes AUPRC near-trivially high and
AUROC the informative metric. Attempts are only ~2-4% of opportunities, so the emphasis inverts:
AUPRC (against the attempt base rate as the no-skill floor) is the headline, AUROC secondary.

v11 (published Steal+/Burst) and v12 (per-attempt success engine) are untouched — v13 is additive.

Inputs:  Data/Raw_Opportunities.csv  ·  Data/sprint_speed.csv  ·  Data/poptime.csv
Run:     python3 Scripts/model_v13.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

from model_v12 import PITCH_CLASS, reliability

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "Data"
RESULTS = ROOT / "Output" / "Results"
FIGS    = ROOT / "Output" / "Figures"

# pc_fastball omitted as the reference category — the three dummies sum to 1 on 99.9% of rows, so
# including all three is the dummy-variable trap (see model_v12 BATTERY_FEATS).
SITUATION = ["balls", "strikes", "outs", "inning", "score_diff", "is_lhp", "bat_side_r",
             "pc_breaking", "pc_offspeed"]
PERSONNEL = ["sprint_speed_all", "runner_rate_prior", "seen_before"]
FEATS     = SITUATION + PERSONNEL


def load() -> tuple[pd.DataFrame, np.ndarray]:
    # committed gzipped (41 MB -> a few MB); the scraper writes the plain .csv
    src = DATA / "Raw_Opportunities.csv"
    if not src.exists():
        src = DATA / "Raw_Opportunities.csv.gz"
    if not src.exists():
        raise SystemExit("Data/Raw_Opportunities.csv[.gz] missing — "
                         "run: python3 Scripts/scrape_statcast.py opportunities")
    df = pd.read_csv(src)
    df = df[(df["balls"] <= 3) & (df["strikes"] <= 2)]          # one malformed feed row in 517k

    # Season: the opportunities table has no date, but game_pk is monotonic in time and the
    # seasons occupy cleanly separated blocks. Learn the block edges from the games that DO
    # contain a Savant-tracked play_id, then assign EVERY game by range — joining on play_id
    # alone would silently drop the ~36% of games with no tracked attempt in them.
    att = pd.read_csv(DATA / "Raw_Attempts.csv", usecols=["play_id", "season"])
    known = (df.merge(att, on="play_id", how="inner")
               .groupby("game_pk")["season"].first().reset_index())
    bounds = known.groupby("season")["game_pk"].agg(["min", "max"]).sort_index()
    seasons = bounds.index.to_numpy()
    edges = [(bounds["max"].iloc[i] + bounds["min"].iloc[i + 1]) / 2
             for i in range(len(bounds) - 1)]
    df["season"] = seasons[np.searchsorted(edges, df["game_pk"].values)]

    cls = df["pitch_code"].map(PITCH_CLASS).fillna("other")
    for c in ("fastball", "breaking", "offspeed"):
        df[f"pc_{c}"] = (cls == c).astype(int)

    sp = pd.read_csv(DATA / "sprint_speed.csv").rename(columns={"runner_id": "runner_1b"})
    df = df.merge(sp, on=["runner_1b", "season"], how="left")

    # NOTE: no catcher feature here. The opportunity feed carries no catcher id, and a
    # season-league mean would be constant within a season — a column that looks like a feature
    # but cannot inform a per-pitch decision. Attaching the real catcher needs the boxscore.

    # The "scouting report" prior: how often does THIS runner go? It must be built from PRIOR
    # SEASONS ONLY. A same-season leave-one-out rate still reads the season being predicted, which
    # inflated the random split to AUPRC 0.25 against 0.06 grouped — the same self-contamination
    # that produced the v11 catcher/pitcher bug and was caught again on Burst. Expanding by season
    # is causal and is what you would actually have on hand at gametime.
    league = float(df["attempt"].mean())
    per = (df.groupby(["runner_1b", "season"])["attempt"].agg(["sum", "count"])
             .sort_index().reset_index())
    per[["cum_s", "cum_n"]] = (per.groupby("runner_1b")[["sum", "count"]]
                                  .transform(lambda c: c.shift(1).cumsum()))
    per["runner_rate_prior"] = (per["cum_s"] / per["cum_n"]).fillna(league)
    df = df.merge(per[["runner_1b", "season", "runner_rate_prior"]],
                  on=["runner_1b", "season"], how="left")
    df["runner_rate_prior"] = df["runner_rate_prior"].fillna(league)
    df["seen_before"] = (df["runner_rate_prior"] != league).astype(int)

    df = df.reset_index(drop=True)
    return df, df["attempt"].values


def evaluate(df, y, feats, split="random", seed=42, return_pred=False):
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from xgboost import XGBClassifier

    feats = [f for f in feats if f in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce").values

    def splits():
        if split == "random":
            yield from StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y)
        elif split == "group":
            yield from GroupKFold(5).split(df, y, df["runner_1b"].values)
        elif split == "forward":
            for T in sorted(df["season"].unique())[1:]:
                tr = np.where(df["season"].values < T)[0]
                va = np.where(df["season"].values == T)[0]
                if len(va) >= 500 and len(tr) >= 2000:
                    yield tr, va

    oof = np.full(len(df), np.nan)
    for tr, va in splits():
        m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=10, reg_lambda=1.0,
                          eval_metric="logloss", verbosity=0, random_state=seed,
                          use_label_encoder=False)
        oof[va] = m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]

    ok = ~np.isnan(oof)
    yy, pp = y[ok], oof[ok]
    out = {"auprc": average_precision_score(yy, pp), "auroc": roc_auc_score(yy, pp),
           "brier": brier_score_loss(yy, pp), "base_rate": float(yy.mean()),
           "n_scored": int(ok.sum())}
    return (out, oof) if return_pred else out


def main():
    df, y = load()
    print(f"v13 DECISION MODEL (2023-2026): {len(df):,} opportunity pitches (runner on 1B, 2B empty) | "
          f"attempts {y.sum():,} | attempt rate {y.mean()*100:.2f}%")
    print(f"seasons: {sorted(df['season'].unique())} | distinct runners: {df['runner_1b'].nunique():,}")

    rows = []
    for name, feats in [("situation only", SITUATION),
                        ("personnel only", PERSONNEL),
                        ("v13 FULL (situation + personnel)", FEATS)]:
        m = evaluate(df, y, feats)
        rows.append({"model": name, "split": "random", **{k: round(v, 4) for k, v in m.items()}})
        print(f"  {name:34s} AUPRC {m['auprc']:.4f} (floor {m['base_rate']:.4f}) | "
              f"AUROC {m['auroc']:.4f} | Brier {m['brier']:.4f}")

    print("\nvalidation regimes (full model):")
    for split in ("random", "group", "forward"):
        m = evaluate(df, y, FEATS, split=split)
        rows.append({"model": "v13 FULL", "split": split, **{k: round(v, 4) for k, v in m.items()}})
        print(f"  {split:8s} AUPRC {m['auprc']:.4f} (floor {m['base_rate']:.4f}) | "
              f"AUROC {m['auroc']:.4f} | Brier {m['brier']:.4f} | n={m['n_scored']:,}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "DF_v13_Attempt.csv", index=False)

    # ── the diagnostic this model exists for: the selection effect, without a model ──
    # SPEED BINS, DEFINED. Quintiles of the OPPORTUNITY population — every pitch with a runner on
    # 1st — so a runner who reaches base often carries more weight. They are NOT MLB-wide
    # percentiles and NOT equal numbers of runners, and a runner can fall in different bins in
    # different seasons because sprint speed is measured per season.
    q, edges = pd.qcut(df["sprint_speed_all"], 5, retbins=True, duplicates="drop",
                       labels=["slowest", "slow", "mid", "fast", "fastest"])
    tab = df.groupby(q).agg(low_ftps=("sprint_speed_all", "min"),
                            high_ftps=("sprint_speed_all", "max"),
                            mean_ftps=("sprint_speed_all", "mean"),
                            runner_seasons=("runner_1b", "nunique"),
                            opportunities=("attempt", "size"),
                            attempts=("attempt", "sum")).round(2)
    tab["attempt_pct"] = (100 * tab["attempts"] / tab["opportunities"]).round(2)
    tab["share_of_opps_pct"] = (100 * tab["opportunities"] / len(df)).round(1)
    print("\nSPEED BINS — quintiles of the opportunity population (pitch-weighted, NOT MLB percentile):")
    print(tab.to_string())
    tab.to_csv(RESULTS / "DF_v13_SelectionEffect.csv")

    pcts = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    pct = pd.DataFrame({"percentile": pcts,
                        "sprint_speed_ftps": [round(float(np.percentile(
                            df["sprint_speed_all"].dropna(), p)), 1) for p in pcts]})
    pct.to_csv(RESULTS / "DF_v13_SpeedPercentiles.csv", index=False)
    print("speed percentiles in this population (ft/s): " +
          " · ".join(f"p{r.percentile}={r.sprint_speed_ftps}" for r in pct.itertuples()))

    print("\nattempt rate by count:")
    ct = df.groupby(["balls", "strikes"])["attempt"].agg(["mean", "size"])
    ct = ct[ct["size"] >= 2000].sort_values("mean", ascending=False)
    print((ct.assign(attempt_pct=(ct["mean"] * 100).round(2))
             .drop(columns="mean").head(6)).to_string())

    _, oof = evaluate(df, y, FEATS, split="forward", return_pred=True)
    rel = reliability(y, oof)
    rel.to_csv(RESULTS / "DF_v13_Calibration.csv", index=False)
    print(f"\ncalibration (forward): max |predicted − observed| = {rel['gap'].abs().max():.4f}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGS.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        t = df.groupby(q)["attempt"].mean() * 100
        ax.bar([str(i) for i in t.index], t.values, color="#2F6FB0", width=0.6)
        ax.set_ylabel("attempt rate (%)"); ax.set_xlabel("runner sprint-speed quintile")
        ax.set_title("Runners self-select: who even tries to steal (2023–2026)", fontsize=12)
        for i, v in enumerate(t.values):
            ax.text(i, v + 0.03, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=10)
        plt.tight_layout(); plt.savefig(FIGS / "Fig_v13_Attempt.png", dpi=160); plt.close()
    except ImportError:
        pass
    return rows


if __name__ == "__main__":
    main()
