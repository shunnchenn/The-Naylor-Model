#!/usr/bin/env python3
"""
model_classic.py — the CLASSIC era (2015–2022): Steal+ / Burst and the per-attempt success
model, fit SEPARATELY from the modern 2023–2026 model.

Why a separate fit. The 2023 rule package (two-disengagement limit, bigger bases, pitch clock)
changed base-stealing enough that pooling the eras would blur two different games: success rates
and secondary leads both jump after 2023. So this file re-fits the same architecture as
model_v11/v12 on the pre-2023 pool alone — same formulas, its own league constants — and writes
`*_classic` outputs alongside the modern ones. The published 2023–2026 leaderboard is untouched.

Same architecture as v11 (Steal+ = net bases above a same-speed average; Burst = ground gained
above speed-predicted) and v12 (per-attempt SB-success AUROC on the lead distances). The v13
decision model has no classic counterpart — it is deliberately a 2023–2026 model.

Builds its own pool from the scraped drawer leads (Data/leads_cache/*_2015..2022.csv), the full
sprint-speed leaderboard (Data/sprint_speed.csv), and year-correct season SB/CS from the MLB Stats
API — NOT from the frozen DF_v7_SSSI, whose pre-2023 lead columns come from an incompatible source.

Run:  python3 Scripts/model_classic.py   (after: scrape_statcast.py discover --start 2015 --end 2022 --expand)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import model_v11 as M
import scrape_statcast as S

DATA, RESULTS, FIGS = M.DATA, M.RESULTS, M.ROOT / "Output" / "Figures"
CLASSIC = list(range(2015, 2023))               # 2015–2022
MIN_ATT = 5                                     # qualified runner-season
_FNAME = re.compile(r"(\d+)_(\d{4})\.csv$")


def _classic_attempts() -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(DATA / "leads_cache" / "*.csv")):
        m = _FNAME.search(f)
        if not m:
            continue
        rid, yr = int(m.group(1)), int(m.group(2))
        if yr not in CLASSIC:
            continue
        try:
            a = pd.read_csv(f)
        except Exception:
            continue
        if a.empty:
            continue
        if "runner_id" not in a.columns:
            a.insert(0, "runner_id", rid); a.insert(1, "season", yr)
        frames.append(a)
    att = pd.concat(frames, ignore_index=True)
    att = att[att["result"].isin(["SB", "CS"])].copy()
    att["y"] = (att["result"] == "SB").astype(int)
    att.to_csv(DATA / "Raw_Attempts_classic.csv", index=False)
    return att


def _classic_season(att: pd.DataFrame) -> pd.DataFrame:
    # year-correct season SB/CS totals (Stats API), one call per season
    rows = []
    for y in CLASSIC:
        for pid, a in S.fetch_sb_cs(y, y).items():
            rows.append({"runner_id": pid, "season": y, "player_name": a["name"],
                         "team": a["team"] or "", "SB": a["sb"], "CS": a["cs"],
                         "sb_attempts": a["sb"] + a["cs"]})
    season = pd.DataFrame(rows)

    sp = pd.read_csv(DATA / "sprint_speed.csv").rename(columns={"sprint_speed_all": "sprint_speed"})
    season = season.merge(sp[["runner_id", "season", "sprint_speed"]], on=["runner_id", "season"], how="left")

    # 'ground' is the same calculator-weighted blend v11 uses (see M.ground_weights): the classic
    # calculator's OWN fitted coefficients, from CLASSIC attempts, not the modern ones — the two
    # eras are fit separately throughout, and this stays consistent with that.
    classic_json = DATA / "v11_players_classic.json"
    w_lead, w_gain = M.ground_weights(classic_json)
    att = att.assign(_ground=w_lead * pd.to_numeric(att["lead_at_firstmove_ft"], errors="coerce")
                     + w_gain * pd.to_numeric(att["gain_to_release_ft"], errors="coerce"))
    ground = (att.groupby(["runner_id", "season"])
                 .agg(ground=("_ground", "mean"),
                      lead_rel=("lead_at_release_ft", "mean"),
                      tracked=("gain_to_release_ft", "count")).reset_index())
    season = season.merge(ground, on=["runner_id", "season"], how="inner")   # need leads for Burst
    season = season.merge(M.catcher_faced(att), on=["runner_id", "season"], how="left")

    season["net_sb"] = (season["SB"] - season["CS"]).astype(int)
    season["raw_succ"] = season["SB"] / season["sb_attempts"].clip(lower=1)
    season = (season[season["sb_attempts"] >= MIN_ATT]
              .dropna(subset=["sprint_speed", "ground"]).reset_index(drop=True))
    season.to_csv(DATA / "Raw_Season_classic.csv", index=False)
    return season


def _perattempt_auc(att: pd.DataFrame, season: pd.DataFrame, seed=42):
    """Per-attempt SB-success AUROC on the classic leads + runner skill (the v12 baseline, since the
    battery/pop-time context was not scraped for the classic era). Nested-OOF is not needed here —
    there is no target-encoded feature — but the SSSI runner mechanics are merged where available."""
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier
    except ImportError:
        return None
    df = att.merge(season[["runner_id", "season", "sprint_speed"]], on=["runner_id", "season"], how="left")
    sssi = pd.read_csv(RESULTS / "DF_v7_SSSI.csv")
    mech = ["jump_time", "accel_gap", "primary_lead", "lead_gain", "bolts"]
    keep = ["runner_id", "season"] + [c for c in mech if c in sssi.columns]
    df = df.merge(sssi[keep].drop_duplicates(["runner_id", "season"]), on=["runner_id", "season"], how="left")
    df["base_is_3b"] = (df["base"].astype(str) == "3B").astype(int)
    feats = [c for c in M.PA_LEAD_FEATS + ["base_is_3b", "sprint_speed"] + mech if c in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce").values
    y = df["y"].values
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(df))
    for tr, va in cv.split(df, y):
        m = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                          eval_metric="logloss", verbosity=0, random_state=seed, use_label_encoder=False)
        oof[va] = m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return roc_auc_score(y, oof), len(df)


def _classic_calculator(att, scored):
    """Fit the same 4-input logistic calculator (sprint speed, lead at first move, ground gained
    to release, catcher pop time) on CLASSIC attempts, so the site's era switch changes the odds
    model too rather than silently showing modern coefficients over classic players."""
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    a = att.merge(scored[["runner_id", "season", "sprint_speed", "burst_ft"]],
                  on=["runner_id", "season"], how="left")
    pop_path = DATA / "poptime.csv"                     # same 4-input spec as the modern calculator
    if pop_path.exists():
        pop = pd.read_csv(pop_path)[["catcher_id", "season", "pop_2b_sba"]] \
                .rename(columns={"pop_2b_sba": "pop_faced"})
        a = a.merge(pop, on=["catcher_id", "season"], how="left")
    else:
        a["pop_faced"] = np.nan
    a[M.SIMPLE_FEATS] = a[M.SIMPLE_FEATS].apply(pd.to_numeric, errors="coerce")

    pl = pd.to_numeric(a["lead_at_firstmove_ft"], errors="coerce")
    seas = a.assign(_pl=pl).dropna(subset=["_pl"]).groupby(["runner_id", "season"])["_pl"]
    means, counts = seas.mean(), seas.count()
    qual, grand = means[counts >= 15], float(pl.mean())
    ss_between = float((counts[counts >= 15] * (qual - grand) ** 2).sum())
    dq = a.assign(_pl=pl).dropna(subset=["_pl"])
    dq = dq[dq.set_index(["runner_id", "season"]).index.isin(qual.index)]
    ss_total = float(((dq["_pl"] - grand) ** 2).sum())
    primary_lead = {"mean": round(grand, 1), "runner_min": round(float(qual.min()), 1),
                    "runner_max": round(float(qual.max()), 1),
                    "within_pct": round(100 * (1 - ss_between / ss_total), 1),
                    "n_runner_seasons": int(len(qual))}

    a = a.dropna(subset=M.SIMPLE_FEATS).reset_index(drop=True)
    X, y = a[M.SIMPLE_FEATS].values, a["y"].values
    pipe = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=5000))
    auc = roc_auc_score(y, cross_val_predict(
        pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1])
    pipe.fit(X, y)
    lr = pipe.named_steps["logisticregression"]
    return {"intercept": float(lr.intercept_[0]),
            "coef": {k: float(v) for k, v in zip(M.SIMPLE_FEATS, lr.coef_[0])},
            "auc": round(float(auc), 4), "n": int(len(a)),
            "base_rate": round(float(y.mean()), 4), "primary_lead": primary_lead,
            "range": {f: [round(float(a[f].min()), 1), round(float(a[f].max()), 1),
                          round(float(a[f].median()), 1)] for f in M.SIMPLE_FEATS}}


def main():
    att = _classic_attempts()
    season = _classic_season(att)
    fit = M.fit_league(season)
    scored = M.score(season, fit)
    M.check_invariants(scored)

    lb_cols = ["runner_id", "season", "player_name", "team", "sprint_speed", "ground",
               "burst_ft", "SB", "CS", "net_sb", "raw_succ", "sb_attempts",
               "steal_plus", "steal_plus_pct", "burst_pct", "netspeed", "surplus"]
    board = scored[lb_cols].sort_values("steal_plus", ascending=False)
    board.to_csv(RESULTS / "DF_classic_leaderboard.csv", index=False)

    # validation: same honest tests as v11, computed on the classic pool
    rows = []
    def add(q, metric, v, note=""):
        rows.append({"question": q, "metric": metric, "value": round(v, 3), "note": note})
    for col, lab in [("net_sb", "Net Bases"), ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        add("corr with sprint speed", lab, M.pearson_r(scored[col], scored["sprint_speed"]))
    add("describes net steals (same season)", "Steal+", M.pearson_r(scored["steal_plus"], scored["net_sb"]))
    add("describes success rate (same season)", "Steal+", M.pearson_r(scored["steal_plus"], scored["raw_succ"]))
    for col, lab in [("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        r, n = M.year_over_year_corr(scored, col, col); add("year-over-year self-stability", lab, r, f"n={n}")
    add("Steal+ vs Burst (independent lenses)", "Steal+ x Burst",
        M.pearson_r(scored["steal_plus"], scored["burst_ft"]))
    pd.DataFrame(rows).to_csv(RESULTS / "DF_classic_validation.csv", index=False)

    # ── web payload for the site's era toggle (same record schema as v11_players.json) ──
    w = scored.copy()
    tmc = DATA / "team_map_classic.csv"
    if tmc.exists():
        w = w.merge(pd.read_csv(tmc), on=["runner_id", "season"], how="left", suffixes=("", "_tm"))
        w["team"] = w["team_tm"].fillna(w.get("team", "")).fillna("")
    w["speed_pct"] = M.percentile_rank(w["sprint_speed"], w["sprint_speed"].dropna().values)
    w["ground_pct"] = M.percentile_rank(w["ground"], w["ground"].dropna().values)
    sssi = pd.read_csv(RESULTS / "DF_v7_SSSI.csv")
    jt = sssi[["runner_id", "season", "jump_time"]].drop_duplicates(["runner_id", "season"]) \
        if "jump_time" in sssi.columns else None
    w = w.merge(jt, on=["runner_id", "season"], how="left") if jt is not None else w.assign(jump_time=np.nan)
    w["jump_pct"] = (M.percentile_rank(w["jump_time"], w["jump_time"].dropna().values)
                     if w["jump_time"].notna().any() else np.nan)
    payload = {"meta": {"version": "classic", "era": "2015-2022", "n_player_seasons": int(len(w)),
                        "league_success_pct": round(fit["league"] * 100, 1),
                        "p_speed_fit": {"a0": fit["a0"], "a1": fit["a1"]},
                        "ground_fit": {"b0": fit["b0"], "b1": fit["b1"]}},
               "validation": pd.DataFrame(rows).to_dict(orient="records"),
               "players": M.to_records(w),
               "success_model": _classic_calculator(att, scored)}
    (DATA / "v11_players_classic.json").write_text(json.dumps(payload, separators=(",", ":")))
    M.sync_site_payload(payload, "const PAYLOAD_CLASSIC = ")
    print(f"[write] v11_players_classic.json  ({len(w)} players, "
          f"{int((w['team'].fillna('').astype(str).str.len() > 0).sum())} with team)")

    pa = _perattempt_auc(att, season, )
    fit_out = {"era": "2015-2022", "n_runner_seasons": int(len(scored)),
               "n_attempts": int(len(att)), "league_success_pct": round(fit["league"] * 100, 1),
               "p_speed_fit": {"a0": fit["a0"], "a1": fit["a1"]},
               "ground_fit": {"b0": fit["b0"], "b1": fit["b1"]},
               "perattempt_auc": None if pa is None else round(pa[0], 4),
               "perattempt_n": None if pa is None else pa[1]}
    (RESULTS / "DF_classic_fit.json").write_text(json.dumps(fit_out, indent=1))

    # ── report to stdout, side by side with the modern fit ──
    print(f"CLASSIC era 2015-2022 | {len(scored)} runner-seasons | {len(att):,} tracked attempts")
    print(f"  league SB% {fit['league']*100:.1f}  |  p_speed = {fit['a0']:.3f} + {fit['a1']:.4f}*speed"
          f"  |  ground = {fit['b0']:.2f} + {fit['b1']:.3f}*speed")
    if pa:
        print(f"  per-attempt SB-success AUROC (leads + runner skill): {pa[0]:.4f}  (n={pa[1]:,})")
    try:
        modern = json.loads((DATA / "v11_players.json").read_text())["meta"]
        print(f"\n  vs MODERN 2023-2026: league SB% {modern['league_success_pct']}  |  "
              f"p_speed = {modern['p_speed_fit']['a0']:.3f} + {modern['p_speed_fit']['a1']:.4f}*speed  |  "
              f"ground = {modern['ground_fit']['b0']:.2f} + {modern['ground_fit']['b1']:.3f}*speed")
        print("  (the rule change shows up here: post-2023 success and leads both run higher)")
    except Exception:
        pass
    print("\nCLASSIC Steal+ leaderboard (top 12):")
    show = ["player_name", "season", "sprint_speed", "SB", "CS", "net_sb", "steal_plus", "burst_ft"]
    print(board.head(12)[show].round(2).to_string(index=False))
    print("\nCLASSIC validation:")
    print(pd.DataFrame(rows).to_string(index=False))
    return board


if __name__ == "__main__":
    main()
