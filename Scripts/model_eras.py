#!/usr/bin/env python3
"""
model_eras.py — v11's analysis, run three ways: PRE-2023, POST-2023, and POOLED (2016-2026),
plus the ingestion audit and the n=5 verification harness.

PURELY ADDITIVE. It imports model_v11's own fit_league() / score() / check_invariants() and runs
them on three different pools. Nothing in model_v11, model_classic, model_v12 or model_v13 is
modified, and the published 2023-2026 leaderboard is untouched — this only adds a comparison.

  PRE    2016-2022   pre rule change            (2015 has no tracked lead data at all)
  POST   2023-2026   post rule change           (identical to the published v11 pool)
  ALL    2016-2026   pooled across the change   (deliberately included to show it is a bad idea)

WHY POOLING IS SHOWN BUT NOT RECOMMENDED. The 2023 package (disengagement limit, bigger bases,
pitch clock) moved both baselines. A single pooled fit splits the difference and then scores every
runner against a league that never existed. The pooled column is reported so that claim is
falsifiable rather than asserted.

THE n=5 VERIFICATION (why this file exists as much as the era comparison). score() is written to be
a pure function of the league `fit`, so scoring 5 random runner-seasons ON THEIR OWN must produce
byte-identical numbers to scoring those same 5 inside the full pool. If any statistic were computed
within-batch — a percentile ranked against whoever happens to be in the dataframe, a mean taken over
the scored rows — that equality would break. Running it over many random draws of 5 is therefore a
standing leak detector, not a formality.

LEAK AUDIT. Every column that reaches a feature list is checked against a deny-list of
outcome-derived fields, and the decomposition invariant is asserted on every subsample.

Run:  python3 Scripts/model_eras.py
Writes Output/Results/DF_eras_*.csv and Output/Figures/Fig_eras_*.png
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path

import numpy as np
import pandas as pd

import model_v11 as M

ROOT, DATA, RESULTS = M.ROOT, M.DATA, M.RESULTS
FIGS = ROOT / "Output" / "Figures"

PRE  = list(range(2016, 2023))
POST = list(range(2023, 2027))
ERAS = {"PRE 2016-2022": PRE, "POST 2023-2026": POST, "ALL 2016-2026": PRE + POST}

# Columns the league fit and the scorer are allowed to see. Anything outcome-derived that is not
# on this list must never reach a feature: sb_residual, real_sb_pct, steal_plus (legacy), SSSI_v7,
# sb_run_value, xsb_outcome, y. SB/CS/net_sb/raw_succ ARE outcomes, but they are the TARGET of
# Steal+ by definition — they are allowed on the left-hand side, never as a predictor of themselves.
POOL_COLS = ["runner_id", "season", "player_name", "sprint_speed", "ground",
             "SB", "CS", "sb_attempts", "net_sb", "raw_succ"]
LEAK_DENY = ["sb_residual", "real_sb_pct", "expected_sb_pct", "SSSI_v7", "rank_v7",
             "sb_run_value", "xsb_outcome", "z_net_sb", "steal_plus_legacy", "y", "exp_net"]


# ── pools: built offline from what is already on disk ────────────────────────
MIN_ATT = 10        # common qualification gate across ALL eras — see _pool()


def _pool(seasons, min_att: int = MIN_ATT) -> pd.DataFrame:
    """One runner-season pool for the given seasons, in the exact shape fit_league/score expect.

    A COMMON attempt gate is applied to every era. This matters: the classic pool is built at >=5
    attempts while the published modern pool inherits a >=10 gate from the frozen SSSI file, so
    comparing them untouched would put 46% low-volume runner-seasons in PRE and none in POST — an
    apples-to-oranges era comparison. 10 is used because it leaves the published POST pool
    unchanged."""
    frames = []
    post = M.load_era()                                   # 2023-2026, the published path
    frames.append(post[[c for c in POOL_COLS if c in post.columns]])
    cls_path = DATA / "Raw_Season_classic.csv"
    if cls_path.exists():
        c = pd.read_csv(cls_path)
        frames.append(c[[col for col in POOL_COLS if col in c.columns]])
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["season"].isin(seasons)]
    pool = pool[pool["sb_attempts"] >= min_att]
    pool = pool.dropna(subset=["sprint_speed", "ground", "raw_succ"]).reset_index(drop=True)
    return pool


# ── the n=5 verification harness ─────────────────────────────────────────────
def verify_n5(pool: pd.DataFrame, fit: dict, n: int = 5, trials: int = 200, seed: int = 0):
    """Score n random runner-seasons ALONE and compare to the same rows scored inside the full
    pool. score() is pure given `fit`, so these must be identical to floating-point exactness.
    Any within-batch statistic (a percentile ranked against the current dataframe, a mean over the
    scored rows) would show up here as a non-zero difference."""
    full = M.score(pool, fit).set_index(["runner_id", "season"])
    rng = np.random.default_rng(seed)
    cols = ["steal_plus", "burst_ft", "netspeed", "steal_plus_pct", "burst_pct"]
    worst, worst_col = 0.0, ""
    for _ in range(trials):
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        sub = M.score(pool.iloc[idx], fit).set_index(["runner_id", "season"])
        M.check_invariants(sub.reset_index())             # decomposition must close on every draw
        for c in cols:
            d = float((sub[c] - full.loc[sub.index, c]).abs().max())
            if d > worst:
                worst, worst_col = d, c
    return {"trials": trials, "n_per_trial": n, "max_abs_diff": worst, "worst_column": worst_col}


def verify_detector(pool: pd.DataFrame, fit: dict) -> dict:
    """NEGATIVE CONTROL — proves verify_n5 can actually fail.

    A test that always returns 0.0 is worthless unless you show it would catch the thing it is
    meant to catch. Here score() is temporarily swapped for a deliberately leaky version that ranks
    percentiles WITHIN the batch instead of against the frozen league reference. The n=5 check must
    blow up on it; if it does not, the check itself is broken and the clean result means nothing."""
    real = M.score

    def leaky(rows, f):
        d = real(rows, f)
        d["steal_plus_pct"] = d["steal_plus"].rank(pct=True) * 100     # batch-dependent on purpose
        return d

    M.score = leaky
    try:
        caught = verify_n5(pool, fit, trials=25)["max_abs_diff"]
    finally:
        M.score = real                                                 # always restore
    clean = verify_n5(pool, fit, trials=25)["max_abs_diff"]
    assert caught > 1.0, "the n=5 check FAILED to detect a deliberately leaky scorer"
    assert clean == 0.0, "score() is not clean after restoring the real implementation"
    return {"leaky_scorer_diff": round(float(caught), 3), "real_scorer_diff": float(clean)}


def vif_audit(max_vif: float = 10.0) -> pd.DataFrame:
    """STANDING COLLINEARITY GUARD.

    v12 once shipped three lead features where one was the exact sum of the other two
    (lead_at_release = lead_at_firstmove + gain_to_release, R^2 = 0.999895), giving VIFs of
    1,588 / 5,836 / 9,532. XGBoost's predictions were unaffected, so nothing failed — but feature
    importance was split arbitrarily across perfectly dependent columns, which quietly made the
    importance chart unquotable. Nothing in the pipeline noticed. This does."""
    from sklearn.linear_model import LinearRegression
    import model_v12 as V12

    df, _ = V12.load()
    sets = {
        "v12 leads": M.PA_LEAD_FEATS,
        "v12 shipped": (M.PA_LEAD_FEATS + ["base_is_3b"]
                        + [c for c in M.PA_RUNNER_FEATS if c in df.columns]
                        + V12.BATTERY_FEATS + V12.SAFE_SITUATION_FEATS + V12.ARM_FEATS),
        "v11 calculator": M.SIMPLE_FEATS,
    }
    calc = pd.read_csv(DATA / "Raw_Attempts.csv")
    calc = calc[calc["result"].isin(["SB", "CS"])]
    pop = DATA / "poptime.csv"
    if pop.exists():
        calc = calc.merge(pd.read_csv(pop)[["catcher_id", "season", "pop_2b_sba"]]
                          .rename(columns={"pop_2b_sba": "pop_faced"}),
                          on=["catcher_id", "season"], how="left")
    sp = pd.read_csv(DATA / "sprint_speed.csv").rename(columns={"sprint_speed_all": "sprint_speed"})
    calc = calc.merge(sp[["runner_id", "season", "sprint_speed"]], on=["runner_id", "season"], how="left")

    rows = []
    for name, feats in sets.items():
        src = calc if name == "v11 calculator" else df
        cols = [c for c in feats if c in src.columns]
        sub = src[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 50 or len(cols) < 2:
            continue
        worst, worst_col = 0.0, ""
        for c in cols:
            X = sub[[x for x in cols if x != c]].values
            r2 = LinearRegression().fit(X, sub[c].values).score(X, sub[c].values)
            v = float("inf") if r2 >= 1 - 1e-12 else 1 / (1 - r2)
            if v > worst:
                worst, worst_col = v, c
        rows.append({"feature_set": name, "n_features": len(cols), "n_rows": len(sub),
                     "max_vif": round(worst, 2), "worst_feature": worst_col,
                     "passes": bool(worst < max_vif)})
    out = pd.DataFrame(rows)
    bad = out[~out.passes]
    assert bad.empty, ("COLLINEARITY: " + "; ".join(
        f"{r.feature_set} has VIF {r.max_vif:.0f} on {r.worst_feature}" for r in bad.itertuples()))
    return out


def leak_audit(pool: pd.DataFrame) -> pd.DataFrame:
    """Assert no outcome-derived column smuggled itself into the pool the fit sees."""
    rows = []
    for c in LEAK_DENY:
        rows.append({"denied_column": c, "present_in_pool": bool(c in pool.columns)})
    bad = [r["denied_column"] for r in rows if r["present_in_pool"]]
    assert not bad, f"LEAK: outcome-derived column(s) reached the pool: {bad}"
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────────
def fig_ingestion(pools):
    """Where every row comes from, season by season — the ingestion audit."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    att = pd.read_csv(DATA / "Raw_Attempts.csv")
    cls = DATA / "Raw_Attempts_classic.csv"
    if cls.exists():
        att = pd.concat([att, pd.read_csv(cls)], ignore_index=True)
    att = att[att["result"].isin(["SB", "CS"])]
    a = att.groupby("season").size()
    allp = pools["ALL 2016-2026"]
    r = allp.groupby("season").size()

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=150)
    yrs = sorted(set(a.index) | set(r.index))
    col = ["#9AA0A6" if y < 2023 else "#2F6FB0" for y in yrs]
    ax[0].bar([str(y) for y in yrs], [a.get(y, 0) for y in yrs], color=col)
    ax[0].set_title("Tracked steal attempts ingested, by season", fontsize=11, fontweight="bold")
    ax[0].set_ylabel("attempts (SB + CS)")
    ax[1].bar([str(y) for y in yrs], [r.get(y, 0) for y in yrs], color=col)
    ax[1].set_title("Qualified runner-seasons entering the fit", fontsize=11, fontweight="bold")
    ax[1].set_ylabel("runner-seasons")
    for x in ax:
        x.tick_params(axis="x", rotation=45, labelsize=8)
        for sp in ("top", "right"): x.spines[sp].set_visible(False)
    fig.text(0.5, -0.02, "grey = pre-2023 rules · blue = post-2023 rules · 2015 carries no tracked "
             "lead data and 2020 was the shortened season", ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(); fig.savefig(FIGS / "Fig_eras_ingest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_fits(fits, pools):
    """THE TWO LEAGUE BASELINES — the only two lines the whole model subtracts from.

    An earlier version of this figure drew three bare lines with axis-name titles and no
    annotation, which left the reader with no way to answer the only two questions that matter:
    what is this for, and what does a slope mean. Both are now written onto the figure itself.

    WHAT IT IS FOR. Steal+ and Burst are each "actual minus expected", and these are the two
    expectation lines. Neither metric can be read without seeing them.

      LEFT   ground = b0 + b1 x speed        -> Burst  = ground gained  -  this line
      RIGHT  p      = a0 + a1 x speed        -> Steal+ = net bases      -  attempts x (2p - 1)

    WHAT THE SLOPES MEAN, in words, because that was the actual complaint.
      b1 (left)  is NEGATIVE: every +1 ft/s of sprint speed comes with ~0.89 ft LESS ground gained
                 post-2023. Faster runners take a shorter lead and break later relative to the
                 pitch, so raw ground gained is partly a slow-runner statistic. That is exactly
                 why Burst subtracts this line instead of quoting ground gained directly.
      a1 (right) is the price of speed: +1 ft/s bought +2.20 percentage points of success before
                 2023 and only +1.14 after — the rules HALVED what raw wheels are worth, while
                 lifting everybody's floor (intercept 76.4% -> 80.4% at league mean). This is the
                 single strongest argument in the project for not pooling the eras.

    Each panel is drawn over the actual runner-seasons it was fit on, so the line is visibly a
    summary of data rather than an assertion."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"PRE 2016-2022": "#9AA0A6", "POST 2023-2026": "#2F6FB0", "ALL 2016-2026": "#C0392B"}
    PRE_K, POST_K, ALL_K = "PRE 2016-2022", "POST 2023-2026", "ALL 2016-2026"
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.4), dpi=150)
    xs = np.linspace(24, 31, 50)

    # the data underneath the lines: post-2023 runner-seasons, so "the line" is visibly a fit
    post = pools[POST_K]
    ax[0].scatter(post.sprint_speed, post.ground, s=9, color="#2F6FB0", alpha=0.16,
                  lw=0, zorder=1, label="_nolegend_")

    for k, f in fits.items():
        ls = "--" if k is ALL_K or k == ALL_K else "-"
        ax[0].plot(xs, f["b0"] + f["b1"] * xs, lw=2.6, color=colors[k], ls=ls, zorder=3,
                   label=f"{k}   slope {f['b1']:+.2f} ft per ft/s")
        ax[1].plot(xs, 100 * (f["a0"] + f["a1"] * xs), lw=2.6, color=colors[k], ls=ls, zorder=3,
                   label=f"{k}   slope {100*f['a1']:+.2f} pts per ft/s")

    # ── LEFT: what Burst is measured against ────────────────────────────────
    fp = fits[POST_K]
    g25, g30 = fp["b0"] + fp["b1"] * 25, fp["b0"] + fp["b1"] * 30
    ax[0].set_title("A · The line Burst is measured from\n"
                    "Slower runners gain MORE ground — so raw ground gained must be speed-adjusted",
                    fontsize=10.5, fontweight="bold", color="#0C2340")
    for x0, y0, lab in [(25, g25, f"25 ft/s expects {g25:.1f} ft"),
                        (30, g30, f"30 ft/s expects {g30:.1f} ft")]:
        ax[0].plot([x0], [y0], "o", ms=7, color="#0C2340", zorder=5)
        ax[0].annotate(lab, (x0, y0), textcoords="offset points", xytext=(8, 12),
                       fontsize=9, fontweight="bold", color="#0C2340")
    # one runner drawn as the vertical gap he actually is
    ax[0].annotate("", xy=(26.6, g25 + fp["b1"] * 1.6 + 2.6), xytext=(26.6, g25 + fp["b1"] * 1.6),
                   arrowprops=dict(arrowstyle="<->", lw=1.8, color="#C0392B"))
    ax[0].annotate("BURST = this gap\n(actual − expected)", (26.75, g25 + fp["b1"] * 1.6 + 1.3),
                   fontsize=9, fontweight="bold", color="#C0392B", va="center")
    ax[0].set_xlabel("sprint speed (ft/s)"); ax[0].set_ylabel("ground gained on the pitch (ft)")
    ax[0].set_ylim(0, 30)

    # ── RIGHT: what the rule change did to the price of speed ───────────────
    a1p, a1o = 100 * fits[PRE_K]["a1"], 100 * fits[POST_K]["a1"]
    ax[1].set_title("B · What one ft/s of sprint speed buys\n"
                    f"The 2023 rules HALVED it: {a1p:.2f} → {a1o:.2f} points of success per ft/s",
                    fontsize=10.5, fontweight="bold", color="#0C2340")
    for k, xa, off in [(PRE_K, 27.2, (0, -20)), (POST_K, 26.4, (0, 16))]:
        f = fits[k]
        ax[1].annotate(f"+{100*f['a1']:.2f} pts per ft/s", (xa, 100 * (f["a0"] + f["a1"] * xa)),
                       textcoords="offset points", xytext=off, ha="center",
                       fontsize=9.5, fontweight="bold", color=colors[k])
    ax[1].set_xlabel("sprint speed (ft/s)"); ax[1].set_ylabel("expected success rate (%)")
    lo = min(100 * (f["a0"] + f["a1"] * 24) for f in fits.values())
    hi = max(100 * (f["a0"] + f["a1"] * 31) for f in fits.values())
    ax[1].set_ylim(lo - 0.12 * (hi - lo), hi + 0.06 * (hi - lo))

    for x in ax:
        x.legend(fontsize=8.5, frameon=False, loc="lower left")
        x.grid(alpha=0.13, lw=0.7)
        for sp in ("top", "right"): x.spines[sp].set_visible(False)

    fig.text(0.5, -0.055,
             "HOW TO READ A SLOPE HERE.  Left: the line runs DOWNWARD, so a fast runner is "
             "expected to gain less ground than a slow one; Burst is the vertical distance a "
             "runner sits above it, which is why Burst\ncorrelates 0.00 with speed while raw "
             "ground gained correlates −0.49.  Right: the line's steepness IS the value of raw "
             "speed, and it is half as steep after 2023 while sitting higher —\nthe league got "
             "safer and speed got less decisive.  The pooled red line splits the two eras and "
             "describes neither, which is why PRE and POST are fit separately.",
             ha="center", fontsize=8.6, color="#444")
    fig.tight_layout(); fig.savefig(FIGS / "Fig_eras_fits.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_verify(ver, det):
    """The n=5 purity check, drawn against its own negative control — bars at exactly zero are
    meaningless on their own, so the deliberately leaky scorer is charted beside them."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = [f"{k}\n(real scorer)" for k in ver] + ["NEGATIVE CONTROL\n(leaky scorer)"]
    vals = [ver[k]["max_abs_diff"] for k in ver] + [det["leaky_scorer_diff"]]
    cols = ["#1A7F47"] * len(ver) + ["#C0392B"]
    fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=150)
    ax.barh(labs, vals, color=cols, height=0.6)
    ax.set_xlim(0, max(vals) * 1.25)
    for i, v in enumerate(vals):
        ax.text(max(vals) * 0.012, i, "0.00  — identical to the last bit" if v == 0
                else f"{v:.1f}  — caught", va="center",
                fontsize=10, fontweight="bold", color="#fff" if v > 0 else "#1A7F47")
    ax.set_xlabel("max |value scored alone  −  value scored inside the full pool|")
    ax.set_title("n=5 purity check: 200 random draws of 5 runner-seasons, every metric",
                 fontsize=11.5, fontweight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.text(0.5, -0.08, "Scoring 5 runners alone gives exactly the numbers they get in the full "
             "pool, so nothing is computed against the batch.\nThe red bar is the same test run on a "
             "scorer rigged to rank percentiles within the batch — it fails loudly, which is what "
             "makes the green zeros meaningful.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(); fig.savefig(FIGS / "Fig_eras_verify.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def pooling_cost(pools, fits) -> pd.DataFrame:
    """What it costs to score modern runners with the POOLED fit instead of their own era's.
    This is the falsifiable version of "don't pool the eras"."""
    rows = []
    for era_key, other in [("POST 2023-2026", "ALL 2016-2026"), ("PRE 2016-2022", "ALL 2016-2026")]:
        pool = pools[era_key]
        own = M.score(pool, fits[era_key]).set_index(["runner_id", "season"])
        alt = M.score(pool, fits[other]).set_index(["runner_id", "season"])
        rows.append({
            "scored_pool": era_key, "fit_used_instead": other,
            "steal_plus_mean_shift": round(float((alt.steal_plus - own.steal_plus).mean()), 3),
            "steal_plus_max_shift": round(float((alt.steal_plus - own.steal_plus).abs().max()), 3),
            "burst_mean_shift": round(float((alt.burst_ft - own.burst_ft).mean()), 3),
            "spearman_steal_plus": round(float(own.steal_plus.corr(alt.steal_plus, method="spearman")), 4),
            "corr_burst_speed_own": round(M.pearson_r(own.burst_ft, own.sprint_speed), 3),
            "corr_burst_speed_pooled": round(M.pearson_r(alt.burst_ft, alt.sprint_speed), 3),
        })
    return pd.DataFrame(rows)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    pools = {k: _pool(v) for k, v in ERAS.items()}
    fits, ver, rows = {}, {}, []

    for k, pool in pools.items():
        leak_audit(pool)
        fit = M.fit_league(pool)
        scored = M.score(pool, fit)
        M.check_invariants(scored)
        fits[k] = fit
        ver[k] = verify_n5(pool, fit)
        rows.append({
            "era": k, "runner_seasons": len(pool), "min_attempts_gate": MIN_ATT,
            "seasons": f"{int(pool.season.min())}-{int(pool.season.max())}",
            "league_SB_pct": round(100 * fit["league"], 1),
            "p_speed_a0": round(fit["a0"], 4), "p_speed_a1": round(fit["a1"], 5),
            "ground_b0": round(fit["b0"], 3), "ground_b1": round(fit["b1"], 4),
            "ground_at_25ftps": round(fit["b0"] + fit["b1"] * 25, 2),
            "ground_at_30ftps": round(fit["b0"] + fit["b1"] * 30, 2),
            "corr_steal_plus_speed": round(M.pearson_r(scored.steal_plus, scored.sprint_speed), 3),
            "corr_burst_speed": round(M.pearson_r(scored.burst_ft, scored.sprint_speed), 3),
            "corr_ground_speed": round(M.pearson_r(scored.ground, scored.sprint_speed), 3),
            "n5_max_abs_diff": ver[k]["max_abs_diff"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "DF_eras_comparison.csv", index=False)
    pd.DataFrame(ver).T.reset_index(names="era").to_csv(RESULTS / "DF_eras_verification.csv", index=False)
    leak_audit(pools["ALL 2016-2026"]).to_csv(RESULTS / "DF_eras_leak_audit.csv", index=False)

    det = verify_detector(pools["POST 2023-2026"], fits["POST 2023-2026"])
    vifs = vif_audit()
    vifs.to_csv(RESULTS / "DF_eras_vif_audit.csv", index=False)

    cost = pooling_cost(pools, fits)
    cost.to_csv(RESULTS / "DF_eras_pooling_cost.csv", index=False)

    fig_ingestion(pools); fig_fits(fits, pools); fig_verify(ver, det)

    pd.set_option("display.width", 200)
    print("=== THE SAME v11 MODEL, FIT THREE WAYS ===")
    print(out[["era", "seasons", "runner_seasons", "league_SB_pct", "p_speed_a0", "p_speed_a1",
               "ground_b0", "ground_b1"]].to_string(index=False))
    print("\n=== WHAT THE RULE CHANGE DID (ground gained predicted at two speeds) ===")
    print(out[["era", "ground_at_25ftps", "ground_at_30ftps", "corr_ground_speed"]].to_string(index=False))
    print("\n=== SPEED-NEUTRALITY HOLDS IN EVERY ERA (should be ~0) ===")
    print(out[["era", "corr_steal_plus_speed", "corr_burst_speed"]].to_string(index=False))
    print("\n=== n=5 VERIFICATION — 200 random draws of 5 runner-seasons per era ===")
    for k, v in ver.items():
        ok = "PASS" if v["max_abs_diff"] < 1e-9 else "FAIL"
        print(f"  {k:16s} max|alone − in-pool| = {v['max_abs_diff']:.2e}  over "
              f"{v['trials']} draws x {v['n_per_trial']} rows   [{ok}]")
    print(f"\n=== NEGATIVE CONTROL — does the n=5 check actually work? ===")
    print(f"  deliberately leaky scorer (within-batch percentiles) -> {det['leaky_scorer_diff']:.3f}  "
          f"[correctly caught]")
    print(f"  real scorer, restored                                -> {det['real_scorer_diff']:.1f}  "
          f"[clean]")
    print("\n=== WHAT POOLING WOULD COST (scoring an era with the pooled fit) ===")
    print(cost.to_string(index=False))
    r = cost[cost.scored_pool == "POST 2023-2026"].iloc[0]
    share = 100 * len(pools["PRE 2016-2022"]) / len(pools["ALL 2016-2026"])
    print(f"  -> ranks barely move (Spearman {r.spearman_steal_plus:.3f}), but Burst STOPS being "
          f"speed-neutral")
    print(f"     ({r.corr_burst_speed_own:+.3f} -> {r.corr_burst_speed_pooled:+.3f} with sprint speed) "
          f"and modern Steal+ shifts {r.steal_plus_mean_shift:+.2f} on average,")
    print(f"     because {share:.0f}% of the pooled rows are pre-2023. Shown for completeness, "
          f"not recommended.")

    print("\n=== COLLINEARITY GUARD (max VIF must stay under 10) ===")
    print(vifs.to_string(index=False))

    print("\n=== LEAK AUDIT === no outcome-derived column reached any pool "
          f"({len(LEAK_DENY)} denied columns checked, all absent)")
    return out


if __name__ == "__main__":
    main()
