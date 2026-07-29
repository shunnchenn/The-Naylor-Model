#!/usr/bin/env python3
"""
model_v11.py — the Naylor Model v11 metric suite (single source of truth).

Written in two stages a coder can test on a tiny sample before trusting the full pool:

  load_era()            -> the 2023-26 runner-season pool, with measured ground covered
                           folded in from the per-attempt data.
  fit_league(era)       -> league CONSTANTS (the ground~speed regression, the league SB
                           rate, and the Steal+/Burst reference distributions used for
                           percentiles). Depends on the whole population — fit ONCE.
  score(rows, fit)      -> every v11 metric for ANY set of runner-seasons. PURE given
                           `fit`: scoring 1, 5, or 408 rows gives byte-identical per-row
                           values. That is what lets us smoke-test the pipeline on n=1,
                           then n=5, then run the full pool and get the SAME numbers for
                           the test rows (asserted in the notebook).

Two independent metrics (2023-26 lead-tracking era, clear units) — kept SEPARATE, not
averaged, because they answer different questions:
  Steal+   HEADLINE. Net bases (SB - CS) above what an average runner of his sprint speed
           would produce over the same attempts. VOLUME-AWARE and speed-normalized: 30 bags
           from a 24 ft/s body counts. Best single answer to "who steals well" — same-season
           r≈0.88 with success rate.       — "the bases your skill adds above your wheels"
  Burst    A DIFFERENT lens (near-zero corr with Steal+): feet of ground the runner gains off
           the base from the pitcher's first move until the pitcher's RELEASE (his secondary
           lead) ABOVE what his speed predicts. The endpoint is release, not the ball reaching
           the catcher — the feed has no ball-arrival timestamp, so that is not observable. The coachable process / upside;
           replaces v10's Statcast 'SB Run Value'.  — "the coachable jump/lead, before the throw"
           'Ground' is a WEIGHTED blend of the calculator's own two lead features — lead at first
           move and gain to release — weighted by the calculator's fitted coefficients (see
           ground_weights()), so Burst is built from the same two quantities the v14 calculator
           runs on, weighted the way the calculator itself weighs them.
  (A v10 'Steal Grade' averaged the two percentiles; dropped in v11 — validation showed it
   predicts net steals / success no better than Steal+ alone and mis-ranks pure producers.)

Decomposition (exact):  Net Bases = NetSpeed + Steal+   (Steal+ IS the surplus term)
  p_speed  = a0 + a1*sprint_speed                  league success rate expected from SPEED ONLY
  NetSpeed = sb_attempts*(2*p_speed - 1)           net bases your raw speed alone buys
  Steal+   = Net Bases - NetSpeed                  net bases your SKILL adds above your speed

Every number in the report/web app is validated empirically by validate() below:
  - speed-independence (corr with sprint speed ~ 0 for the v11 metrics);
  - the honest year T -> T+1 tests (Net Bases forecasts next-year VOLUME best; Steal+
    forecasts next-year SKILL best; Burst is the most repeatable TECHNIQUE).

Also fits the PER-ATTEMPT SB-success model (run_perattempt) on the ~11k individual
attempts — the grain that actually decides a steal — as the quantitative proof that the
per-pitch lead distances (which drive Burst) carry the signal (5-fold OOF AUC ~0.74).

No network. Reads Data/Raw_Season.csv + Data/Raw_Attempts.csv.
Writes Data/v11_players.json, Output/Results/DF_v11_leaderboard.csv + DF_v11_validation.csv,
Output/Results/DF_perattempt_AUC.csv + DF_perattempt_Importance.csv, and Fig_AUC/Fig_Importance.png.
Usage:  python3 Scripts/model_v11.py     (per-attempt stage needs xgboost+sklearn; skipped if absent)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "Data"
RESULTS = ROOT / "Output" / "Results"
RESULTS.mkdir(parents=True, exist_ok=True)

ERA_MIN    = 2023           # lead-tracking era start
TEST_IDS   = [647304, 665742]   # Naylor, Soto — the two pipeline-test subjects


# ── small helpers ───────────────────────────────────────────────────────────
def pearson_r(x, y) -> float:
    """Pearson correlation coefficient between two equal-length series, ignoring rows
    where either is NaN. Returns NaN if fewer than 6 usable pairs remain."""
    paired = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    return float(np.corrcoef(paired.x, paired.y)[0, 1]) if len(paired) > 5 else float("nan")

def percentile_rank(values, reference) -> np.ndarray:
    """For each value, its 0-100 percentile against a FIXED reference distribution.
    Reference-based (not rank-within-the-batch), so a runner's percentile does not
    change with how many other runners happen to be scored alongside him."""
    ref = np.sort(np.asarray(reference, float)); n = len(ref)
    return np.array([np.nan if pd.isna(v) else np.searchsorted(ref, v, "right") / n * 100
                     for v in values])

def year_over_year_corr(df, metric_col, next_year_col, min_att=8):
    """Correlate a metric in season T with an outcome in season T+1 for the SAME runner
    (does metric(T) predict next_year_col(T+1)?). Pairs each qualified runner-season to
    that runner's following season. Returns (correlation, number of paired seasons)."""
    cols = ["runner_id", "season"] + list(dict.fromkeys([metric_col, next_year_col]))
    qualified = df[df.sb_attempts >= min_att][cols].dropna()
    year_t  = qualified[["runner_id", "season", metric_col]].rename(columns={metric_col: "x"}).copy()
    year_t["next_season"] = year_t.season + 1                       # line season T up with T+1
    year_t1 = qualified[["runner_id", "season", next_year_col]].rename(
        columns={next_year_col: "y", "season": "next_season"})
    paired = year_t.merge(year_t1, on=["runner_id", "next_season"])
    return pearson_r(paired["x"].values, paired["y"].values), len(paired)


def catcher_faced(attempts: pd.DataFrame) -> pd.DataFrame:
    """Average catcher POP TIME and ARM STRENGTH faced, per runner-season — the opponent-
    difficulty context behind a steal. Pop time is the catch-to-second-base transfer+throw; arm
    is max-effort velocity. Both come from Savant's poptime leaderboard (2015-2026) joined on the
    catcher actually behind the plate for each tracked attempt."""
    pop_path = DATA / "poptime.csv"
    if not pop_path.exists():
        return pd.DataFrame(columns=["runner_id", "season", "pop_faced", "arm_faced"])
    pop = pd.read_csv(pop_path)[["catcher_id", "season", "pop_2b_sba", "maxeff_arm_2b_3b_sba"]]
    a = attempts.merge(pop, on=["catcher_id", "season"], how="left")
    return (a.groupby(["runner_id", "season"])
             .agg(pop_faced=("pop_2b_sba", "mean"),
                  arm_faced=("maxeff_arm_2b_3b_sba", "mean")).reset_index())


def sync_site_payload(payload: dict, marker: str) -> None:
    """Write a payload into docs/index.html in place of the JSON literal after `marker`.
    Keeps the published site in lockstep with the model instead of relying on a manual swap
    (the main payload silently went stale once when catcher pop/arm were added)."""
    site = ROOT / "docs" / "index.html"
    if not site.exists():
        return
    h = site.read_text(encoding="utf-8")
    if marker not in h:
        return
    i = h.index(marker) + len(marker)
    if h[i] != "{":
        return
    depth, j, instr, esc = 0, i, False, False
    while j < len(h):
        c = h[j]
        if instr:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': instr = False
        else:
            if c == '"': instr = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    j += 1; break
        j += 1
    site.write_text(h[:i] + json.dumps(payload, separators=(",", ":")) + h[j:], encoding="utf-8")
    print(f"[sync] docs/index.html <- {marker.strip()}")

# ── ground gained: a blend of the calculator's own two lead features ────────
def ground_weights(coef_path: Path) -> tuple[float, float]:
    """(w_lead, w_gain), summing to 1, so 'ground' stays in feet — a weighted AVERAGE of
    lead_at_firstmove_ft and gain_to_release_ft rather than an unweighted mean of the two.

    v14's calculator fits both as separate per-attempt inputs and learns that gain_to_release
    matters roughly 4x more per foot than lead_at_firstmove (their coefficients, in the currently
    committed fit: 0.303 vs 0.069). An unweighted 50/50 average was tested against the current
    gain-only definition and made the season metric WORSE, not better: year-over-year self-
    stability fell from 0.526 to 0.380, because lead_at_firstmove barely varies between runners
    (its variance is dominated by within-runner noise — see run_success_model's primary_lead
    stat) and diluting it in unweighted made Burst noisier for no offsetting gain. Weighting by
    the calculator's OWN fitted coefficients recovers almost all of that cost (YoY 0.489, Burst
    YoY 0.370 vs the gain-only 0.402) while making the season metric a direct reflection of what
    the per-pitch model actually weighs — which is the point of this change.

    Reads the coefficients the calculator last fit (DF_success_model.csv for the modern era, the
    classic payload's success_model.coef for the classic one) rather than re-fitting here, so this
    has no import-time dependency on sklearn and no ordering dependency on run_success_model()
    having already run in THIS invocation — the coefficients are deterministic given fixed data
    and a fixed seed, so reading yesterday's fit and today's are the same to the reported
    precision. Falls back to (0, 1) — pure gain_to_release, the pre-v15 definition — if no fit
    has been written yet (a fresh clone before the first full run)."""
    if not coef_path.exists():
        return 0.0, 1.0
    if coef_path.suffix == ".json":
        coef = json.loads(coef_path.read_text(encoding="utf-8"))["success_model"]["coef"]
    else:
        sm = pd.read_csv(coef_path)
        coef = {r.term: r.coefficient for r in sm.itertuples() if isinstance(r.term, str)}
    b_lead, b_gain = coef.get("lead_at_firstmove_ft"), coef.get("gain_to_release_ft")
    if not b_lead or not b_gain or b_lead <= 0 or b_gain <= 0:
        return 0.0, 1.0
    return b_lead / (b_lead + b_gain), b_gain / (b_lead + b_gain)


# ── raw era pool (per-attempt ground folded onto each runner-season) ────────
def load_era() -> pd.DataFrame:
    S = pd.read_csv(DATA / "Raw_Season.csv")
    A = pd.read_csv(DATA / "Raw_Attempts.csv")
    assert (S["sb_attempts"] == S["SB"] + S["CS"]).all(), "sb_attempts must equal SB+CS"
    era = S[S["season"] >= ERA_MIN].copy()
    era["raw_succ"] = era["SB"] / era["sb_attempts"].clip(lower=1)
    era["net_sb"]   = (era["SB"] - era["CS"]).astype(int)
    av = A[A["result"].isin(["SB", "CS"])].copy()
    w_lead, w_gain = ground_weights(RESULTS / "DF_success_model.csv")
    av["_ground"] = w_lead * pd.to_numeric(av["lead_at_firstmove_ft"], errors="coerce") \
                  + w_gain * pd.to_numeric(av["gain_to_release_ft"], errors="coerce")
    g = (av.groupby(["runner_id", "season"])
           .agg(ground=("_ground", "mean"),
                lead_rel=("lead_at_release_ft", "mean"),
                tracked=("gain_to_release_ft", "count")).reset_index())
    era = era.merge(g, on=["runner_id", "season"], how="left")
    return era.merge(catcher_faced(av), on=["runner_id", "season"], how="left")


# ── stage 1: fit league constants on the whole population (once) ────────────
def fit_league(era: pd.DataFrame) -> dict:
    f = era.dropna(subset=["ground", "sprint_speed"])
    b1, b0 = np.polyfit(f["sprint_speed"], f["ground"], 1)        # ground_hat = b0 + b1*speed
    # success rate expected from SPEED ONLY (linear; slope ~0.01/ft/s, nearly flat)
    s = era.dropna(subset=["sprint_speed", "raw_succ"])
    a1, a0 = np.polyfit(s["sprint_speed"], s["raw_succ"], 1)      # p_speed = a0 + a1*speed
    league = era["SB"].sum() / era["sb_attempts"].sum()
    p_speed  = a0 + a1 * era["sprint_speed"]
    netspeed = era["sb_attempts"] * (2 * p_speed - 1)
    sp_ref = (era["net_sb"] - netspeed).dropna()                 # volume-aware surplus distribution
    bu_ref = (era["ground"] - (b0 + b1 * era["sprint_speed"])).dropna()
    return dict(b0=float(b0), b1=float(b1), a0=float(a0), a1=float(a1),
                league=float(league), sp_ref=sp_ref.values, bu_ref=bu_ref.values)


# ── stage 2: score ANY set of runner-seasons (pure given `fit`) ─────────────
def score(rows: pd.DataFrame, fit: dict) -> pd.DataFrame:
    d = rows.copy()
    p_speed = fit["a0"] + fit["a1"] * d["sprint_speed"]       # success rate expected from SPEED ONLY
    d["netspeed"]   = d["sb_attempts"] * (2 * p_speed - 1)    # net bases his wheels alone buy
    d["steal_plus"] = d["net_sb"] - d["netspeed"]             # (SB-CS) above the same-speed average
    d["surplus"]    = d["steal_plus"]                         # Steal+ IS the surplus term (kept as alias)
    d["burst_ft"]   = d["ground"] - (fit["b0"] + fit["b1"] * d["sprint_speed"])
    d["steal_plus_pct"] = percentile_rank(d["steal_plus"], fit["sp_ref"])
    d["burst_pct"]      = percentile_rank(d["burst_ft"],   fit["bu_ref"])
    return d

def check_invariants(scored: pd.DataFrame) -> dict:
    """Row-level guarantees that hold for ANY n — the smoke test asserts these."""
    closes = (scored["netspeed"] + scored["surplus"] - scored["net_sb"]).abs().max()
    assert closes < 1e-9, f"decomposition must close exactly (off by {closes})"
    return {"n": len(scored), "decomp_max_resid": float(closes)}


# ── validation: the empirical "why" (kept honest) ───────────────────────────
def validate(era: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(question, metric, value, note=""):
        rows.append({"question": question, "metric": metric, "value": round(value, 3), "note": note})

    # (1) speed-neutrality — a good skill metric should NOT just re-measure raw speed
    for col, lab in [("net_sb", "Net Bases Gained"), ("sb_run_value", "Statcast SB Run Value"),
                     ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        add("corr with sprint speed (|low| = skill not wheels)", lab, pearson_r(era[col], era["sprint_speed"]))

    # (1b) agreement with Statcast's accepted SB run value (context, not a target)
    for col, lab in [("net_sb", "Net Bases Gained"), ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        add("corr with Statcast SB Run Value", lab, pearson_r(era[col], era["sb_run_value"]))

    # (2) how well each metric DESCRIBES the two targets in the SAME season
    for col, lab in [("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        add("describes net steals SB-CS (same season)", lab, pearson_r(era[col], era["net_sb"]))
    for col, lab in [("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        add("describes success rate (same season)", lab, pearson_r(era[col], era["raw_succ"]))

    # (3) how well each metric PREDICTS the same runner's NEXT season (T -> T+1)
    for col, lab in [("net_sb", "Net Bases Gained"), ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        r, n = year_over_year_corr(era, col, "net_sb");   add("predicts NEXT-YEAR net steals (SB-CS)", lab, r, f"n={n}")
    for col, lab in [("net_sb", "Net Bases Gained"), ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        r, n = year_over_year_corr(era, col, "raw_succ"); add("predicts NEXT-YEAR success rate", lab, r, f"n={n}")

    # (4) which metric is the most self-stable year to year (repeatable = more skill, less luck)
    for col, lab in [("net_sb", "Net Bases Gained"), ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        r, n = year_over_year_corr(era, col, col); add("year-over-year self-stability", lab, r, f"n={n}")

    # (5) Steal+ and Burst are independent lenses (near-zero corr = they measure different things)
    add("Steal+ vs Burst (independent lenses; ~0 = orthogonal)", "Steal+ x Burst",
        pearson_r(era["steal_plus"], era["burst_ft"]))
    return pd.DataFrame(rows)


# ── reliability: how much of one season's Steal+ is skill, and how much is coin-flips ──
def reliability_audit(full: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """HOW MUCH OF A SINGLE SEASON'S Steal+ IS ACTUALLY SIGNAL?

    Steal+ is a function of roughly 17 binary outcomes at the median volume, so a large part of its
    spread has to be luck. That is measurable rather than a matter of opinion.

    Steal+ = 2(SB - attempts x p_speed), and under the null that a runner is exactly as good as his
    speed predicts, SB ~ Binomial(attempts, p_speed). So the spread Steal+ would show from pure
    chance alone is

        sd_chance = 2 * sqrt(attempts * p * (1 - p))

    Subtracting the mean chance variance from the observed variance leaves the true between-runner
    skill variance, and their ratio is the reliability of a one-season Steal+.

    WHY THIS MATTERS. The reliability that falls out here (~0.20) independently reproduces the
    observed year-over-year self-correlation of Steal+ (0.192, n=224) — two calculations that share
    no code arriving at the same number. That is a strong argument that the low year-to-year figure
    is NOT a defect in Steal+: it is the irreducible consequence of grading a season on ~17 coin
    flips. Burst, which averages hundreds of tracked pitches instead, repeats at 0.479.

    The practical reading: Steal+ is a description of what a runner DID, and is only a projection of
    what he WILL DO once volume is large. sd_chance is the honest error bar to quote beside it."""
    p = fit["a0"] + fit["a1"] * full["sprint_speed"]
    n = full["sb_attempts"]
    sd_chance = 2 * np.sqrt(n * p * (1 - p))
    obs_var = float(full["steal_plus"].var())
    noise_var = float((sd_chance ** 2).mean())
    true_var = max(obs_var - noise_var, 0.0)
    z = full["steal_plus"] / sd_chance
    rows = [
        ("observed SD of Steal+ (bases)", round(float(np.sqrt(obs_var)), 2)),
        ("SD expected from chance alone (bases)", round(float(np.sqrt(noise_var)), 2)),
        ("implied true skill SD (bases)", round(float(np.sqrt(true_var)), 2)),
        ("reliability of a one-season Steal+", round(true_var / obs_var, 3)),
        ("observed year-over-year self-corr (independent check)",
         round(year_over_year_corr(full, "steal_plus", "steal_plus")[0], 3)),
        ("share of runner-seasons beyond 1 chance-SD (%)", round(100 * float((z.abs() > 1).mean()), 1)),
        ("share of runner-seasons beyond 2 chance-SD (%)", round(100 * float((z.abs() > 2).mean()), 1)),
        ("chance SD at the median volume (bases)",
         round(float(2 * np.sqrt(n.median() * 0.79 * 0.21)), 2)),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "value"])
    out.to_csv(RESULTS / "DF_v11_reliability.csv", index=False)

    ranked = full.assign(sd_chance=sd_chance, z=z).nlargest(8, "z")[
        ["player_name", "season", "sb_attempts", "steal_plus", "sd_chance", "z"]]
    ranked.round(2).to_csv(RESULTS / "DF_v11_reliability_top.csv", index=False)
    fig_reliability(full, sd_chance, z, true_var, noise_var, obs_var)
    return out


def fig_reliability(full, sd_chance, z, true_var, noise_var, obs_var):
    """Two panels answering 'is Steal+ real?' and 'where do net bases come from?'."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figs = ROOT / "Output" / "Figures"; figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=150)

    # A · how much of the spread is luck
    ax[0].bar(["observed\nspread", "expected from\nCHANCE alone", "implied true\nSKILL spread"],
              [np.sqrt(obs_var), np.sqrt(noise_var), np.sqrt(true_var)],
              color=["#0C2340", "#C0392B", "#1A7F47"], width=0.55)
    for i, v in enumerate([np.sqrt(obs_var), np.sqrt(noise_var), np.sqrt(true_var)]):
        ax[0].text(i, v + 0.08, f"{v:.2f}", ha="center", fontweight="bold", fontsize=11)
    ax[0].set_ylabel("standard deviation of Steal+ (bases)")
    ax[0].set_title("A · Most of one season's Steal+ spread is coin-flips\n"
                    f"reliability = {true_var/obs_var:.2f}  —  and Steal+ repeats year to year at 0.19",
                    fontsize=10.5, fontweight="bold", color="#0C2340")
    ax[0].set_ylim(0, np.sqrt(obs_var) * 1.25)

    # B · where net bases come from, by speed quintile
    d = full.dropna(subset=["sprint_speed", "net_sb"]).copy()
    d["q"] = pd.qcut(d["sprint_speed"], 5, labels=["slowest", "slow", "mid", "fast", "fastest"])
    g = d.groupby("q", observed=True).agg(ns=("netspeed", "mean"), sp=("steal_plus", "mean"),
                                          spd=("sprint_speed", "mean"))
    xs = np.arange(len(g))
    ax[1].bar(xs, g.ns, color="#9AA0A6", label="NetSpeed — what the wheels alone buy")
    ax[1].bar(xs, g.sp, bottom=g.ns, color="#2F6FB0", label="Steal+ — what the skill adds")
    ax[1].axhline(0, color="#333", lw=0.8)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"{i}\n{s:.1f} ft/s" for i, s in zip(g.index, g.spd)], fontsize=9)
    ax[1].set_ylabel("mean net bases (SB − CS)")
    ax[1].set_title("B · Net Bases = NetSpeed + Steal+, exactly\n"
                    "Speed sets the level; Steal+ averages ~0 in every speed group by construction",
                    fontsize=10.5, fontweight="bold", color="#0C2340")
    ax[1].legend(fontsize=8.5, frameon=False, loc="upper left")

    for a in ax:
        a.grid(alpha=0.13, lw=0.7, axis="y")
        for sp_ in ("top", "right"): a.spines[sp_].set_visible(False)
    fig.text(0.5, -0.04,
             "Left: under the null that a runner is exactly as good as his speed predicts, "
             "SB ~ Binomial(attempts, p), so chance alone produces an SD of 3.73 bases at these "
             "volumes.\nOnly 20% of the observed variance survives as skill — which is why "
             "Steal+ describes a season well (r = 0.64 with net steals) but forecasts the next one "
             "weakly.  Right: the two components sum to net bases with zero residual (< 1e-9).",
             ha="center", fontsize=8.6, color="#444")
    fig.tight_layout(); fig.savefig(figs / "Fig_v11_Reliability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ── webapp record export ─────────────────────────────────────────────────────
def to_records(full: pd.DataFrame) -> list:
    def num(v, nd=1):
        return None if (v is None or pd.isna(v)) else round(float(v), nd)
    recs = []
    for _, r in full.iterrows():
        recs.append({
            "id": int(r["runner_id"]), "name": r["player_name"],
            "team": (r["team"] if isinstance(r["team"], str) else ""), "season": int(r["season"]),
            "speed": num(r["sprint_speed"], 1), "speed_pct": num(r["speed_pct"], 0),
            "attempts": int(r["sb_attempts"]), "pop": num(r.get("pop_faced"), 2),
            "arm": num(r.get("arm_faced"), 1),
            "jump": num(r["jump_time"], 2), "jump_pct": num(r["jump_pct"], 0),
            "ground": num(r.get("ground"), 1), "ground_pct": num(r.get("ground_pct"), 0),
            "lead_rel": num(r.get("lead_rel"), 1),
            "steal_plus": num(r["steal_plus"], 1), "sp_pct": num(r["steal_plus_pct"], 0),
            "burst": num(r["burst_ft"], 1), "burst_pct": num(r["burst_pct"], 0),
            "sb": int(r["SB"]), "cs": int(r["CS"]), "net": int(r["net_sb"]),
            "netspeed": num(r["netspeed"], 1), "surplus": num(r["surplus"], 1),
            "success": (None if pd.isna(r["raw_succ"]) else int(round(r["raw_succ"] * 100))),
            "srv": num(r.get("sb_run_value"), 1),   # Statcast value — shown for reference only
        })
    return recs


# ── full run: score everyone, validate, write the artifacts ─────────────────
def build():
    era = load_era()
    fit = fit_league(era)
    full = score(era.copy(), fit)
    check_invariants(full)
    full["speed_pct"]  = percentile_rank(full["sprint_speed"], era["sprint_speed"].dropna().values)
    full["jump_pct"]   = 100 - percentile_rank(full["jump_time"], era["jump_time"].dropna().values)
    full["ground_pct"] = percentile_rank(full["ground"], era["ground"].dropna().values)
    return full, fit, validate(full)


# ── per-attempt model: the ~11k-attempt SB-success AUC (the project's grain) ──
# Confirms the thesis quantitatively: the per-pitch LEAD distances (which drive Burst)
# predict whether an individual attempt succeeds. Heavy deps (xgboost/sklearn) are
# imported lazily so the season model above never depends on them.
# Only the two INDEPENDENT lead quantities: where he already was when the pitcher committed, and
# how much he gained from there. lead_at_release_ft is deliberately EXCLUDED because it is their
# exact sum (lead_at_release = lead_at_firstmove + gain_to_release, R^2 = 0.999895; the residual
# caps at 0.1 ft, which is just the rounding granularity). Carrying all three gave VIFs of
# 1,588 / 5,836 / 9,532 — harmless for XGBoost's predictions, but it split feature importance
# arbitrarily across perfectly dependent columns, which made the importance chart unquotable.
# Dropping it costs 0.0014 AUROC (inside noise) and buys an interpretable model.
PA_LEAD_FEATS = ["lead_at_firstmove_ft", "gain_to_release_ft"]
# jump_time is omitted: it is a second measurement of the same thing as sprint_speed (r = -0.59,
# and bolts r = +0.71), which pushed sprint_speed to VIF 16.8. Dropping it takes max VIF to 6.6 AND
# nudges AUROC up (0.7820 -> 0.7829), so nothing is traded away. jump_time is still carried in the
# data and shown on the player card — it is only excluded as a MODEL feature.
PA_RUNNER_FEATS = ["sprint_speed", "accel_gap", "primary_lead", "lead_gain", "bolts"]
PA_FRIENDLY = {"lead_at_firstmove_ft": "Lead at first move (ft)",
               "gain_to_release_ft": "Ground gained to release (ft)",
               "lead_at_release_ft": "Lead at release (ft)", "base_is_3b": "Stealing 3rd",
               "sprint_speed": "Sprint speed", "jump_time": "Jump time", "accel_gap": "Accel gap",
               "primary_lead": "Primary lead (career)", "lead_gain": "Lead gain (career)", "bolts": "Bolts"}


def run_perattempt(seed: int = 42):
    """Fit the per-attempt SB-success model on Data/Raw_Attempts.csv (5-fold OOF AUC),
    write the AUC + importance tables and their two figures. Needs xgboost + sklearn;
    returns None (with a note) if they are unavailable rather than failing the run."""
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"per-attempt model skipped (missing {e.name}); season metrics unaffected")
        return None

    figs = ROOT / "Output" / "Figures"; figs.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "Raw_Attempts.csv")
    df = df[df["result"].isin(["SB", "CS"])].copy()
    df["y"] = (df["result"] == "SB").astype(int)
    df["base_is_3b"] = (df["base"].astype(str) == "3B").astype(int)

    sssi = pd.read_csv(RESULTS / "DF_v7_SSSI.csv")
    keep = ["runner_id", "season"] + [c for c in PA_RUNNER_FEATS if c in sssi.columns]
    df = df.merge(sssi[keep].drop_duplicates(["runner_id", "season"]), on=["runner_id", "season"], how="left")
    runner_cols = [c for c in PA_RUNNER_FEATS if c in df.columns]
    feats = PA_LEAD_FEATS + ["base_is_3b"] + runner_cols
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")

    y = df["y"].values
    prior = float(df["y"].mean())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def xgb():
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                             eval_metric="logloss", verbosity=0, random_state=seed, use_label_encoder=False)

    def encode_map(idx, key, smoothing=20.0):
        """Smoothed mean-target encoding learned from the rows in `idx`."""
        stats = df.iloc[idx].groupby(key)["y"].agg(["sum", "count"])
        return (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)

    def apply_map(enc, idx, key):
        return df.iloc[idx][key].map(enc).fillna(prior).values

    def cv_auc(battery_keys=()):
        """Pooled out-of-fold AUC. Battery tendency is target-encoded, and the TRAINING rows
        are encoded with an inner K-fold so no row ever sees its own outcome. Encoding the
        train rows from the same rows leaks the label into the feature: the model over-trusts
        it in training, the clean val encoding then behaves differently, and AUC drops. The
        leak is ~1/(n+smoothing) per row, so it is worst for sparsely-seen pitchers."""
        oof = np.zeros(len(df))
        for tr, va in cv.split(df, y):
            Xtr, Xva = df.iloc[tr][feats].copy(), df.iloc[va][feats].copy()
            for key in battery_keys:
                col = key + "_enc"
                inner_enc = np.full(len(tr), prior)
                inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
                for i_tr, i_va in inner.split(df.iloc[tr], y[tr]):
                    inner_enc[i_va] = apply_map(encode_map(tr[i_tr], key), tr[i_va], key)
                Xtr[col] = inner_enc                                  # nested OOF -> no self-leak
                Xva[col] = apply_map(encode_map(tr, key), va, key)    # val: encoded from full train
            oof[va] = xgb().fit(Xtr.values, y[tr]).predict_proba(Xva.values)[:, 1]
        return roc_auc_score(y, oof)

    auc_leads   = cv_auc()                                        # leads + base + runner skill
    auc_catcher = cv_auc(["catcher_id"])                          # + catcher tendency
    auc_pitcher = cv_auc(["pitcher_id"])                          # + pitcher tendency
    auc_full    = cv_auc(["catcher_id", "pitcher_id"])            # + both
    pd.DataFrame([{"model": "per-attempt: leads+base+runner",            "auc": round(auc_leads, 4)},
                  {"model": "per-attempt: + catcher (nested OOF)",       "auc": round(auc_catcher, 4)},
                  {"model": "per-attempt: + pitcher (nested OOF)",       "auc": round(auc_pitcher, 4)},
                  {"model": "per-attempt: + both battery (nested OOF)",  "auc": round(auc_full, 4)}]
                 ).to_csv(RESULTS / "DF_perattempt_AUC.csv", index=False)

    imp = (pd.DataFrame({"feature": feats, "importance": xgb().fit(df[feats].values, y).feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    imp.to_csv(RESULTS / "DF_perattempt_Importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    labels = ["Leads +\nrunner skill", "+ catcher\ntendency", "+ pitcher\ntendency", "+ both"]
    vals   = [auc_leads, auc_catcher, auc_pitcher, auc_full]
    ax.bar(labels, vals, color=["#10B981", "#2F6FB0", "#9CA3AF", "#1F2D3D"], width=0.55)
    ax.axhline(auc_leads, color="#10B981", lw=1, ls="--", zorder=0)
    ax.set_ylabel("CV AUC (nested out-of-fold)"); ax.set_ylim(0.5, 0.82)
    ax.set_title("Per-Attempt Model — leads carry most of it; the catcher adds the rest", fontsize=11.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
    ax.text(0.5, -0.20, "Catchers are seen ~46 times each, so their tendency is estimable; pitchers a median of 6 times,\n"
            "and the runner's lead already absorbs most of the pitcher's effect.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555")
    fig.tight_layout(); fig.savefig(figs / "Fig_AUC.png", dpi=160); plt.close(fig)

    g = imp.iloc[::-1]
    ax = plt.subplots(figsize=(7.6, 4.6))[1]
    ax.barh([PA_FRIENDLY.get(f, f) for f in g["feature"]], g["importance"],
            color=["#0EA5E9" if f in PA_LEAD_FEATS else "#1F2D3D" for f in g["feature"]])
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Per-Attempt Model — what decides a steal (blue = per-pitch lead distances)", fontsize=11)
    plt.tight_layout(); plt.savefig(figs / "Fig_Importance.png", dpi=160); plt.close()
    print(f"per-attempt SB-success AUC (nested OOF): {auc_leads:.4f} leads+runner | "
          f"{auc_catcher:.4f} +catcher | {auc_pitcher:.4f} +pitcher | {auc_full:.4f} +both, n={len(df)}")
    return auc_leads, auc_full


# ── the whiteboard model: 4 inputs, plain logistic regression ────────────────
# Chosen by measurement, not taste. Every candidate spec was fit on the sample it could actually
# ship on (see AUC_Roadmap):
#   speed + burst + gain                     n= 7,404   AUROC 0.7259   <- the old spec
#   speed + burst + firstmove + gain + pop   n= 7,264   AUROC 0.7470
#   speed + firstmove + gain + pop           n=10,844   AUROC 0.7559   <- this one
# Burst is DROPPED here and only here. It needs a qualified runner-season (>=10 attempts), so
# carrying it discarded ~3,600 attempts, and once the model already knows what the runner gained
# on THIS pitch it added only +0.002 AUROC while taking a confusing negative coefficient. Burst
# remains the season-level technique metric on the leaderboard, where it is speed-neutral and the
# most repeatable number in the project — it is simply not a per-pitch input.
# Catcher pop time replaces it: worth ~8x more (+0.017), and it is a genuine per-attempt fact.
SIMPLE_FEATS = ["sprint_speed", "lead_at_firstmove_ft", "gain_to_release_ft", "pop_faced"]

def run_success_model(seed: int = 42):
    """P(safe) for ONE attempt from four numbers a coach already has: sprint speed, the lead he
    had when the pitcher committed, the ground he gained from there, and the pop time of the
    catcher he is running on. Deliberately a plain logistic regression on RAW units — it also
    BEAT XGBoost on the same features (0.726 vs 0.722), so nothing is sacrificed for the
    simplicity — and the coefficients read directly as 'per ft/s', 'per foot' and 'per second'.
    Writes Output/Results/DF_success_model.csv and syncs the fit into docs/index.html."""
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        print(f"success model skipped (missing {e.name})")
        return None

    at = pd.read_csv(DATA / "Raw_Attempts.csv")
    at = at[at["result"].isin(["SB", "CS"])].copy()
    at["y"] = (at["result"] == "SB").astype(int)
    lb = pd.read_csv(RESULTS / "DF_v11_leaderboard.csv")[["runner_id", "season", "sprint_speed", "burst_ft"]]
    at = at.merge(lb, on=["runner_id", "season"], how="left")

    # The leaderboard only holds the 408 QUALIFIED runner-seasons (>=10 attempts, gated upstream
    # by the committed DF_v7_SSSI.csv), while the attempts table spans 1,079 — which used to drop
    # 3,619 attempts (35%) out of the calculator. Fill the rest in: sprint speed from the full
    # leaderboard, Burst recomputed from the leads already on disk against the same league line
    # fit_league() uses, so the definition is identical for everyone.
    sp_path = DATA / "sprint_speed.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        at = at.merge(sp, on=["runner_id", "season"], how="left")
        at["sprint_speed"] = at["sprint_speed"].fillna(at["sprint_speed_all"])

    meta = json.loads((DATA / "v11_players.json").read_text(encoding="utf-8"))["meta"]
    b0, b1 = meta["ground_fit"]["b0"], meta["ground_fit"]["b1"]
    w_lead, w_gain = ground_weights(RESULTS / "DF_success_model.csv")   # same blend load_era() used
    gain = (w_lead * pd.to_numeric(at["lead_at_firstmove_ft"], errors="coerce")
            + w_gain * pd.to_numeric(at["gain_to_release_ft"], errors="coerce"))
    grp = at.assign(_g=gain).groupby(["runner_id", "season"])["_g"]
    ground, n_tracked = grp.transform("mean"), grp.transform("count")
    league_ground = float(gain.mean())
    # shrink thin samples toward the league line; below 3 tracked attempts, don't guess at all
    w = (n_tracked / (n_tracked + 10)).clip(upper=1.0)
    ground_shrunk = w * ground + (1 - w) * league_ground
    burst_all = (ground_shrunk - (b0 + b1 * at["sprint_speed"])).where(n_tracked >= 3)
    at["burst_ft"] = at["burst_ft"].fillna(burst_all)

    pop_path = DATA / "poptime.csv"
    if pop_path.exists():
        pop = pd.read_csv(pop_path)[["catcher_id", "season", "pop_2b_sba"]] \
                .rename(columns={"pop_2b_sba": "pop_faced"})
        at = at.merge(pop, on=["catcher_id", "season"], how="left")
    else:
        at["pop_faced"] = np.nan
    at[SIMPLE_FEATS] = at[SIMPLE_FEATS].apply(pd.to_numeric, errors="coerce")

    # Primary lead (the ground he has BEFORE the pitcher commits) is context, not an input:
    # runners converge on it, so it barely separates anyone. Quantify that for the page.
    pl = pd.to_numeric(at["lead_at_firstmove_ft"], errors="coerce")
    seas = at.assign(_pl=pl).dropna(subset=["_pl"]).groupby(["runner_id", "season"])["_pl"]
    means, counts = seas.mean(), seas.count()
    qual = means[counts >= 15]
    grand = float(pl.mean())
    ss_between = float((counts[counts >= 15] * (qual - grand) ** 2).sum())
    dq = at.assign(_pl=pl).dropna(subset=["_pl"])
    dq = dq[dq.set_index(["runner_id", "season"]).index.isin(qual.index)]
    ss_total = float(((dq["_pl"] - grand) ** 2).sum())
    primary_lead = {"mean": round(grand, 1),
                    "runner_min": round(float(qual.min()), 1),
                    "runner_max": round(float(qual.max()), 1),
                    "within_pct": round(100 * (1 - ss_between / ss_total), 1),
                    "n_runner_seasons": int(len(qual))}

    at = at.dropna(subset=SIMPLE_FEATS).reset_index(drop=True)
    X, y = at[SIMPLE_FEATS].values, at["y"].values

    pipe = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    # AUPRC is meaningless without its no-skill floor (= the base rate), and F1 at a fixed 0.5
    # is vacuous on a skewed target — report the floor and the tuned threshold explicitly.
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
    auprc = float(average_precision_score(y, oof))
    brier = float(brier_score_loss(y, oof))
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in grid]
    best = int(np.argmax(f1s))
    pipe.fit(X, y)
    lr = pipe.named_steps["logisticregression"]
    coefs = dict(zip(SIMPLE_FEATS, lr.coef_[0]))
    payload = {"intercept": float(lr.intercept_[0]),
               "coef": {k: float(v) for k, v in coefs.items()},
               "auc": round(float(auc), 4), "n": int(len(at)),
               "auprc": round(auprc, 4), "auprc_floor": round(float(y.mean()), 4),
               "brier": round(brier, 4),
               "f1_best": round(float(f1s[best]), 4), "f1_threshold": round(float(grid[best]), 2),
               "base_rate": round(float(y.mean()), 4),
               "primary_lead": primary_lead,
               # full observed span so the sliders cover everyone (incl. Naylor at 24.4 ft/s)
               "range": {f: [round(float(at[f].min()), 1),
                             round(float(at[f].max()), 1),
                             round(float(at[f].median()), 1)] for f in SIMPLE_FEATS}}
    rows = [{"term": "intercept", "coefficient": round(payload["intercept"], 5), "odds_multiplier": ""}]
    rows += [{"term": f, "coefficient": round(v, 5), "odds_multiplier": round(float(np.exp(v)), 4)}
             for f, v in coefs.items()]
    rows.append({"term": "CV AUC (5-fold)", "coefficient": payload["auc"], "odds_multiplier": ""})
    pd.DataFrame(rows).to_csv(RESULTS / "DF_success_model.csv", index=False)

    # keep the browser calculator locked to this fit — never let the two drift apart
    site = ROOT / "docs" / "index.html"
    if site.exists():
        html = site.read_text(encoding="utf-8")
        a, b = "/*__SUCCESS_MODEL__*/", "/*__END_SUCCESS_MODEL__*/"
        if a in html and b in html:
            head, rest = html.split(a, 1)
            _, tail = rest.split(b, 1)
            site.write_text(head + a + json.dumps(payload, separators=(",", ":")) + b + tail,
                            encoding="utf-8")

    per_ft = np.exp(coefs["gain_to_release_ft"])
    print(f"4-input success model (logistic): AUROC {auc:.4f} | AUPRC {auprc:.4f} "
          f"(floor {y.mean():.4f}) | Brier {brier:.4f} | F1 {f1s[best]:.4f} @thr "
          f"{grid[best]:.2f} | n={len(at)}")
    print(f"  each +1 ft of ground gained multiplies the odds of being safe by {per_ft:.2f}x")
    fig_roc_calculator(y, oof, auc)
    return payload


def fig_roc_calculator(y, oof, auc):
    """The actual ROC curve behind the v14 calculator's AUROC, not just the number.

    An AUROC on its own doesn't show WHERE the model's discrimination comes from or what it costs
    to raise the true-positive rate. The curve does: at this base rate (~81% safe), a caught-
    stealing-averse threshold near the top-left knee trades roughly 1 point of true-positive rate
    for every ~2 points of false-positive rate it gives up, which is the shape a coach is actually
    choosing between when picking a threshold."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        return
    figs = ROOT / "Output" / "Figures"; figs.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, oof)
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=150)
    ax.plot(fpr, tpr, lw=2.6, color="#2F6FB0", label=f"v14 calculator (AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], lw=1.2, ls="--", color="#9AA0A6", label="no-skill (AUROC 0.500)")
    ax.fill_between(fpr, tpr, fpr, alpha=0.08, color="#2F6FB0")
    ax.set_xlabel("false-positive rate (called safe, actually caught)")
    ax.set_ylabel("true-positive rate (called safe, actually safe)")
    ax.set_title("ROC — the calculator vs a coin flip\n5-fold out-of-fold, n=10,844 attempts",
                fontsize=11, fontweight="bold", color="#0C2340")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    fig.tight_layout(); fig.savefig(figs / "Fig_ROC_Calculator.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    full, fit, val = build()
    lb_cols = ["runner_id", "season", "player_name", "team", "sprint_speed", "jump_time",
               "ground", "burst_ft", "SB", "CS", "net_sb", "raw_succ", "sb_attempts",
               "steal_plus", "steal_plus_pct", "burst_pct",
               "netspeed", "surplus", "sb_run_value"]
    full[lb_cols].sort_values("steal_plus", ascending=False).to_csv(
        RESULTS / "DF_v11_leaderboard.csv", index=False)
    val.to_csv(RESULTS / "DF_v11_validation.csv", index=False)

    league = fit["league"]
    payload = {
        "meta": {"version": "v11", "era": f"{ERA_MIN}-2026",
                 "n_player_seasons": len(full), "league_success_pct": round(league * 100, 1),
                 # league fits, so the report/site can draw the speed→expectation curves + Steal+ coefficients
                 "p_speed_fit": {"a0": fit["a0"], "a1": fit["a1"]},   # expected success = a0 + a1*speed
                 "ground_fit":  {"b0": fit["b0"], "b1": fit["b1"]}},  # expected ground = b0 + b1*speed
        "validation": val.to_dict(orient="records"),
        "players": to_records(full),
    }
    (DATA / "v11_players.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    sync_site_payload(payload, "const PAYLOAD = ")

    show = ["player_name", "season", "sprint_speed", "SB", "CS", "net_sb", "steal_plus", "burst_ft"]
    top12 = (full.dropna(subset=["steal_plus"])
             .sort_values("steal_plus", ascending=False).head(12)[show].round(2))
    print(f"{len(full)} runner-seasons ({ERA_MIN}-2026) | league SB% {league*100:.1f} | "
          f"p_speed = {fit['a0']:.3f} + {fit['a1']:.4f}*speed | ground = {fit['b0']:.2f} + {fit['b1']:.3f}*speed")
    print(top12.to_string(index=False))
    print(val.to_string(index=False))
    rel = reliability_audit(full, fit)
    print("\n=== HOW MUCH OF ONE SEASON'S Steal+ IS SIGNAL? ===")
    print(rel.to_string(index=False))
    run_perattempt()
    run_success_model()
    return full, val


if __name__ == "__main__":
    main()
