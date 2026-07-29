#!/usr/bin/env python3
"""
model_v12.py — the per-attempt steal model, extended with the three signals v11 was missing.

v11 asked: given the ground this runner covered on this pitch, did the attempt succeed?
It used lead distances + runner skill and reached CV AUROC 0.741 (0.752 with catcher tendency).
It knew nothing about WHO was pitching, WHAT was thrown, the COUNT, or the catcher's ARM.

v12 adds those, in the order they were expected to pay off:
  1. battery/pitch   is_lhp, pitch_class, bat_side_r   (MLB play-by-play, joined on play_id)
  2. count+situation balls, strikes, outs, inning, score_diff
  3. catcher arm     pop time to 2B, max-effort arm strength, exchange time (Savant poptime)

v11 is untouched and still the source of the published Steal+/Burst metrics. This file only
touches the per-attempt SKILL ENGINE, and reports an ablation so each block's contribution is
visible rather than bundled.

Metrics reported: AUROC (ranking), AUPRC (precision-recall; note the 81% base rate) and Brier
(calibration) — because a model can rank well and still be badly calibrated.

Inputs (all offline once scraped):
  Data/Raw_Attempts.csv            per-attempt leads          (build_features.py)
  Data/Raw_Attempt_Context.csv     pitch + count + situation  (scrape_statcast.py context)
  Data/poptime.csv                 catcher arm                (scrape_statcast.py poptime)
  Output/Results/DF_v7_SSSI.csv    runner season skill

Run:  python3 Scripts/model_v12.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_v11 import PA_LEAD_FEATS, PA_RUNNER_FEATS

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "Data"
RESULTS = ROOT / "Output" / "Results"
FIGS    = ROOT / "Output" / "Figures"

# Pitch codes collapse to four classes — per-code dummies would be mostly noise at this n.
PITCH_CLASS = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball", "FA": "fastball", "FT": "fastball",
    "SL": "breaking", "CU": "breaking", "KC": "breaking", "ST": "breaking", "SV": "breaking",
    "SC": "breaking", "CS": "breaking", "KN": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "EP": "offspeed",
}
# pc_fastball is the REFERENCE category and is deliberately omitted: the three pitch-class dummies
# sum to 1 on 99.94% of rows (only 7 "unknown" codes), so carrying all three is the dummy-variable
# trap — it put a VIF of 561 on pc_fastball. Two dummies + an implicit fastball baseline encode the
# same information without the dependency.
BATTERY_FEATS   = ["is_lhp", "bat_side_r", "pc_breaking", "pc_offspeed"]
SITUATION_FEATS = ["balls", "strikes", "outs", "inning", "score_diff", "ahead_in_count"]
# what actually ships: the post-pitch count columns are dropped (see load()), the
# unambiguous situation columns are kept.
SAFE_SITUATION_FEATS = ["outs", "inning", "score_diff"]
ARM_FEATS       = ["pop_2b_sba", "maxeff_arm_2b_3b_sba", "exchange_2b_3b_sba"]


def load() -> tuple[pd.DataFrame, np.ndarray]:
    """Assemble the per-attempt table: leads + runner skill + pitch context + catcher arm."""
    df = pd.read_csv(DATA / "Raw_Attempts.csv")
    df = df[df["result"].isin(["SB", "CS"])].copy()
    df["y"] = (df["result"] == "SB").astype(int)
    df["base_is_3b"] = (df["base"].astype(str) == "3B").astype(int)

    sssi = pd.read_csv(RESULTS / "DF_v7_SSSI.csv")
    keep = ["runner_id", "season"] + [c for c in PA_RUNNER_FEATS if c in sssi.columns]
    df = df.merge(sssi[keep].drop_duplicates(["runner_id", "season"]),
                  on=["runner_id", "season"], how="left")

    ctx_path = DATA / "Raw_Attempt_Context.csv"
    if ctx_path.exists():
        ctx = pd.read_csv(ctx_path)
        df = df.merge(ctx, on="play_id", how="left")
    else:
        raise SystemExit("Data/Raw_Attempt_Context.csv missing — "
                         "run: python3 Scripts/scrape_statcast.py context")

    cls = df["pitch_code"].map(PITCH_CLASS).fillna("other")
    for c in ("fastball", "breaking", "offspeed"):
        df[f"pc_{c}"] = (cls == c).astype(int)
    # CAVEAT: balls/strikes come from the play-by-play feed POST-pitch, so strikes==3 is a
    # strikeout and balls==4 a walk (which is why 3-3 and 0-3 appear). That is consistent with
    # the rest of this model — it is descriptive and already uses the lead the runner actually
    # achieved — but it is NOT the count a coach sees when deciding to send him. Deriving the
    # pre-pitch count needs the per-pitch call, which the cache does not keep; a re-scrape
    # would be required. The block is worth ~+0.003 AUROC either way (see the ablation).
    df["ahead_in_count"] = pd.to_numeric(df["balls"], errors="coerce") - \
                           pd.to_numeric(df["strikes"], errors="coerce")

    pop_path = DATA / "poptime.csv"
    if pop_path.exists():
        pop = pd.read_csv(pop_path)[["catcher_id", "season"] + ARM_FEATS]
        df = df.merge(pop, on=["catcher_id", "season"], how="left")
    else:
        for c in ARM_FEATS:
            df[c] = np.nan

    df = df.reset_index(drop=True)
    return df, df["y"].values


def _splits(df, y, split: str, seed: int):
    """Yield (train_idx, val_idx) for a named validation regime.

    random  — StratifiedKFold. Optimistic: a runner's other attempts sit in the training set,
              and his season-level features are constant, so identity partly carries over.
    group   — GroupKFold on runner_id. No runner appears on both sides.
    forward — train on seasons < T, test on season T. The only regime that mirrors real use
              (predicting a season you have not seen) and the only one immune to the season
              aggregates in the feature set leaking backwards.
    """
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    if split == "random":
        yield from StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y)
    elif split == "group":
        yield from GroupKFold(5).split(df, y, df["runner_id"].values)
    elif split == "forward":
        for T in sorted(df["season"].unique())[1:]:
            tr = np.where(df["season"].values < T)[0]
            va = np.where(df["season"].values == T)[0]
            if len(va) >= 200 and len(tr) >= 500:
                yield tr, va
    else:
        raise ValueError(split)


def evaluate(df: pd.DataFrame, y: np.ndarray, feats: list[str], catcher_enc: bool,
             seed: int = 42, split: str = "random", return_pred: bool = False):
    """Pooled out-of-fold AUROC / AUPRC / Brier under a chosen validation regime. Catcher
    tendency, when used, is encoded with an inner fold so no row ever sees its own outcome
    (the v11 leak, fixed)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from xgboost import XGBClassifier

    feats = [f for f in feats if f in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    prior = float(y.mean())

    def xgb():
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                             eval_metric="logloss", verbosity=0, random_state=seed,
                             use_label_encoder=False)

    def enc_map(idx, sm=20.0):
        s = df.iloc[idx].groupby("catcher_id")["y"].agg(["sum", "count"])
        return (s["sum"] + prior * sm) / (s["count"] + sm)

    oof = np.full(len(df), np.nan)
    for tr, va in _splits(df, y, split, seed):
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        if catcher_enc:
            inner_vals = np.full(len(tr), prior)
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
            for i_tr, i_va in inner.split(df.iloc[tr], y[tr]):
                m = enc_map(tr[i_tr])
                inner_vals[i_va] = df.iloc[tr[i_va]]["catcher_id"].map(m).fillna(prior).values
            Xtr["catcher_enc"] = inner_vals
            Xva["catcher_enc"] = df.iloc[va]["catcher_id"].map(enc_map(tr)).fillna(prior).values
        oof[va] = xgb().fit(Xtr.values, y[tr]).predict_proba(Xva.values)[:, 1]

    scored = ~np.isnan(oof)            # forward leaves the first season unscored
    yy, pp = y[scored], oof[scored]
    out = {"auroc": roc_auc_score(yy, pp), "auprc": average_precision_score(yy, pp),
           "brier": brier_score_loss(yy, pp), "n_scored": int(scored.sum())}
    return (out, oof) if return_pred else out


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency. The calculator states a probability, not a rank, so
    this is the check that matters for it — AUROC would not notice systematic over-confidence."""
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b + 1, "n": int(m.sum()),
                     "predicted": round(float(p[m].mean()), 4),
                     "observed": round(float(y[m].mean()), 4),
                     "gap": round(float(y[m].mean() - p[m].mean()), 4)})
    return pd.DataFrame(rows)


def main():
    df, y = load()
    matched = df["is_lhp"].notna().mean()
    print(f"v12 per-attempt table: {len(df)} attempts | base rate {y.mean():.3f} | "
          f"pitch context matched on {matched*100:.1f}% | "
          f"catcher arm on {df['pop_2b_sba'].notna().mean()*100:.1f}%")

    base = PA_LEAD_FEATS + ["base_is_3b"] + [c for c in PA_RUNNER_FEATS if c in df.columns]
    # SHIP_FEATS drops the post-pitch count (see the caveat in load()): balls/strikes arrive
    # after the pitch, so they are not what a coach sees, and they are worth +0.0002. outs /
    # inning / score_diff are unambiguous and stay.
    ship = base + BATTERY_FEATS + SAFE_SITUATION_FEATS + ARM_FEATS
    ladder = [
        ("v11 baseline — leads + runner skill",        base,                                    False),
        ("v11 + catcher tendency (nested OOF)",        base,                                    True),
        ("+ 1. battery & pitch type",                  base + BATTERY_FEATS,                    True),
        ("+ 2. count & situation",                     base + BATTERY_FEATS + SITUATION_FEATS,  True),
        ("+ 3. catcher arm (pop time)",                base + BATTERY_FEATS + SITUATION_FEATS + ARM_FEATS, True),
        ("v12 SHIPPED (post-pitch count dropped)",     ship,                                    True),
    ]

    # each block ALONE on top of the baseline — cumulative order can hide redundancy
    standalone = [
        ("alone: battery & pitch type", base + BATTERY_FEATS,   False),
        ("alone: count & situation",    base + SITUATION_FEATS, False),
        ("alone: catcher arm",          base + ARM_FEATS,       False),
    ]

    rows = []
    for name, feats, enc in ladder + standalone:
        m = evaluate(df, y, feats, enc)
        rows.append({"model": name, "n_features": len([f for f in feats if f in df.columns]) + int(enc),
                     "auroc": round(m["auroc"], 4), "auprc": round(m["auprc"], 4),
                     "brier": round(m["brier"], 4)})
        print(f"  {name:44s} AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | Brier {m['brier']:.4f}")

    # ── validation regimes: the random split is optimistic; report the honest ones too ──
    print("\nvalidation regimes (shipped feature set):")
    vrows = []
    for split in ("random", "group", "forward"):
        m = evaluate(df, y, ship, True, split=split)
        vrows.append({"split": split, "auroc": round(m["auroc"], 4), "auprc": round(m["auprc"], 4),
                      "brier": round(m["brier"], 4), "n_scored": m["n_scored"]})
        print(f"  {split:8s} AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | "
              f"Brier {m['brier']:.4f} | n={m['n_scored']}")
    pd.DataFrame(vrows).to_csv(RESULTS / "DF_v12_Validation.csv", index=False)

    # ── calibration: does a stated probability mean what it says? ──
    # Assessed under BOTH regimes on purpose. In-distribution the model is well calibrated;
    # the forward holdout drifts because the league itself is drifting (SB success is falling
    # season over season), so a model trained on the past predicts an easier environment than
    # the one it is scored in. That is a base-rate shift, not model over-confidence — isotonic
    # on pooled data would hide the cause rather than fix it, so no correction is applied.
    rels, roc_oof = [], {}
    for split in ("random", "forward"):
        _, oof = evaluate(df, y, ship, True, split=split, return_pred=True)
        roc_oof[split] = oof
        r = reliability(y, oof); r.insert(0, "split", split); rels.append(r)
        print(f"\ncalibration ({split}): max |predicted − observed| = {r['gap'].abs().max():.3f}, "
              f"mean gap {r['gap'].mean():+.4f}")
    pd.concat(rels).to_csv(RESULTS / "DF_v12_Calibration.csv", index=False)

    rate = df.groupby("season")["y"].mean()
    print("league SB rate by season: " + " · ".join(f"{s} {v:.3f}" for s, v in rate.items()) +
          "  <- the environment is getting harder, which is what the forward gap reflects")

    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "DF_v12_AUC.csv", index=False)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGS.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.2, 4.6))
        labels = ["v11\nbaseline", "+ catcher\ntendency", "+ battery\n& pitch",
                  "+ count &\nsituation", "+ catcher\narm"]
        vals = out["auroc"].tolist()[:5]
        colors = ["#9CA3AF", "#6B7280", "#2F6FB0", "#1D4ED8", "#10B981"]
        ax.bar(labels, vals, color=colors, width=0.58)
        ax.axhline(vals[0], color="#9CA3AF", ls="--", lw=1, zorder=0)
        ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
        ax.set_ylabel("CV AUROC (out-of-fold)")
        ax.set_title("v12 — what each added signal is worth, per attempt", fontsize=12)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10.5)
        plt.tight_layout(); plt.savefig(FIGS / "Fig_v12_AUC.png", dpi=160); plt.close()

        # the actual ROC curve for the shipped model, random AND forward — the gap between the two
        # curves IS the cost of not being allowed to see the future, drawn rather than just quoted
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=150)
        for split, color in [("random", "#2F6FB0"), ("forward", "#C0392B")]:
            oof = roc_oof[split]; scored = ~np.isnan(oof)
            fpr, tpr, _ = roc_curve(y[scored], oof[scored])
            auroc = vrows[0]["auroc"] if split == "random" else vrows[2]["auroc"]
            ax.plot(fpr, tpr, lw=2.4, color=color, label=f"{split} (AUROC {auroc:.3f})")
        ax.plot([0, 1], [0, 1], lw=1.1, ls="--", color="#9AA0A6", label="no-skill (0.500)")
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
        ax.set_title("ROC — v12 shipped model, random vs forward split\n"
                     "the gap between the curves is the cost of testing on the future",
                     fontsize=10.5, fontweight="bold", color="#0C2340")
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
        fig.tight_layout(); fig.savefig(FIGS / "Fig_v12_ROC.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass

    gain = out["auroc"].iloc[5] - out["auroc"].iloc[0]
    print(f"\nv12 vs v11 baseline: AUROC {out['auroc'].iloc[0]:.4f} -> {out['auroc'].iloc[5]:.4f} "
          f"({gain:+.4f})")
    return out


if __name__ == "__main__":
    main()
