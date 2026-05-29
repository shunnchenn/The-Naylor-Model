#!/usr/bin/env python3
"""
Pooled multi-runner per-attempt steal model — does the CV delivery-window velocity
metric add discrimination over the lead/timing features alone?

Pools every Statcast-tracked attempt we have CV deliveries for, across runners:
    Naylor   2024 + 2025 + 2026     (slow, 24.5 ft/s, high-success)
    Soto     2025 + 2026            (fast, 25.8 ft/s, 30/0)
    Vladdy / Yandy / Torres / Bichette 2025   (26-27 ft/s, LOW-success stealers)

The low-success runners are the key addition: they contribute most of the CS
negatives the pool was starved of (Soto/Naylor are near-perfect stealers), which is
exactly what the SB-success classification needs to be anything but noise. Headlines:
  - +VELOCITY−BASE classification delta (sensitive to the SB/CS balance)
  - univariate SB-vs-CS separation per feature (stable)
  - continuous run_value regression (stable)
Still treat the classification AUC as proof-of-harness — even pooled, CS are a minority.

Two targets:
  (1) y_success  = 1 if SB else 0           -> logistic, leave-one-out CV AUC
  (2) run_value  (continuous)                -> OLS, report R² + corr

Feature blocks (all PER-ATTEMPT varying — within one runner, season-constant metrics
like sprint speed cannot discriminate between his own attempts, so they are excluded
here; the cross-runner sprint-vs-velocity comparison is a separate, future test):
    BASE      = lead_at_firstmove_ft, gain_to_release_ft, lead_at_release_ft
    +DELIVERY = BASE + delivery_s
    +VELOCITY = BASE + delivery_s + avg_velocity_ftps      (the metric under test)

Headline = CV-AUC(+VELOCITY) − CV-AUC(BASE). With only ~3 CS in ~36 attempts the
classification AUC is *very* noisy (wide CIs) — this is a proof-of-harness, not a
proof-of-effect. The univariate SB-vs-CS separation table and the run_value regression
are the more stable reads at this sample size.

Output: attempt_auc.csv + printed verdict.
"""
import csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # cv_pilot/
SOURCES = [
    os.path.join(ROOT, "Naylor_2024", "delivery_velocity_2024.csv"),
    os.path.join(ROOT, "Naylor_2025", "delivery_velocity_2025.csv"),
    os.path.join(ROOT, "Naylor_2026", "delivery_velocity_2026.csv"),
    os.path.join(ROOT, "Soto_2025", "delivery_velocity_2025.csv"),
    os.path.join(ROOT, "Soto_2026", "delivery_velocity_2026.csv"),
    os.path.join(ROOT, "Vladdy_2025", "delivery_velocity_2025.csv"),
    os.path.join(ROOT, "Yandy_2025", "delivery_velocity_2025.csv"),
    os.path.join(ROOT, "Torres_2025", "delivery_velocity_2025.csv"),
    os.path.join(ROOT, "Bichette_2025", "delivery_velocity_2025.csv"),
]

try:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import roc_auc_score
except ImportError:
    sys.exit("needs scikit-learn")

FEATS = {
    "BASE": ["lead_at_firstmove_ft", "gain_to_release_ft", "lead_at_release_ft"],
    "+DELIVERY": ["lead_at_firstmove_ft", "gain_to_release_ft", "lead_at_release_ft",
                  "delivery_s"],
    "+VELOCITY": ["lead_at_firstmove_ft", "gain_to_release_ft", "lead_at_release_ft",
                  "delivery_s", "avg_velocity_ftps"],
}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_pool():
    rows = []
    for path in SOURCES:
        if not os.path.exists(path):
            print(f"  ! missing {path} (skipping)")
            continue
        yr = os.path.basename(path).split("_")[-1][:4]
        runner = os.path.basename(os.path.dirname(path)).split("_")[0]
        for r in csv.DictReader(open(path)):
            # require a usable CV velocity (drops the few unmeasurable attempts)
            if r.get("avg_velocity_ftps", "") == "":
                continue
            rec = {
                "year": yr,
                "runner": runner,
                "date": r["date"],
                "pitcher_name": r["pitcher_name"],
                "result": r["result"],
                "y_success": 1 if r["result"] == "SB" else 0,
                "run_value": fnum(r["run_value"]),
                "lead_at_firstmove_ft": fnum(r["lead_at_firstmove_ft"]),
                "gain_to_release_ft": fnum(r["gain_to_release_ft"]),
                "lead_at_release_ft": fnum(r["lead_at_release_ft"]),
                "delivery_s": fnum(r["delivery_s"]),
                "avg_velocity_ftps": fnum(r["avg_velocity_ftps"]),
            }
            if None in (rec["delivery_s"], rec["avg_velocity_ftps"]):
                continue
            rows.append(rec)
    return rows


def loo_auc(X, y):
    """Leave-one-out CV: collect held-out predicted probs, score one global AUC."""
    y = np.asarray(y)
    if len(set(y)) < 2:
        return None
    preds = np.zeros(len(y))
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        if len(set(y[tr])) < 2:
            preds[te] = y[tr].mean()
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, C=1.0))
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, preds)


def univariate_auc(rows, feat):
    """AUC of a single feature ranking SB(1) vs CS(0); flip if negatively oriented."""
    y = np.array([r["y_success"] for r in rows])
    x = np.array([r[feat] for r in rows])
    if len(set(y)) < 2:
        return None
    a = roc_auc_score(y, x)
    return max(a, 1 - a), ("+" if a >= 0.5 else "-")


def main():
    rows = load_pool()
    n = len(rows)
    n_sb = sum(r["y_success"] for r in rows)
    n_cs = n - n_sb
    print("=" * 78)
    print("POOLED MULTI-RUNNER PER-ATTEMPT STEAL MODEL — does CV velocity add over lead/timing?")
    print("=" * 78)
    by_run = {}
    for r in rows:
        by_run.setdefault(r["runner"], []).append(r)
    print(f"pooled attempts: {n}  ({n_sb} SB / {n_cs} CS)")
    print("  by runner:  " + "  ".join(
        f"{rn}:{sum(x['y_success'] for x in v)}SB/{sum(1-x['y_success'] for x in v)}CS"
        for rn, v in sorted(by_run.items())))
    print(f"** small-sample caveat: only {n_cs} CS -> classification AUC is HIGH-VARIANCE")
    print(f"   (wide CIs). Treat as proof-of-harness; univariate + run_value reads below")
    print(f"   are more stable. Real power comes when pooled with Soto et al.")
    print()

    # ---------- (1) logistic SB-success, LOO AUC for each feature block ----------
    print("-" * 78)
    print("(1) SB-success classification — leave-one-out CV AUC:")
    aucs = {}
    for name, feats in FEATS.items():
        X = np.array([[r[f] for f in feats] for r in rows])
        a = loo_auc(X, [r["y_success"] for r in rows])
        aucs[name] = a
        print(f"    {name:11} ({len(feats)} feats)  AUC = {a:.3f}" if a is not None
              else f"    {name:11}  AUC = n/a")
    if aucs.get("BASE") is not None and aucs.get("+VELOCITY") is not None:
        d = aucs["+VELOCITY"] - aucs["BASE"]
        print(f"    --> HEADLINE delta (+VELOCITY − BASE) = {d:+.3f}")

    # ---------- (2) univariate SB-vs-CS separation ----------
    print()
    print("-" * 78)
    print("(2) univariate SB-vs-CS separation (single-feature AUC, orientation):")
    uni = {}
    for f in ["avg_velocity_ftps", "delivery_s", "gain_to_release_ft",
              "lead_at_firstmove_ft", "lead_at_release_ft"]:
        res = univariate_auc(rows, f)
        if res:
            uni[f] = res[0]
            print(f"    {f:22} AUC = {res[0]:.3f}  ({res[1]} oriented)")

    # ---------- (3) run_value regression (continuous, more stable) ----------
    print()
    print("-" * 78)
    print("(3) run_value (continuous) regression — does velocity track value?:")
    yv = np.array([r["run_value"] for r in rows])
    reg_out = {}
    for name, feats in FEATS.items():
        X = np.array([[r[f] for f in feats] for r in rows])
        reg = make_pipeline(StandardScaler(), LinearRegression()).fit(X, yv)
        r2 = reg.score(X, yv)
        reg_out[name] = r2
        print(f"    {name:11} in-sample R² = {r2:.3f}")
    # univariate corr of velocity with run_value
    vv = np.array([r["avg_velocity_ftps"] for r in rows])
    c = np.corrcoef(vv, yv)[0, 1]
    print(f"    corr(avg_velocity_ftps, run_value) = {c:+.3f}")

    # ---------- means by outcome ----------
    print()
    print("-" * 78)
    print("means by outcome:")
    for lab, want in (("SB", 1), ("CS", 0)):
        sub = [r for r in rows if r["y_success"] == want]
        if sub:
            mv = np.mean([r["avg_velocity_ftps"] for r in sub])
            md = np.mean([r["delivery_s"] for r in sub])
            print(f"    {lab} (n={len(sub):2})  mean velocity = {mv:5.2f} ft/s   "
                  f"mean delivery = {md:.3f} s")

    # ---------- write ----------
    os.makedirs(HERE, exist_ok=True)
    with open(os.path.join(HERE, "attempt_auc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_attempts", n]); w.writerow(["n_sb", n_sb]); w.writerow(["n_cs", n_cs])
        for k, v in aucs.items():
            w.writerow([f"loo_auc_{k}", "" if v is None else round(v, 4)])
        if aucs.get("BASE") is not None and aucs.get("+VELOCITY") is not None:
            w.writerow(["loo_auc_delta_velocity_minus_base",
                        round(aucs["+VELOCITY"] - aucs["BASE"], 4)])
        for k, v in uni.items():
            w.writerow([f"univariate_auc_{k}", round(v, 4)])
        for k, v in reg_out.items():
            w.writerow([f"runvalue_R2_{k}", round(v, 4)])
        w.writerow(["corr_velocity_runvalue", round(float(c), 4)])
    print()
    print(f"wrote {os.path.join(HERE, 'attempt_auc.csv')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
