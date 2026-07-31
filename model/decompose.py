#!/usr/bin/env python3
"""
decompose.py — row-level audit of the Naylor Model steal calculator.

Rebuilds the committed 4-input logistic fit (metrics.run_success_model) from raw inputs and
exposes every quantity sklearn computes internally:

    logit_i = b0 + SUM_f  beta_f * x_{i,f}          p_i = 1 / (1 + exp(-logit_i))

Reference fit — Output/Results/DF_success_model.csv, seed 42, n = 10,844, 2023-2026:
    intercept -23.01083 | sprint_speed 0.31684 | lead_at_firstmove_ft 0.06874
    gain_to_release_ft 0.30281 | pop_faced 5.85786 | 5-fold OOF AUROC 0.7559
verify_reference() asserts this module still lands there, so the decomposition below is a
decomposition OF THE SHIPPED MODEL and not of a look-alike refit.

GROUND GAINED (`gain_to_release_ft`) is the feet a runner wins between the pitcher's FIRST
MOVE and the pitcher's RELEASE: Savant `r_sec_minus_prim_lead` = `r_secondary_lead` (lead at
release) - `r_primary_lead` (lead at first move). The endpoint is release. It is not the ball
arriving at the catcher: the per-attempt feed carries no ball-arrival timestamp, so that
endpoint is not observable in this data and was never what the column measured.

Every function returns named objects rather than only a final number; no calculation is left
inside a pipeline that could be reconstructed outside one. Row identity is the modeling-frame
index `attempt_ix`, carried onto every diagnostic frame.

    python3 model/decompose.py            full audit (~4 min: 1000 bootstraps + 200 permutations)
    python3 model/decompose.py --quick    same audit, fewer resamples (~20 s)

    import decompose as D; a = D.run()    a["decomposition"], a["coefficients"], ...

Progress goes to STDERR so it stays visible when stdout is redirected to a file. Without it the
full run prints nothing for four minutes and is indistinguishable from a hang.
"""
from __future__ import annotations
import argparse
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import make_pipeline

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "Data"
RESULTS = ROOT / "Output" / "Results"

FEATURES = ["sprint_speed", "lead_at_firstmove_ft", "gain_to_release_ft", "pop_faced"]
UNITS    = {"sprint_speed": "ft/s", "lead_at_firstmove_ft": "ft",
            "gain_to_release_ft": "ft", "pop_faced": "s"}
IDENTITY = ["play_id", "runner_id", "season", "date", "base",
            "pitcher_name", "catcher_name", "result"]

# lead_at_release_ft = lead_at_firstmove_ft + gain_to_release_ft by construction. Held out of
# the fit for that reason; carried here only so collinearity() can show the VIF it would cause.
REDUNDANT = "lead_at_release_ft"

SEED       = 42   # the calculator's committed seed — changing it changes the shipped fit
BOOT_SEED  = 7
PERM_SEED  = 11
CURVE_SEED = 13

REFERENCE = {"intercept": -23.01083, "sprint_speed": 0.31684, "lead_at_firstmove_ft": 0.06874,
             "gain_to_release_ft": 0.30281, "pop_faced": 5.85786, "auc": 0.7559, "n": 10844}

_T0 = None


def _stage(msg: str) -> None:
    """Progress line on STDERR, flushed. The resampling stages below run for minutes with no
    output of their own; when stdout is redirected to a file Python block-buffers it, so without
    this the whole audit is silent until it finishes and looks like it hung."""
    import sys as _sys, time as _time
    global _T0
    if _T0 is None:
        _T0 = _time.time()
    print(f"[decompose {_time.time() - _T0:6.1f}s] {msg}", file=_sys.stderr, flush=True)


# ── data ─────────────────────────────────────────────────────────────────────
def load_data() -> dict:
    """Rebuild the calculator's modeling frame. Merge order and the terminal dropna are copied
    from run_success_model rather than re-derived: StratifiedKFold(shuffle=True) assigns folds
    by position, so any reordering silently changes the OOF AUROC."""
    attempts = pd.read_csv(DATA / "Raw_Attempts.csv")
    decided  = attempts[attempts["result"].isin(("SB", "CS"))].copy()
    decided["y"] = (decided["result"] == "SB").astype(int)

    speed_qualified = pd.read_csv(RESULTS / "DF_v15_leaderboard.csv")[
        ["runner_id", "season", "sprint_speed"]]
    speed_all = pd.read_csv(DATA / "sprint_speed.csv")            # unqualified runner-seasons
    pop = (pd.read_csv(DATA / "poptime.csv")[["catcher_id", "season", "pop_2b_sba"]]
             .rename(columns={"pop_2b_sba": "pop_faced"}))

    merged = decided.merge(speed_qualified, on=["runner_id", "season"], how="left")
    merged = merged.merge(speed_all, on=["runner_id", "season"], how="left")
    merged["sprint_speed"] = merged["sprint_speed"].fillna(merged["sprint_speed_all"])
    merged = merged.merge(pop, on=["catcher_id", "season"], how="left")
    merged[FEATURES] = merged[FEATURES].apply(pd.to_numeric, errors="coerce")
    merged[REDUNDANT] = pd.to_numeric(merged[REDUNDANT], errors="coerce")

    model = merged.dropna(subset=FEATURES).reset_index(drop=True)
    model.index.name = "attempt_ix"

    ingestion = pd.DataFrame([
        {"stage": "Raw_Attempts.csv rows",        "n": len(attempts)},
        {"stage": "decided (SB/CS only)",         "n": len(decided)},
        {"stage": "+ sprint speed, pop time",     "n": len(merged)},
        {"stage": "complete on all 4 features",   "n": len(model)},
        {"stage": "dropped: any feature missing", "n": len(merged) - len(model)},
    ])
    return {"attempts": attempts, "merged": merged, "model": model, "ingestion": ingestion}


def validate_data(merged: pd.DataFrame, model: pd.DataFrame) -> dict:
    """Assumption checks that run before the fit, not after it fails."""
    balance = (model.groupby("season")
                    .agg(n=("y", "size"), safe=("y", "sum"), sb_rate=("y", "mean"))
                    .reset_index())
    balance.loc[len(balance)] = ["ALL", len(model), int(model["y"].sum()), float(model["y"].mean())]

    missing = pd.DataFrame([
        {"feature": f,
         "missing_pre_drop": int(merged[f].isna().sum()),
         "missing_pct": 100 * float(merged[f].isna().mean()),
         "missing_post_drop": int(model[f].isna().sum())}
        for f in FEATURES + [REDUNDANT]])

    d = model[FEATURES].describe().T
    d["unit"]  = [UNITS[f] for f in d.index]
    d["skew"]  = model[FEATURES].skew()
    d["n_unique"] = model[FEATURES].nunique()
    distributions = d[["unit", "count", "mean", "std", "min", "25%", "50%", "75%", "max",
                       "skew", "n_unique"]]

    # gain_to_release_ft <= 0 means the runner lost ground before release — physically possible
    # (a dive back) but worth counting before it is treated as a linear predictor.
    edge = pd.DataFrame([
        {"check": "duplicate play_id",              "n": int(model["play_id"].duplicated().sum())},
        {"check": "gain_to_release_ft <= 0",        "n": int((model["gain_to_release_ft"] <= 0).sum())},
        {"check": "lead_at_firstmove_ft <= 0",      "n": int((model["lead_at_firstmove_ft"] <= 0).sum())},
        {"check": "violations of firstmove + gain == lead_at_release (> 0.1 ft)",
         "n": int(((model["lead_at_firstmove_ft"] + model["gain_to_release_ft"]
                    - model[REDUNDANT]).abs() > 0.1001).sum())},
    ])
    return {"class_balance": balance, "missingness": missing,
            "distributions": distributions, "edge_cases": edge}


def collinearity(model: pd.DataFrame) -> dict:
    """VIF by explicit OLS-on-the-others, so the R^2 behind each number is inspectable.
    Reported twice: for the shipped 4, and with lead_at_release_ft added back."""
    def vif_table(cols):
        X = model[cols].to_numpy(float)
        rows = []
        for j, col in enumerate(cols):
            others = np.delete(X, j, axis=1)
            design = np.column_stack([np.ones(len(others)), others])
            beta, *_ = np.linalg.lstsq(design, X[:, j], rcond=None)
            resid = X[:, j] - design @ beta
            ss_res = float(resid @ resid)
            ss_tot = float(((X[:, j] - X[:, j].mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot
            rows.append({"feature": col, "r2_on_others": r2,
                         "vif": float("inf") if r2 >= 1 else 1.0 / (1.0 - r2)})
        return pd.DataFrame(rows)

    return {"correlation": model[FEATURES + [REDUNDANT]].corr(),
            "vif_shipped": vif_table(FEATURES),
            "vif_with_redundant": vif_table(FEATURES + [REDUNDANT])}


# ── fit ──────────────────────────────────────────────────────────────────────
def fit_model(model: pd.DataFrame, seed: int = SEED) -> dict:
    """The shipped pipeline, unchanged. SimpleImputer is a no-op here (dropna already ran) and
    is kept only so this is the same estimator object the calculator ships; imputed_cells
    asserts that, because a live imputer would break the manual reconstruction below."""
    X = model[FEATURES].to_numpy(float)
    y = model["y"].to_numpy(int)

    pipe = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=5000))
    pipe.fit(X, y)
    lr = pipe.named_steps["logisticregression"]

    beta = dict(zip(FEATURES, (float(c) for c in lr.coef_[0])))
    intercept = float(lr.intercept_[0])

    p = lr.predict_proba(X)[:, 1]
    se = wald_se(X, p)

    sd = model[FEATURES].std()
    coefficients = pd.DataFrame([
        {"term": f, "unit": UNITS[f], "coefficient": beta[f], "se_wald": se[i + 1],
         "z": beta[f] / se[i + 1] if se[i + 1] > 0 else np.nan,
         "odds_ratio_per_unit": float(np.exp(beta[f])),
         "odds_ratio_per_sd": float(np.exp(beta[f] * sd[f])),
         "sd": float(sd[f])}
        for i, f in enumerate(FEATURES)])
    coefficients.loc[len(coefficients)] = {
        "term": "intercept", "unit": "logit", "coefficient": intercept, "se_wald": se[0],
        "z": intercept / se[0] if se[0] > 0 else np.nan,
        "odds_ratio_per_unit": np.nan, "odds_ratio_per_sd": np.nan, "sd": np.nan}

    # sklearn >=1.8 reports the `penalty` param as the string "deprecated"; the estimator still
    # applies L2 at C=1.0. Name it explicitly so the shrinkage is never invisible in the output.
    penalty = lr.get_params().get("penalty")
    penalty = "l2" if penalty in (None, "deprecated") else penalty

    return {"pipeline": pipe, "estimator": lr, "intercept": intercept, "beta": beta,
            "coefficients": coefficients, "X": X, "y": y,
            "feature_means": model[FEATURES].mean().to_dict(),
            "penalty": f"{penalty}, C={lr.C}, solver={lr.solver}",
            "imputed_cells": int(np.isnan(X).sum())}


def wald_se(X: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Observed-information SEs for [intercept, *FEATURES]. Approximate: the shipped fit is
    L2-penalized (sklearn default C=1.0), so these understate shrinkage. bootstrap_validation()
    is the primary uncertainty statement; these are here to be comparable with textbook output."""
    design = np.column_stack([np.ones(len(X)), X])
    w = p * (1 - p)
    try:
        cov = np.linalg.inv(design.T @ (design * w[:, None]))
        return np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        return np.full(design.shape[1], np.nan)


# ── decomposition ────────────────────────────────────────────────────────────
def decompose_prediction(model: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """One row per attempt, every term of the linear predictor spelled out.

      c_<f> = beta_f * x_f                raw contribution;  intercept + SUM c_<f> == logit
      d_<f> = beta_f * (x_f - mean_f)     centered;  logit_at_mean + SUM d_<f> == logit

    Raw contributions are dominated by constants (pop_faced alone carries ~+11.7 logits at a
    league-average pop time), so d_<f> is the column that answers "why THIS attempt" — it is
    the movement away from an average attempt, and it is what top_driver ranks."""
    beta, intercept = fit["beta"], fit["intercept"]
    means = fit["feature_means"]

    d = model[IDENTITY + FEATURES + ["y"]].copy()
    d.insert(0, "attempt_ix", model.index)

    d["intercept"] = intercept
    for f in FEATURES:
        d[f"c_{f}"] = beta[f] * d[f]
    for f in FEATURES:
        d[f"d_{f}"] = beta[f] * (d[f] - means[f])

    d["logit_at_mean"] = intercept + sum(beta[f] * means[f] for f in FEATURES)
    d["logit_manual"]  = d["intercept"] + d[[f"c_{f}" for f in FEATURES]].sum(axis=1)
    d["p_manual"]      = 1.0 / (1.0 + np.exp(-d["logit_manual"]))
    d["yhat_manual"]   = (d["p_manual"] >= 0.5).astype(int)

    d["logit_sklearn"] = fit["estimator"].decision_function(fit["X"])
    d["p_sklearn"]     = fit["estimator"].predict_proba(fit["X"])[:, 1]
    d["yhat_sklearn"]  = fit["estimator"].predict(fit["X"])

    d["abs_logit_diff"] = (d["logit_manual"] - d["logit_sklearn"]).abs()
    d["abs_p_diff"]     = (d["p_manual"] - d["p_sklearn"]).abs()

    centered = d[[f"d_{f}" for f in FEATURES]]
    d["top_driver"] = centered.abs().idxmax(axis=1).str.removeprefix("d_")
    d["top_driver_logits"] = centered.to_numpy()[
        np.arange(len(d)), centered.abs().to_numpy().argmax(axis=1)]
    return d


def verify_predictions(dec: pd.DataFrame, fit: dict, tol: float = 1e-9) -> dict:
    """Both decomposition identities and the sklearn/manual agreement, asserted not eyeballed."""
    beta, means = fit["beta"], fit["feature_means"]
    raw_close = (dec["intercept"] + dec[[f"c_{f}" for f in FEATURES]].sum(axis=1)
                 - dec["logit_manual"]).abs().max()
    ctr_close = (dec["logit_at_mean"] + dec[[f"d_{f}" for f in FEATURES]].sum(axis=1)
                 - dec["logit_manual"]).abs().max()
    out = {
        "n": len(dec),
        "raw_closure_max_resid":      float(raw_close),
        "centered_closure_max_resid": float(ctr_close),
        "max_abs_logit_diff":         float(dec["abs_logit_diff"].max()),
        "max_abs_prob_diff":          float(dec["abs_p_diff"].max()),
        "class_mismatches":           int((dec["yhat_manual"] != dec["yhat_sklearn"]).sum()),
        "imputed_cells":              fit["imputed_cells"],
        "worst_row":                  int(dec["abs_p_diff"].idxmax()),
    }
    assert out["raw_closure_max_resid"] < tol,      "raw contributions must sum to the logit"
    assert out["centered_closure_max_resid"] < tol, "centered contributions must sum to the logit"
    assert out["max_abs_prob_diff"] < 1e-12,        "manual probability must equal sklearn's"
    assert out["class_mismatches"] == 0,            "manual and sklearn classes must agree"
    assert out["imputed_cells"] == 0,               "imputer must be a no-op for this to hold"
    return out


def predict_one(fit: dict, **x) -> pd.DataFrame:
    """Hand-check a single hypothetical attempt against the site calculator, term by term:

        predict_one(fit, sprint_speed=24.4, lead_at_firstmove_ft=12.0,
                    gain_to_release_ft=14.0, pop_faced=1.95)

    Returns the same terms decompose_prediction() produces, one row per term plus the total."""
    missing = set(FEATURES) - set(x)
    if missing:
        raise ValueError(f"missing feature(s): {sorted(missing)}")
    beta, means = fit["beta"], fit["feature_means"]
    rows = [{"term": "intercept", "value": np.nan, "beta": np.nan, "mean": np.nan,
             "contribution": fit["intercept"], "centered_contribution": np.nan}]
    rows += [{"term": f, "value": float(x[f]), "beta": beta[f], "mean": means[f],
              "contribution": beta[f] * float(x[f]),
              "centered_contribution": beta[f] * (float(x[f]) - means[f])} for f in FEATURES]
    logit = sum(r["contribution"] for r in rows)
    rows.append({"term": "TOTAL logit", "value": np.nan, "beta": np.nan, "mean": np.nan,
                 "contribution": logit,
                 "centered_contribution": sum(r["centered_contribution"] for r in rows[1:])})
    rows.append({"term": "P(safe)", "value": np.nan, "beta": np.nan, "mean": np.nan,
                 "contribution": 1.0 / (1.0 + np.exp(-logit)), "centered_contribution": np.nan})
    out = pd.DataFrame(rows)
    # sklearn on the same input must agree, or predict_one and the fitted model have diverged
    sk = fit["estimator"].predict_proba(np.array([[x[f] for f in FEATURES]], float))[0, 1]
    assert abs(sk - out["contribution"].iloc[-1]) < 1e-12, "predict_one disagrees with sklearn"
    return out


def verify_reference(fit: dict, metrics: dict, tol: float = 5e-5) -> pd.DataFrame:
    """This module must land on the committed fit, or the decomposition describes a different
    model than the one the site and the report quote."""
    rows = [{"quantity": "intercept", "reference": REFERENCE["intercept"], "here": fit["intercept"]}]
    rows += [{"quantity": f, "reference": REFERENCE[f], "here": fit["beta"][f]} for f in FEATURES]
    rows += [{"quantity": "OOF AUROC", "reference": REFERENCE["auc"], "here": metrics["auc"]},
             {"quantity": "n", "reference": REFERENCE["n"], "here": metrics["n"]}]
    out = pd.DataFrame(rows)
    out["abs_diff"] = (out["here"] - out["reference"]).abs()
    out["match"] = out["abs_diff"] <= tol
    assert out["match"].all(), f"drifted from the committed fit:\n{out.to_string(index=False)}"
    return out


# ── calibration (the shipped layer) ─────────────────────────────────────────
def calibrated_model(model: pd.DataFrame, seed: int = SEED) -> dict:
    """NESTED out-of-fold predictions of the CALIBRATED model — the honest scoring of the layer
    that actually ships to the browser.

    The raw logistic discriminates well and states a poor rate: §5.3's decile table reads ~0.876
    where the observed rate is 0.919, five deciles beyond two binomial SE. The web calculator
    prints that number as a probability, so the miscalibration is the user-facing claim.

    The map must be scored by nesting or the result is self-congratulatory: an isotonic fit can
    memorise the rows it is then measured on. So the outer 5-fold holds out a test fold, an inner
    5-fold inside the training side produces clean predictions, the map is fit on those, and only
    then is it applied to the untouched outer fold.

    Isotonic rather than Platt because the miscalibration is a WAVE (over-confident in deciles
    2-3, under-confident in 6-8, reversing at 9) whose signed gaps cancel to -0.0000. A single
    sigmoid slope cannot bend that shape and measurably makes it worse; a monotone step map can.

    NOTE this does not disturb anything else in this file: isotonic is a post-hoc map on the
    model's OUTPUT, so the coefficients, the odds ratios, the linear-predictor decomposition and
    verify_reference() are all untouched. Only the probabilities move."""
    from sklearn.isotonic import IsotonicRegression

    X = model[FEATURES].to_numpy(float)
    y = model["y"].to_numpy(int)
    out = np.zeros(len(y))
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        lr = LogisticRegression(max_iter=5000).fit(X[tr], y[tr])
        inner_oof = np.zeros(len(tr))
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
        for itr, ite in inner.split(X[tr], y[tr]):
            li = LogisticRegression(max_iter=5000).fit(X[tr][itr], y[tr][itr])
            inner_oof[ite] = li.predict_proba(X[tr][ite])[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip").fit(inner_oof, y[tr])
        out[te] = ir.predict(lr.predict_proba(X[te])[:, 1])

    grid = np.linspace(0.05, 0.95, 91)
    f1s = np.array([f1_score(y, (out >= t).astype(int)) for t in grid])
    best = int(f1s.argmax())
    return {"oof": pd.Series(out, index=model.index, name="p_oof_cal"),
            "auc": float(roc_auc_score(y, out)),
            "auprc": float(average_precision_score(y, out)),
            "brier": float(brier_score_loss(y, out)),
            "f1_best": float(f1s[best]), "f1_threshold": float(grid[best])}


# ── evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model: pd.DataFrame, seed: int = SEED) -> dict:
    """5-fold out-of-fold predictions — the AUROC the project quotes. Fold ids are returned so
    any single attempt's prediction can be traced to the fold that produced it."""
    X = model[FEATURES].to_numpy(float)
    y = model["y"].to_numpy(int)
    pipe = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    fold = np.empty(len(y), dtype=int)
    for k, (_, test_ix) in enumerate(cv.split(X, y)):
        fold[test_ix] = k

    grid = np.linspace(0.05, 0.95, 91)
    f1s = np.array([f1_score(y, (oof >= t).astype(int)) for t in grid])
    best = int(f1s.argmax())

    per_fold = pd.DataFrame([
        {"fold": k, "n": int((fold == k).sum()), "base_rate": float(y[fold == k].mean()),
         "auc": float(roc_auc_score(y[fold == k], oof[fold == k]))} for k in range(5)])

    return {"oof": pd.Series(oof, index=model.index, name="p_oof"),
            "fold": pd.Series(fold, index=model.index, name="fold"),
            "per_fold": per_fold, "n": len(y),
            "auc": float(roc_auc_score(y, oof)),
            "auprc": float(average_precision_score(y, oof)),
            "auprc_floor": float(y.mean()),      # a no-skill classifier's AUPRC IS the base rate
            "brier": float(brier_score_loss(y, oof)),
            "f1_best": float(f1s[best]), "f1_threshold": float(grid[best]),
            "base_rate": float(y.mean())}


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Decile bins on predicted probability. binomial_se sets the scale a gap has to clear
    before it counts as miscalibration rather than bin noise."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    rows = []
    for b in sorted(pd.unique(q)):
        m = q == b
        obs, n = float(y[m].mean()), int(m.sum())
        rows.append({"bin": int(b) + 1, "n": n,
                     "p_pred_mean": float(p[m].mean()), "p_pred_lo": float(p[m].min()),
                     "p_pred_hi": float(p[m].max()), "observed": obs,
                     "gap": obs - float(p[m].mean()),
                     "binomial_se": float(np.sqrt(max(obs * (1 - obs), 1e-12) / n))})
    out = pd.DataFrame(rows)
    out["gap_in_se"] = out["gap"] / out["binomial_se"]
    return out


def calibration_error(cal: pd.DataFrame) -> dict:
    """ECE = the n-weighted mean |gap| over the bins; max_gap is the single worst bin. Kept as a
    scalar next to the table so miscalibration is quotable, not just visible."""
    w = cal["n"] / cal["n"].sum()
    return {"ece": float((w * cal["gap"].abs()).sum()),
            "max_abs_gap": float(cal["gap"].abs().max()),
            "worst_bin": int(cal.loc[cal["gap"].abs().idxmax(), "bin"]),
            "bins_beyond_2se": int((cal["gap_in_se"].abs() > 2).sum())}


# ── resampling-based validation ──────────────────────────────────────────────
def bootstrap_validation(model: pd.DataFrame, n_boot: int = 1000,
                         seed: int = BOOT_SEED) -> dict:
    """Coefficient stability and an AUROC interval from the same resamples. AUROC is scored on
    the OUT-OF-BAG rows of each draw, so it is not an in-sample number."""
    X = model[FEATURES].to_numpy(float)
    y = model["y"].to_numpy(int)
    rng = np.random.default_rng(seed)
    n = len(y)

    coefs, aucs = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        if len(np.unique(y[ix])) < 2:
            continue
        lr = LogisticRegression(max_iter=5000).fit(X[ix], y[ix])
        coefs.append(np.concatenate([lr.intercept_, lr.coef_[0]]))
        oob = np.setdiff1d(np.arange(n), np.unique(ix))
        if len(oob) and len(np.unique(y[oob])) > 1:
            aucs.append(roc_auc_score(y[oob], lr.predict_proba(X[oob])[:, 1]))

    C = np.array(coefs)
    terms = ["intercept"] + FEATURES
    coef_ci = pd.DataFrame({
        "term": terms, "mean": C.mean(axis=0), "sd": C.std(axis=0, ddof=1),
        "ci_lo": np.percentile(C, 2.5, axis=0), "ci_hi": np.percentile(C, 97.5, axis=0)})
    coef_ci["sign_stable"] = (np.sign(coef_ci["ci_lo"]) == np.sign(coef_ci["ci_hi"]))
    coef_ci["cv_abs"] = coef_ci["sd"] / coef_ci["mean"].abs()

    a = np.array(aucs)
    return {"n_boot": len(C), "coef_ci": coef_ci, "auc_draws": a,
            "auc_mean": float(a.mean()), "auc_sd": float(a.std(ddof=1)),
            "auc_lo": float(np.percentile(a, 2.5)), "auc_hi": float(np.percentile(a, 97.5))}


def repeated_cv(model: pd.DataFrame, n_repeats: int = 10, seed: int = SEED) -> dict:
    """Spread of the headline AUROC across CV partitions — how much of 0.756 is the seed."""
    X, y = model[FEATURES].to_numpy(float), model["y"].to_numpy(int)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    scores = []
    for rep in range(n_repeats):
        oof = np.empty(len(y))
        for k, (tr, te) in enumerate(cv.split(X, y)):
            if k // 5 != rep:
                continue
            lr = LogisticRegression(max_iter=5000).fit(X[tr], y[tr])
            oof[te] = lr.predict_proba(X[te])[:, 1]
        scores.append(roc_auc_score(y, oof))
    s = np.array(scores)
    return {"n_repeats": n_repeats, "scores": s, "mean": float(s.mean()),
            "sd": float(s.std(ddof=1)), "min": float(s.min()), "max": float(s.max())}


def permutation_test(model: pd.DataFrame, n_perm: int = 200, seed: int = PERM_SEED,
                     observed: float | None = None) -> dict:
    """Label-permutation null for the OOF AUROC. The null is centred on 0.5 by construction;
    the point is the width of it at this n, which is what makes 0.756 interpretable."""
    X, y = model[FEATURES].to_numpy(float), model["y"].to_numpy(int)
    if observed is None:
        observed = evaluate_model(model)["auc"]
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pipe = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=5000))

    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        oof = cross_val_predict(pipe, X, yp, cv=cv, method="predict_proba")[:, 1]
        null[i] = roc_auc_score(yp, oof)
    n_ge = int((null >= observed).sum())
    return {"n_perm": n_perm, "observed": float(observed), "null": null,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95)), "n_null_ge_observed": n_ge,
            # the resolution floor: with n_perm shuffles no p below 1/(n_perm+1) is measurable
            "p_floor": 1 / (n_perm + 1), "p_value": (1 + n_ge) / (n_perm + 1)}


def learning_curve(model: pd.DataFrame, fractions=(0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0),
                   n_draws: int = 3, seed: int = CURVE_SEED) -> pd.DataFrame:
    """Is 10,844 attempts enough, or is the AUROC still climbing? Each fraction is a stratified
    subsample scored with the same 5-fold protocol, averaged over n_draws draws."""
    X, y = model[FEATURES].to_numpy(float), model["y"].to_numpy(int)
    rng = np.random.default_rng(seed)
    pipe = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=5000))
    rows = []
    for frac in fractions:
        scores, n_used = [], 0
        for d in range(1 if frac == 1.0 else n_draws):
            if frac == 1.0:
                ix = np.arange(len(y))
            else:
                ix = np.concatenate([rng.choice(np.where(y == c)[0],
                                                int(round(frac * (y == c).sum())), replace=False)
                                     for c in (0, 1)])
                ix.sort()
            n_used = len(ix)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + d)
            oof = cross_val_predict(pipe, X[ix], y[ix], cv=cv, method="predict_proba")[:, 1]
            scores.append(roc_auc_score(y[ix], oof))
        rows.append({"fraction": frac, "n": n_used, "auc_mean": float(np.mean(scores)),
                     "auc_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0})
    return pd.DataFrame(rows)


# ── the season-level decomposition (Net Bases = NetSpeed + Steal+) ───────────
def decompose_season() -> dict:
    """The other decomposition in the project, re-derived from the published league constants
    instead of trusted from the leaderboard columns:

        p_speed  = a0 + a1 * sprint_speed        success rate expected from SPEED ALONE
        NetSpeed = sb_attempts * (2*p_speed - 1) net bases the wheels alone buy
        Steal+   = net_sb - NetSpeed             net bases the SKILL adds

    Closure is exact by construction; recomputing it catches a leaderboard written by a
    different fit than the one in v15_players.json."""
    lb = pd.read_csv(RESULTS / "DF_v15_leaderboard.csv")
    meta = json.loads((DATA / "v15_players.json").read_text(encoding="utf-8"))["meta"]
    a0, a1 = meta["p_speed_fit"]["a0"], meta["p_speed_fit"]["a1"]

    d = lb.copy()
    d["p_speed"]          = a0 + a1 * d["sprint_speed"]
    d["netspeed_recomp"]  = d["sb_attempts"] * (2 * d["p_speed"] - 1)
    d["stealplus_recomp"] = d["net_sb"] - d["netspeed_recomp"]
    d["closure_resid"]    = d["netspeed"] + d["surplus"] - d["net_sb"]
    d["netspeed_resid"]   = d["netspeed"] - d["netspeed_recomp"]
    d["stealplus_resid"]  = d["steal_plus"] - d["stealplus_recomp"]

    checks = {"n": len(d),
              "p_speed_fit": {"a0": a0, "a1": a1},
              "max_closure_resid":   float(d["closure_resid"].abs().max()),
              "max_netspeed_resid":  float(d["netspeed_resid"].abs().max()),
              "max_stealplus_resid": float(d["stealplus_resid"].abs().max())}
    assert checks["max_closure_resid"] < 1e-9,   "Net Bases = NetSpeed + Steal+ must close exactly"
    assert checks["max_netspeed_resid"] < 1e-9,  "leaderboard NetSpeed must match the published fit"
    assert checks["max_stealplus_resid"] < 1e-9, "leaderboard Steal+ must match the published fit"
    return {"table": d, "checks": checks}


# ── orchestration ────────────────────────────────────────────────────────────
def run(n_boot: int = 1000, n_perm: int = 200, n_repeats: int = 10) -> dict:
    _stage("loading Raw_Attempts + sprint speed + pop time")
    data = load_data()
    model = data["model"]

    _stage(f"n = {len(model)} attempts; checks, VIF, fit, 5-fold OOF")
    checks  = validate_data(data["merged"], model)
    vif     = collinearity(model)
    fit     = fit_model(model)
    metrics = evaluate_model(model)
    dec     = decompose_prediction(model, fit)

    _stage("asserting the decomposition identities against sklearn")
    verification = verify_predictions(dec, fit)
    reference    = verify_reference(fit, metrics)

    dec["p_oof"] = metrics["oof"]
    dec["fold"]  = metrics["fold"]

    cal = calibration(model["y"].to_numpy(int), metrics["oof"].to_numpy())

    # the shipped calibration layer, scored by nesting, through the SAME two functions so the
    # before/after comparison is like-for-like
    _stage("nested isotonic calibration (the layer that ships)")
    calmodel = calibrated_model(model)
    cal_after = calibration(model["y"].to_numpy(int), calmodel["oof"].to_numpy())
    dec["p_oof_cal"] = calmodel["oof"]

    # the four resampling stages below are the slow ones — named here rather than inlined in the
    # return dict purely so each can be announced before it starts
    _stage(f"bootstrap: {n_boot} resamples")
    boot = bootstrap_validation(model, n_boot=n_boot)
    _stage(f"repeated CV: {n_repeats} partitions")
    rcv = repeated_cv(model, n_repeats=n_repeats)
    _stage(f"permutation null: {n_perm} label shuffles")
    perm = permutation_test(model, n_perm=n_perm, observed=metrics["auc"])
    _stage("learning curve")
    curve = learning_curve(model)
    _stage("season decomposition (Net Bases = NetSpeed + Steal+)")
    season = decompose_season()
    _stage("done")

    return {"data": data, "checks": checks, "collinearity": vif, "fit": fit,
            "coefficients": fit["coefficients"], "metrics": metrics,
            "decomposition": dec, "verification": verification, "reference": reference,
            "calibration": cal, "calibration_error": calibration_error(cal),
            "calibrated": calmodel, "calibration_after": cal_after,
            "calibration_error_after": calibration_error(cal_after),
            "bootstrap": boot,
            "repeated_cv": rcv,
            "permutation": perm,
            "learning_curve": curve,
            "season": season}


def sample_rows(dec: pd.DataFrame, n: int = 8, seed: int = SEED) -> pd.DataFrame:
    """A spread of attempts across the probability range — extremes plus a random middle — so a
    hand-check covers the tails and not only the mode."""
    ordered = dec.sort_values("p_manual")
    edges = pd.concat([ordered.head(2), ordered.tail(2)])
    middle = ordered.iloc[2:-2].sample(min(n - 4, len(ordered) - 4), random_state=seed)
    return pd.concat([edges, middle]).sort_values("p_manual")


def report(audit: dict) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 60)
    m, fit, dec = audit["metrics"], audit["fit"], audit["decomposition"]

    print("=== INGESTION ===")
    print(audit["data"]["ingestion"].to_string(index=False))

    print("\n=== CLASS BALANCE ===")
    print(audit["checks"]["class_balance"].round(4).to_string(index=False))

    print("\n=== MISSINGNESS (pre-drop) ===")
    print(audit["checks"]["missingness"].round(2).to_string(index=False))

    print("\n=== FEATURE DISTRIBUTIONS ===")
    print(audit["checks"]["distributions"].round(3).to_string())

    print("\n=== EDGE CASES ===")
    print(audit["checks"]["edge_cases"].to_string(index=False))

    print("\n=== COLLINEARITY (shipped 4) ===")
    print(audit["collinearity"]["vif_shipped"].round(4).to_string(index=False))
    print("--- with lead_at_release_ft added back (why it is not an input) ---")
    print(audit["collinearity"]["vif_with_redundant"].round(6).to_string(index=False))

    print(f"\n=== COEFFICIENTS  (penalty {fit['penalty']}) ===")
    print(audit["coefficients"].round(4).to_string(index=False))

    print("\n=== REFERENCE FIT CHECK ===")
    print(audit["reference"].round(6).to_string(index=False))

    print(f"\nAUROC = {m['auc']:.4f} | AUPRC = {m['auprc']:.4f} (floor {m['auprc_floor']:.4f}) "
          f"| Brier = {m['brier']:.4f} | F1 = {m['f1_best']:.4f} @ thr {m['f1_threshold']:.2f} "
          f"| n = {m['n']}")
    print(audit["metrics"]["per_fold"].round(4).to_string(index=False))

    v = audit["verification"]
    print(f"\nmanual vs sklearn: max |logit diff| = {v['max_abs_logit_diff']:.3e} | "
          f"max |p diff| = {v['max_abs_prob_diff']:.3e} | class mismatches = {v['class_mismatches']}")
    print(f"closure: raw {v['raw_closure_max_resid']:.3e} | centered "
          f"{v['centered_closure_max_resid']:.3e} | imputed cells {v['imputed_cells']}")

    print("\n=== ROW-LEVEL DECOMPOSITION (8 attempts across the probability range) ===")
    s = sample_rows(dec)
    print(s[["attempt_ix", "season", "base", "result"] + FEATURES].round(2).to_string(index=False))
    print("\n--- raw contributions: intercept + sum(c_*) == logit ---")
    print(s[["attempt_ix", "intercept"] + [f"c_{f}" for f in FEATURES]
            + ["logit_manual", "p_manual", "p_sklearn", "yhat_manual", "y"]]
          .round(4).to_string(index=False))
    print("\n--- centered contributions: logit_at_mean + sum(d_*) == logit ---")
    print(s[["attempt_ix", "logit_at_mean"] + [f"d_{f}" for f in FEATURES]
            + ["logit_manual", "top_driver", "top_driver_logits", "p_oof", "fold"]]
          .round(4).to_string(index=False))

    med = audit["data"]["model"][FEATURES].median()
    print("\n=== SINGLE-ATTEMPT HAND CHECK (league-median attempt) ===")
    print(predict_one(fit, **med.to_dict()).round(4).to_string(index=False))

    print("\n=== CALIBRATION (OOF deciles) — RAW ===")
    print(audit["calibration"].round(4).to_string(index=False))
    ce = audit["calibration_error"]
    print(f"ECE = {ce['ece']:.4f} | worst bin {ce['worst_bin']} gap {ce['max_abs_gap']:.4f} | "
          f"{ce['bins_beyond_2se']}/10 bins beyond 2 binomial SE")

    print("\n=== CALIBRATION (OOF deciles) — AFTER ISOTONIC (nested; this is what ships) ===")
    print(audit["calibration_after"].round(4).to_string(index=False))
    ca, cm = audit["calibration_error_after"], audit["calibrated"]
    print(f"ECE = {ca['ece']:.4f} | worst bin {ca['worst_bin']} gap {ca['max_abs_gap']:.4f} | "
          f"{ca['bins_beyond_2se']}/10 bins beyond 2 binomial SE")
    print(f"  raw -> calibrated:  ECE {ce['ece']:.4f} -> {ca['ece']:.4f}  |  "
          f"bins beyond 2 SE {ce['bins_beyond_2se']} -> {ca['bins_beyond_2se']}  |  "
          f"AUROC {m['auc']:.4f} -> {cm['auc']:.4f}  |  Brier {m['brier']:.4f} -> {cm['brier']:.4f}")
    print("  the AUROC cost is the step map creating ties; it is inside the bootstrap CI below.")

    b = audit["bootstrap"]
    print(f"\n=== BOOTSTRAP ({b['n_boot']} resamples, seed {BOOT_SEED}) ===")
    print(b["coef_ci"].round(4).to_string(index=False))
    print(f"out-of-bag AUROC = {b['auc_mean']:.4f} (SD {b['auc_sd']:.4f}, "
          f"95% CI {b['auc_lo']:.4f}-{b['auc_hi']:.4f})")

    r = audit["repeated_cv"]
    print(f"\nrepeated 5-fold CV ({r['n_repeats']} partitions): AUROC {r['mean']:.4f} "
          f"(SD {r['sd']:.4f}, range {r['min']:.4f}-{r['max']:.4f})")

    p = audit["permutation"]
    print(f"permutation null ({p['n_perm']} label shuffles): {p['null_mean']:.4f} "
          f"(SD {p['null_sd']:.4f}, p95 {p['null_p95']:.4f}) vs observed {p['observed']:.4f} "
          f"-> {p['n_null_ge_observed']} of {p['n_perm']} nulls >= observed, "
          f"p = {p['p_value']:.4g} (floor {p['p_floor']:.4g})")

    print("\n=== LEARNING CURVE ===")
    print(audit["learning_curve"].round(4).to_string(index=False))

    sea = audit["season"]
    print(f"\n=== SEASON DECOMPOSITION — Net Bases = NetSpeed + Steal+  (n={sea['checks']['n']}) ===")
    print(f"p_speed = {sea['checks']['p_speed_fit']['a0']:.4f} + "
          f"{sea['checks']['p_speed_fit']['a1']:.5f} * sprint_speed | max closure resid "
          f"{sea['checks']['max_closure_resid']:.3e} | max recompute resid "
          f"{max(sea['checks']['max_netspeed_resid'], sea['checks']['max_stealplus_resid']):.3e}")
    top = sea["table"].sort_values("net_sb", ascending=False).head(8)
    print(top[["player_name", "season", "sprint_speed", "sb_attempts", "net_sb",
               "netspeed_recomp", "stealplus_recomp", "closure_resid", "burst_ft"]]
          .round(3).to_string(index=False))


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="fewer resamples (200 bootstrap / 50 permutations / 3 CV repeats)")
    args = ap.parse_args()
    audit = (run(n_boot=200, n_perm=50, n_repeats=3) if args.quick else run())
    report(audit)
    return audit


if __name__ == "__main__":
    main()
