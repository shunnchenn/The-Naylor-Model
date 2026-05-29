#!/usr/bin/env python3
"""
evaluate.py  —  Accuracy / repeatability / coverage + go-no-go verdict
======================================================================

Joins the automatic detector output (pilot_results.csv) against the manual
ground-truth labels (labels_manual.csv) and answers the only question the
pilot exists to answer:  is CV-derived pitch delivery time reliable enough
to trust a per-pitcher average?

Reports, and writes to PILOT_FINDINGS.md:
  • Accuracy     — release & first-move frame error (frames + ms), bias, MAE;
                   a fitted constant release offset (peak-vs-true correction).
  • Repeatability— within-pitcher delivery std across that pitcher's clips.
  • Coverage     — % of clips that yield a usable auto measurement.
  • Cross-check  — per-pitcher mean vs. a scout-TTP-minus-flight sanity band.
  • Verdict      — GO / NO-GO against the gate, reported honestly either way.

Gate (from the plan):
    release error  ≤ 2 frames   (≈ ±66 ms @ 30 fps)
    within-pitcher delivery std ≤ 0.10 s
    coverage       ≥ 70 %

Usage:
    python3 cv_pilot/evaluate.py
"""

from __future__ import annotations
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "pilot_results.csv"
LABELS_CSV  = ROOT / "labels_manual.csv"
FINDINGS_MD = ROOT / "PILOT_FINDINGS.md"

# Gate thresholds
GATE_REL_FRAMES = 2.0
GATE_WITHIN_STD = 0.10
GATE_COVERAGE   = 0.70

# Cross-check: typical first-move->release for MLB ~ 0.95-1.45 s; flight ~0.40 s.
SANITY_BAND = (0.85, 1.55)


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    if not RESULTS_CSV.exists():
        raise SystemExit(f"Missing {RESULTS_CSV}. Run extract_delivery.py first.")
    res = pd.read_csv(RESULTS_CSV)

    n_total = len(res)
    usable = res[res.get("usable", False) == True] if "usable" in res else res
    coverage = len(usable) / n_total if n_total else 0.0

    lines = []
    lines.append("# Pitch Delivery CV Pilot — Findings")
    lines.append("")
    lines.append(f"_Generated {date.today().isoformat()}_")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Clips processed: **{n_total}**")
    lines.append(f"- Usable auto measurements (events found pre-cut, in-band, "
                 f"confident): **{len(usable)}**")
    lines.append(f"- **Coverage = {coverage*100:.1f}%**  "
                 f"(gate ≥ {GATE_COVERAGE*100:.0f}%)")
    lines.append("")

    # ── Accuracy vs manual labels ───────────────────────────────────────────
    rel_mae = fm_mae = rel_bias = rel_off = None
    have_labels = LABELS_CSV.exists()
    merged = pd.DataFrame()
    if have_labels:
        lab = pd.read_csv(LABELS_CSV)
        merged = res.merge(lab, on="clip_id", how="inner", suffixes=("", "_lab"))
        merged = merged.dropna(subset=["release_frame", "manual_release_frame"])

    lines.append("## Accuracy (auto vs. manual ground truth)")
    lines.append("")
    if not have_labels or merged.empty:
        lines.append("_No manual labels yet — run `label_tool.py` to build "
                     "ground truth, then re-run this script._")
        lines.append("")
    else:
        rel_err = merged["release_frame"] - merged["manual_release_frame"]
        fm_err  = merged["first_move_frame"] - merged["manual_first_move_frame"]
        rel_mae  = float(rel_err.abs().mean())
        rel_bias = float(rel_err.mean())
        fm_mae   = float(fm_err.abs().mean())
        rel_off  = rel_bias                       # constant correction to apply
        fps_mean = float(merged["fps"].mean())
        ms = 1000.0 / fps_mean

        lines.append(f"- n labeled clips: **{len(merged)}**  (mean {fps_mean:.1f} fps)")
        lines.append(f"- Release frame error — MAE **{rel_mae:.2f} frames "
                     f"(≈ {rel_mae*ms:.0f} ms)**, bias {rel_bias:+.2f} frames")
        lines.append(f"- First-move frame error — MAE **{fm_mae:.2f} frames "
                     f"(≈ {fm_mae*ms:.0f} ms)**")
        lines.append(f"- Fitted constant release offset (subtract from auto): "
                     f"**{rel_off:+.2f} frames** — applying it would reduce "
                     f"systematic bias.")
        # after constant-offset correction
        corr_mae = float((rel_err - rel_off).abs().mean())
        lines.append(f"- Release MAE after offset correction: **{corr_mae:.2f} frames**")
        lines.append("")
        lines.append("| clip | auto rel | manual rel | err (frames) | auto delivery_s | manual delivery_s |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in merged.iterrows():
            lines.append(
                f"| {r['clip_id']} | {fmt(r['release_frame'],1)} | "
                f"{fmt(r['manual_release_frame'],1)} | "
                f"{fmt(r['release_frame']-r['manual_release_frame'],1)} | "
                f"{fmt(r.get('delivery_s'),3)} | "
                f"{fmt(r.get('manual_delivery_s'),3)} |")
        lines.append("")

    # ── Repeatability (within-pitcher std) ──────────────────────────────────
    lines.append("## Repeatability (within-pitcher delivery std)")
    lines.append("")
    worst_std = None
    key = "pitcher_name" if "pitcher_name" in usable.columns else "pitcher_id"
    if key in usable.columns and "delivery_s" in usable.columns and len(usable):
        grp = (usable.dropna(subset=["delivery_s"])
                     .groupby(key)["delivery_s"]
                     .agg(["count", "mean", "std"]))
        if not grp.empty:
            lines.append(f"| pitcher | n | mean delivery_s | std (s) |")
            lines.append("|---|---|---|---|")
            for name, r in grp.iterrows():
                lines.append(f"| {name} | {int(r['count'])} | "
                             f"{fmt(r['mean'],3)} | {fmt(r['std'],3)} |")
            multi = grp[grp["count"] >= 2]
            if not multi.empty:
                worst_std = float(multi["std"].max())
                lines.append("")
                lines.append(f"- Worst within-pitcher std (n≥2): "
                             f"**{worst_std:.3f} s**  (gate ≤ {GATE_WITHIN_STD:.2f} s)")
            lines.append("")
    else:
        lines.append("_Need ≥2 usable clips per pitcher to assess repeatability._")
        lines.append("")

    # ── External sanity band ────────────────────────────────────────────────
    lines.append("## External sanity check")
    lines.append("")
    if "delivery_s" in usable.columns and len(usable):
        d = usable["delivery_s"].dropna()
        in_band = ((d >= SANITY_BAND[0]) & (d <= SANITY_BAND[1])).mean() if len(d) else 0
        lines.append(f"- Usable deliveries within plausible MLB band "
                     f"{SANITY_BAND[0]:.2f}–{SANITY_BAND[1]:.2f}s: "
                     f"**{in_band*100:.0f}%**")
        lines.append(f"- Median usable delivery_s: **{fmt(d.median(),3)}** "
                     f"(scout TTP ≈ delivery + ~0.40s ball-flight)")
        lines.append("")

    # ── Verdict ─────────────────────────────────────────────────────────────
    pass_cov = coverage >= GATE_COVERAGE
    pass_acc = (rel_mae is not None) and (rel_mae <= GATE_REL_FRAMES)
    pass_rep = (worst_std is not None) and (worst_std <= GATE_WITHIN_STD)
    checks = []
    checks.append(("coverage ≥ 70%", pass_cov, f"{coverage*100:.1f}%"))
    checks.append(("release MAE ≤ 2 frames", pass_acc,
                   fmt(rel_mae, 2) + " frames" if rel_mae is not None else "no labels"))
    checks.append(("within-pitcher std ≤ 0.10s", pass_rep,
                   fmt(worst_std, 3) + " s" if worst_std is not None else "insufficient n"))

    go = pass_cov and (pass_acc is True) and (pass_rep is True)
    verdict = "GO" if go else ("NO-GO" if (rel_mae is not None and worst_std is not None)
                               else "INCOMPLETE (need manual labels / more clips)")

    lines.append("## Verdict")
    lines.append("")
    for name, ok, val in checks:
        mark = "✅" if ok else ("❌" if ok is False else "⚠️")
        lines.append(f"- {mark} {name}: {val}")
    lines.append("")
    lines.append(f"### → **{verdict}**")
    lines.append("")
    if verdict == "GO":
        lines.append("Extraction is reliable enough to build per-pitcher averages. "
                     "Proceed to the scaling step (fetch_clips.py → DF_PitcherDelivery.csv → "
                     "replace LEAGUE_PITCHER_TTP in v6_explore.py).")
    elif verdict.startswith("NO-GO"):
        lines.append("Extraction is **not** reliable enough at this resolution. "
                     "The model keeps the LEAGUE_PITCHER_TTP=1.30 constant unchanged. "
                     "This is a clean negative result, not a bug — see the error "
                     "numbers above. Options: higher-fps clips, better release model, "
                     "or accept per-pitcher means with wider uncertainty.")
    else:
        lines.append("Run `label_tool.py` on all clips and/or add more clips per "
                     "pitcher, then re-run `evaluate.py` for a final verdict.")
    lines.append("")

    FINDINGS_MD.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[write] {FINDINGS_MD}")


if __name__ == "__main__":
    main()
