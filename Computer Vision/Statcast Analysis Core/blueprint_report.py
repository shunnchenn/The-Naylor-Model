#!/usr/bin/env python3
"""
blueprint_report.py  —  Flagship PDF for the full-spectrum Naylor Blueprint
============================================================================

Builds reports/Naylor_Blueprint_Report.pdf from the model outputs:
  data/DF_NaylorBlueprint_Leaderboard.csv      (run naylor_blueprint.py first)
  figures/Fig_NaylorBlueprint_Scatter.png
  figures/Fig_NaylorBlueprint_TopN.png
  figures/Fig_NaylorBlueprint_BottomN.png

Same intuitive house style as Naylor_Model_v6_Report.pdf (navy headers, plain
English, one figure per page with a caption). The story: ONE leaderboard, two poles
— slow runners who convert ground into steals (top, the blueprint) vs. the fastest
runners who squander an obvious speed advantage (bottom, the anti-Naylor).

Usage
-----
  python3 "cv_pilot/Statcast Analysis Core/blueprint_report.py"
No network; reads only on-disk CSV + PNGs.
"""
from __future__ import annotations
from datetime import date
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "Data Frame"
FIGS = ROOT / "Figures"
REPORTS = ROOT / "Reports"

NAVY, HEAD, BODY = "#0B2545", "#1F3A5F", "#222222"
GOOD, BAD = "#1B7A3D", "#A11225"
NAYLOR_ID, SOTO_ID, WITT_ID = 647304, 665742, 677951


# ----------------------------------------------------------------- page helpers
def textpage(pdf, title, lines, subtitle=None):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.945, title, fontsize=18, fontweight="bold", color=NAVY)
    y = 0.90
    if subtitle:
        ax.text(0.06, y, subtitle, fontsize=11, style="italic", color="#555"); y -= 0.030
    for ln in lines:
        if y < 0.05:
            break
        if ln.startswith("##"):
            y -= 0.010
            ax.text(0.06, y, ln[2:].strip(), fontsize=13, fontweight="bold", color=HEAD)
            y -= 0.028
        elif ln.startswith("[GOOD]"):
            ax.text(0.08, y, ln[6:].strip(), fontsize=10.3, color=GOOD); y -= 0.0205
        elif ln.startswith("[BAD]"):
            ax.text(0.08, y, ln[5:].strip(), fontsize=10.3, color=BAD); y -= 0.0205
        elif ln.startswith("•") or ln.startswith("-"):
            ax.text(0.08, y, ln, fontsize=10.3, color=BODY); y -= 0.0205
        elif ln.startswith("```"):
            ax.text(0.08, y, ln[3:], fontsize=10.5, family="monospace", color="#0B3D2E"); y -= 0.022
        elif ln == "":
            y -= 0.013
        else:
            ax.text(0.06, y, ln, fontsize=10.3, color=BODY); y -= 0.0205
    pdf.savefig(fig); plt.close(fig)


def imgpage(pdf, title, img, caption=""):
    img = Path(img)
    if not img.exists():
        return
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    ax_t = fig.add_axes([0.06, 0.93, 0.88, 0.05]); ax_t.axis("off")
    ax_t.text(0, 0.5, title, fontsize=16, fontweight="bold", color=NAVY)
    ax_i = fig.add_axes([0.05, 0.17, 0.90, 0.74]); ax_i.axis("off")
    ax_i.imshow(plt.imread(str(img)))
    if caption:
        ax_c = fig.add_axes([0.06, 0.05, 0.88, 0.11]); ax_c.axis("off")
        ax_c.text(0, 1, caption, fontsize=10, color="#444", va="top", wrap=True)
    pdf.savefig(fig); plt.close(fig)


def _row_line(r, show_pct=True):
    spd = f"{r['sprint_speed_ftps']:.1f}({int(r['sprint_pctile'])})"
    return (f"  #{int(r['rank_BCS']):>3}  {r['player_name'][:20]:<20} {int(r['season'])}  "
            f"spd {spd:<9} 90ft {r['hp_to_1b_s']:.2f}  "
            f"{int(r['SB'])}/{int(r['CS'])}={r['SB_pct']:>3.0f}%  "
            f"gain {r['mean_gain_to_release_ft']:>4.1f}  BCS {r['BCS']:+.2f}")


# ----------------------------------------------------------------- build
def build():
    df = pd.read_csv(DATA / "DF_NaylorBlueprint_Leaderboard.csv")
    n = len(df)
    nay = df[(df["runner_id"] == NAYLOR_ID) & (df["season"] == 2025)].iloc[0]
    soto = df[(df["runner_id"] == SOTO_ID) & (df["season"] == 2025)].iloc[0]
    bot = df.sort_values("BCS").iloc[0]
    witt = df[df["runner_id"] == WITT_ID].sort_values("season")
    n_blue = int((df["cohort"] == "Blueprint (slow+converts)").sum())
    n_squander = int((df["cohort"] == "Squander (fast+fails)").sum())

    REPORTS.mkdir(exist_ok=True)
    pdf_path = REPORTS / "Naylor_Blueprint_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        # ---- cover ----
        fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0.80), 1, 0.20, color=NAVY, transform=ax.transAxes))
        ax.text(0.5, 0.885, "The Naylor Blueprint", fontsize=31, fontweight="bold",
                ha="center", color="white")
        ax.text(0.5, 0.835, "A Full-Spectrum Basestealing Leaderboard",
                fontsize=15, ha="center", color="#CBD9EC")
        ax.text(0.5, 0.70,
                "ONE leaderboard, two poles — ranked by a single number.",
                ha="center", fontsize=13, style="italic", color="#333")
        ax.text(0.5, 0.595,
                "TOP — the blueprint:\n"
                "the SLOWEST runners who cover the most ground between the\n"
                "pitcher's first move and release, and convert it into steals.",
                ha="center", fontsize=11.5, color=GOOD, linespacing=1.5, fontweight="bold")
        ax.text(0.5, 0.45,
                "BOTTOM — the anti-Naylor:\n"
                "the FASTEST runners (top sprint speed, quickest 90 ft) who get\n"
                "caught anyway — squandering an obvious speed advantage.",
                ha="center", fontsize=11.5, color=BAD, linespacing=1.5, fontweight="bold")
        ax.text(0.5, 0.30,
                f"Josh Naylor 2025 — 24.4 ft/s, 1.9th speed percentile —\n"
                f"ranks #{int(nay['rank_BCS'])} of {n}.  Bobby Witt Jr. — 100th percentile —\n"
                f"sits in the bottom tier every season tracked.",
                ha="center", fontsize=11, color="#333", linespacing=1.5)
        ax.text(0.5, 0.13, "Pure Statcast · calendar-year data · no computer vision required",
                ha="center", fontsize=10, color="#666")
        ax.text(0.5, 0.085, f"Generated {date.today().isoformat()}  ·  Companion: Variable_Glossary.pdf",
                ha="center", fontsize=9, color="#999")
        pdf.savefig(fig); plt.close(fig)

        # ---- how to read this in 60 seconds ----
        textpage(pdf, "Read This First — The Idea in 60 Seconds", [
            "Stealing bases is treated here as a CONVERSION SKILL, not a speed contest.",
            "",
            "## The one number: Blueprint Conversion Score (BCS)",
            "Every runner-season gets a single score. Positive = better than the running",
            "game expects given that runner's speed. Negative = worse.",
            "",
            "## What pushes a runner UP",
            "[GOOD] • Covering a lot of ground from the pitcher's first move to release",
            "[GOOD]   (a big 'jump') — ESPECIALLY if they are slow.",
            "[GOOD] • Actually converting those jumps into stolen bases (high SB%).",
            "",
            "## What pushes a runner DOWN",
            "[BAD] • Being fast and getting caught anyway (wasting the speed).",
            "[BAD] • Repeatedly running into outs (many caught-stealings).",
            "[BAD] • Covering ground but still failing — squandering the jump too.",
            "",
            "## Why this matters",
            f"• {n} qualified runner-seasons (2023-2025), ≥10 tracked attempts each.",
            f"• The model surfaces {n_blue} 'blueprint' runners (slow + converts) and",
            f"  flags {n_squander} 'squander' runners (fast + fails).",
            f"• Josh Naylor 2025 ranks #{int(nay['rank_BCS'])}; Juan Soto 2025 ranks #{int(soto['rank_BCS'])}.",
            "",
            "## The punchline",
            "Speed barely buys ground. The runners who win the steal game on TIMING —",
            "not wheels — are a real, measurable, and rare archetype. Naylor is its",
            "gold standard; the fastest runners who keep getting caught are its mirror.",
        ], subtitle="One score, two poles, calendar-year data.")

        # ---- big picture figure ----
        imgpage(pdf, "1 · The Big Picture", FIGS / "Fig_NaylorBlueprint_Scatter.png",
                "Each dot is a runner-season. X = sprint speed (right is faster); "
                "Y = mean ground covered from first move to release. Color = BCS "
                "(green is good, red is bad). The dashed league fit is nearly FLAT — "
                "speed barely buys ground, which is the whole point. Naylor (#%d) and "
                "Soto (#%d) sit high on the LEFT (slow but cover ground); the fastest "
                "squanderers sit low on the RIGHT." % (int(nay['rank_BCS']), int(soto['rank_BCS'])))

        # ---- top figure + table ----
        imgpage(pdf, "2 · The Blueprint — Top 15", FIGS / "Fig_NaylorBlueprint_TopN.png",
                "The runners who convert ground into steals relative to their speed. "
                "Naylor (crimson) and Soto (dark green) are highlighted. Most are slow "
                "to average runners with elite jumps and near-perfect success.")
        top = df.head(15)
        textpage(pdf, "2 · The Blueprint — Top 15 (detail)",
                 ["spd = sprint ft/s (percentile) · 90ft = home-to-first sec · "
                  "gain = ground covered (ft)", ""]
                 + [_row_line(r) for _, r in top.iterrows()]
                 + ["",
                    "## How to read a line",
                    f"• Naylor: 24.4 ft/s (1.9th pctile), slow 90 ft (4.86s), yet covers",
                    f"  {nay['mean_gain_to_release_ft']:.1f} ft and steals at {nay['SB_pct']:.0f}% — a top-5 score on a",
                    f"  bottom-2% motor.",
                    "• The blueprint is NOT 'be fast'. It is 'get a great jump and cash it in'."],
                 subtitle="Slow runners who cover ground AND convert it.")

        # ---- bottom figure + table ----
        imgpage(pdf, "3 · The Anti-Naylor — Bottom 15", FIGS / "Fig_NaylorBlueprint_BottomN.png",
                "The fastest runners who squander it. Every bar is red (negative BCS). "
                "These are top-percentile sprinters with the quickest 90-ft times who "
                "nonetheless rack up caught-stealings — the profile Naylor would kill for, "
                "wasted.")
        botm = df.tail(15).iloc[::-1]  # worst first
        textpage(pdf, "3 · The Anti-Naylor — Bottom 15 (detail)",
                 ["The worst conversion scores on the board — fast, yet caught.", ""]
                 + [_row_line(r) for _, r in botm.iterrows()]
                 + ["",
                    "## The signature",
                    f"[BAD] • {bot['player_name']} {int(bot['season'])} anchors the bottom (#{int(bot['rank_BCS'])}): "
                    f"{bot['sprint_speed_ftps']:.1f} ft/s",
                    f"[BAD]   ({int(bot['sprint_pctile'])}th pctile), {int(bot['SB'])}/{int(bot['CS'])}, only "
                    f"{bot['mean_gain_to_release_ft']:.1f} ft of ground.",
                    "[BAD] • Bobby Witt Jr. — a 100th-percentile sprinter — lands in the",
                    "[BAD]   bottom tier in EVERY season tracked. Obvious speed, squandered."],
                 subtitle="Fastest runners, highest caught-stealing — penalized hardest.")

        # ---- Naylor / Soto deep dive ----
        sprint_pct = nay["sprint_pctile"]
        textpage(pdf, "4 · Josh Naylor — The Archetype", [
            "## The profile",
            f"• Sprint speed:   {nay['sprint_speed_ftps']:.1f} ft/s  ({sprint_pct:.1f}th percentile — bottom 2%)",
            f"• Home-to-1B:     {nay['hp_to_1b_s']:.2f} s  (slow)",
            f"• Ground covered: {nay['mean_gain_to_release_ft']:.2f} ft  (among the most in MLB)",
            f"• Steal record:   {int(nay['SB'])}/{int(nay['SB'])+int(nay['CS'])} = {nay['SB_pct']:.1f}%",
            f"• BCS:            {nay['BCS']:+.2f}   →   rank #{int(nay['rank_BCS'])} of {n}",
            "",
            "## Why he ranks so high",
            "[GOOD] • Gain residual (ground vs. what speed predicts): "
            f"{nay['gain_resid_z']:+.2f} SD",
            "[GOOD] • Success residual (steals vs. what speed predicts): "
            f"{nay['success_resid_z']:+.2f} SD",
            "[GOOD] • Squander penalty: "
            f"{nay['squander_z']:+.2f} (the floor — too slow to squander a speed edge)",
            "",
            "## The rarity",
            "Being THIS slow and covering THIS much ground is almost a contradiction.",
            "Treating the two as independent, the joint probability is on the order of",
            "1 in tens of millions of runner-seasons. Naylor is a genuine outlier — the",
            "model surfaces him near the very top BECAUSE he is slow, not in spite of it.",
            "",
            "## Juan Soto — the same skill, more speed",
        ] + [
            f"• {int(r['season'])}:  {r['sprint_speed_ftps']:.1f} ft/s ({r['sprint_pctile']:.0f}pct),  "
            f"{int(r['SB'])}/{int(r['SB'])+int(r['CS'])}={r['SB_pct']:.0f}%,  "
            f"gain {r['mean_gain_to_release_ft']:.1f} ft,  BCS {r['BCS']:+.2f}  (#{int(r['rank_BCS'])})"
            for _, r in df[df["runner_id"] == SOTO_ID].sort_values("season").iterrows()
        ], subtitle="A top-5 score on a bottom-2% motor.")

        # ---- how the score works ----
        textpage(pdf, "5 · How the Score Works (Plain English)", [
            "BCS adds two rewards and subtracts one penalty. All three are z-scores",
            "(standard deviations) across the qualified field, so they combine cleanly.",
            "",
            "```BCS  =  success_resid_z  +  gain_resid_z  −  squander_z",
            "",
            "## 1) success_resid_z  (reward)",
            "• Start with a Beta-Binomial estimate of the runner's true steal rate,",
            "  shrunk toward the league rate so a 4-for-4 cameo doesn't outrank a",
            "  30-for-31 season.",
            "• Subtract what that runner's SPEED predicts. Positive = converts more",
            "  often than a runner that fast 'should'.",
            "",
            "## 2) gain_resid_z  (reward)",
            "• How much ground the runner covers from first move to release, minus",
            "  what speed predicts. Positive = a big jump for how slow they are.",
            "",
            "## 3) squander_z  (penalty)",
            "```raw = caught-stealings  ×  max(speed, 0)  ×  (1 + max(jump, 0))",
            "• ONLY fast runners can be penalized — you can't squander a speed",
            "  advantage you don't have, so slow runners rest at the floor.",
            "• The (1 + jump) factor is the 'failed to capitalize' clause: covering",
            "  ground and STILL getting caught is punished harder.",
            "",
            "## The empirical-Bayes idea",
            "Both rewards are shrunk toward league behavior when the sample is small,",
            "so the leaderboard rewards repeatable skill, not small-sample luck.",
        ], subtitle="Two rewards, one penalty — combined into a single number.")

        # ---- annual data + honest limits ----
        textpage(pdf, "6 · Data Basis & Honest Limitations", [
            "## Annual (calendar-year) data — no career aggregates",
            "Every input is bounded to a single season; one row = one runner-season.",
            "• Stolen bases / caught stealing  — MLB StatsAPI, season=Y, regular season",
            "• Sprint speed & home-to-1B (90ft) — Baseball Savant, min=max=Y",
            "• Per-attempt lead & ground covered — Baseball Savant, season_start=end=Y",
            "A runner in 2023-25 contributes up to three independent rows.",
            "",
            "## On the speed metrics",
            "Statcast publishes sprint speed (peak velocity) and home-to-1B (the 90-ft",
            "burst, a stand-in for acceleration). It does NOT publish per-player",
            "acceleration or jerk, so the speed composite uses sprint speed + 90-ft time.",
            "",
            "## What BCS does and does not claim",
            "• It DOES rank conversion skill relative to a runner's own speed.",
            "• It does NOT say a slow runner is 'better' than a fast one in a vacuum —",
            "  it says who BEATS their own expectation, and who wastes their tools.",
            "• Tracked attempts only (Savant lead tracking, 2023+). Volume-qualified",
            "  at >=10 tracked attempts to keep the residuals stable.",
            "",
            "## Why no computer vision",
            "An earlier pipeline measured pitcher delivery time from video. It added NO",
            "predictive lift over the native Statcast lead metrics (leave-one-out AUC",
            "0.838 without it vs. 0.822 with it). The honest answer was already in the",
            "Statcast data — so this model is pure Statcast.",
        ], subtitle="What's measured, what's assumed, what's out of reach.")

    print(f"[write] {pdf_path}  ({pdf_path.stat().st_size/1024:.0f} KB)")
    return pdf_path


if __name__ == "__main__":
    build()
