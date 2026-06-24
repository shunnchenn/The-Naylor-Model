#!/usr/bin/env python3
"""
statcast_tables.py — V10 leaderboards rendered Statcast-style with great_tables.

All-time (top 15):   Output/Tables/Slow_Steal_Skill.png  etc.
Per-year (top 50):   Output/Tables/{2023,2024,2025,2026}/{name}.png

Color scheme — diverging, anchored to full league distribution:
  100th %ile = darkest green  |  50th = white  |  0th = darkest red
  Inverse metrics (jump_time: lower = better) are rank-flipped before coloring.

Run:  python3 Scripts/statcast_tables.py
"""
from pathlib import Path
import pandas as pd
from great_tables import GT, loc, style, md

ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "Data"
ASSETS    = ROOT / "Output" / "assets"
HEADSHOTS = ASSETS / "headshots"
LOGOS     = ASSETS / "logos"
OUT       = ROOT / "Output" / "Tables"
OUT.mkdir(parents=True, exist_ok=True)

NAVY   = "#1f2d3d"
GREENS = ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#005a32"]  # headline: low→high green

# Full league reference for percentile anchoring (all years)
_FULL = pd.read_csv(DATA / "Raw_Season.csv")
_BL   = pd.ExcelFile(DATA / "Naylor Blueprint.xlsx")
_BCS_FULL = _BL.parse("BCS Top 25 by Season")
_GC_FULL  = _BL.parse("Ground Covered")
_GC_Q     = _GC_FULL[_GC_FULL["volume_qualified"] == True].copy()

_PLACEHOLDER = ASSETS / "_blank.png"

# ── colour helpers ─────────────────────────────────────────────────────────────

_DIV_STOPS = [
    (0,   0x99, 0x00, 0x0d),
    (25,  0xfc, 0xae, 0x91),
    (50,  0xff, 0xff, 0xff),
    (75,  0x74, 0xc4, 0x76),
    (100, 0x00, 0x5a, 0x32),
]

def _diverging_hex(pct: float) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    for i in range(len(_DIV_STOPS) - 1):
        p0, r0, g0, b0 = _DIV_STOPS[i]
        p1, r1, g1, b1 = _DIV_STOPS[i + 1]
        if p0 <= pct <= p1:
            t = (pct - p0) / (p1 - p0)
            return "#{:02x}{:02x}{:02x}".format(
                int(r0 + t * (r1 - r0)),
                int(g0 + t * (g1 - g0)),
                int(b0 + t * (b1 - b0)),
            )
    return "#ffffff"

def _league_pctrank(values: pd.Series, ref: pd.Series,
                    higher_is_better: bool = True) -> pd.Series:
    """Percentile rank 0–100 vs full league ref. higher_is_better=False inverts rank."""
    ref_clean = ref.dropna()
    ranks = values.apply(
        lambda v: (ref_clean <= v).mean() * 100 if pd.notna(v) else float("nan")
    )
    return 100 - ranks if not higher_is_better else ranks

# ── image helpers ──────────────────────────────────────────────────────────────

def _ensure_placeholder():
    if not _PLACEHOLDER.exists():
        from PIL import Image
        Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(_PLACEHOLDER)
    return str(_PLACEHOLDER)

def _img(candidates):
    blank = _ensure_placeholder()
    return [str(p) if (p is not None and Path(p).is_file()) else blank for p in candidates]

def _headshot_col(ids):
    return _img([HEADSHOTS / f"{int(i)}.png" for i in ids])

def _logo_col(teams):
    return _img([LOGOS / f"{t}.png" if isinstance(t, str) and t else None for t in teams])

def _sbcs(df):
    return df["SB"].astype(int).astype(str) + " / " + df["CS"].astype(int).astype(str)

# ── core renderer ─────────────────────────────────────────────────────────────

def _render(df, *, headline, headline_label, stat_cols, title, subtitle, out_path,
            headline_fmt, source, diverging_cols=None):
    """
    stat_cols: [(col, label, fmt_code)]  fmt: 'n0','n1','n2','pct','raw'
    diverging_cols: [(display_col, pct_series_0_to_100)]
    """
    df = df.copy()
    df["headshot"] = _headshot_col(df["runner_id"])
    df["logo"]     = _logo_col(df.get("team", ""))
    df["who"]      = df["player_name"] + "  ·  " + df["season"].astype(int).astype(str)

    show = ["rank", "headshot", "who", "logo"] + [c for c, _, _ in stat_cols] + [headline]
    gt = (GT(df[show], rowname_col=None)
          .tab_header(title=md(f"**{title}**"), subtitle=subtitle)
          .fmt_image(columns="headshot", height=34)
          .fmt_image(columns="logo", height=26)
          .cols_label(rank="", headshot="", who="Player", logo="",
                      **{c: lbl for c, lbl, _ in stat_cols}, **{headline: headline_label})
          .data_color(columns=headline, palette=GREENS, na_color="#ffffff"))

    for code, dec in [("n0", 0), ("n1", 1), ("n2", 2)]:
        cols = [c for c, _, f in stat_cols if f == code]
        if cols:
            gt = gt.fmt_number(columns=cols, decimals=dec)
    pct_cols = [c for c, _, f in stat_cols if f == "pct"]
    if pct_cols:
        gt = gt.fmt_percent(columns=pct_cols, decimals=0)

    if diverging_cols:
        for display_col, pct_series in diverging_cols:
            if display_col not in show:
                continue
            for row_idx in range(len(df)):
                pct = pct_series.iloc[row_idx]
                if pd.isna(pct):
                    continue
                gt = gt.tab_style(
                    style=style.fill(color=_diverging_hex(pct)),
                    locations=loc.body(columns=display_col, rows=[row_idx]),
                )

    gt = (gt
          .tab_source_note(md(f"*{source}*"))
          .cols_align("center")
          .cols_align("left", columns="who")
          .tab_options(table_font_size="13px", heading_title_font_size="20px",
                       heading_subtitle_font_size="12px", column_labels_font_weight="bold",
                       column_labels_background_color=NAVY,
                       column_labels_font_size="11px", table_border_top_style="none",
                       data_row_padding="3px"))
    gt = gt.fmt(columns=headline, fns=headline_fmt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt.gtsave(str(out_path), zoom=2.5, expand=12)
    print(f"wrote {out_path}")
    return out_path

# ── per-table builders ────────────────────────────────────────────────────────

def slow_steal_skill(year=None, n=15):
    src = _FULL.copy()
    if year is not None:
        src = src[src["season"] == year]
    df = src.sort_values("SSSI_v7", ascending=False).head(n).reset_index(drop=True)
    df["rank"]       = df.index + 1
    df["sbcs"]       = _sbcs(df)
    df["ground"]     = df["avg_post_release_distance"]
    df["steals_exp"] = df["sb_residual"] * 100

    pct_jump   = _league_pctrank(df["jump_time"], _FULL["jump_time"],                higher_is_better=False)
    pct_ground = _league_pctrank(df["ground"],    _FULL["avg_post_release_distance"], higher_is_better=True)

    subtitle = (f"Skill left over after sprint speed is removed. {year}, top {len(df)}."
                if year else f"Skill left over after sprint speed is removed. 2015–2026, top {n}.")
    note = " (partial season)" if year == 2026 else ""
    out_path = (OUT / str(year) / "Slow_Steal_Skill.png") if year else (OUT / "Slow_Steal_Skill.png")

    _render(df,
        headline="SSSI_v7", headline_label="Slow-Steal Skill",
        headline_fmt=lambda v: f"{v:+.2f}",
        stat_cols=[("pct_speed", "Speed %ile", "n0"), ("jump_time", "Jump (s)", "n2"),
                   ("ground", "Ground Gained (ft)", "n1"),
                   ("steals_exp", "Steals Above Exp", "n1"), ("sbcs", "SB / CS", "raw")],
        diverging_cols=[("jump_time", pct_jump), ("ground", pct_ground)],
        title=f"Slow-Steal Skill{note} — {year if year else '2015–2026'}",
        subtitle=subtitle,
        source="Naylor Model V10  ·  Statcast running splits + SB/CS",
        out_path=out_path)

def blueprint_conversion(year=None, n=50):
    src = _BCS_FULL.copy()
    if year is not None:
        src = src[src["season"] == year]
    df = src.sort_values("rank_BCS").head(n).reset_index(drop=True)
    df["rank"]  = df.index + 1
    df["sbcs"]  = _sbcs(df)
    df["sbpct"] = df["SB_pct"] / 100 if df["SB_pct"].max() > 1.5 else df["SB_pct"]

    pct_speed   = df["sprint_pctile"]   # already 0–100
    pct_gain    = _league_pctrank(df["gain_resid_z"],    _BCS_FULL["gain_resid_z"],    higher_is_better=True)
    pct_success = _league_pctrank(df["success_resid_z"], _BCS_FULL["success_resid_z"], higher_is_better=True)

    note = " (partial season)" if year == 2026 else ""
    out_path = (OUT / str(year) / "Blueprint_Conversion.png") if year else (OUT / "Blueprint_Conversion.png")
    top_n = len(df)

    _render(df,
        headline="BCS", headline_label="Blueprint Score",
        headline_fmt=lambda v: f"{v:+.2f}",
        stat_cols=[("sprint_pctile", "Speed %ile", "n0"), ("sbcs", "SB / CS", "raw"),
                   ("sbpct", "SB%", "pct"), ("gain_resid_z", "Ground vs Exp", "n2"),
                   ("success_resid_z", "Convert vs Exp", "n2")],
        diverging_cols=[("sprint_pctile", pct_speed),
                        ("gain_resid_z",   pct_gain),
                        ("success_resid_z", pct_success)],
        title=f"Blueprint Conversion Score{note} — {year if year else '2023–2026'}",
        subtitle=(f"Speed-adjusted conversion + ground gained. {year}{note}, top {top_n}."
                  if year else f"Speed-adjusted conversion + ground gained. 2023–2026, top {n}."),
        source="Naylor Model V10  ·  Blueprint Conversion Score",
        out_path=out_path)

def ground_covered(year=None, n=50):
    src = _GC_Q.copy()
    if year is not None:
        src = src[src["season"] == year]
    df = src.sort_values("gain_residual_ft", ascending=False).head(n).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["sbcs"] = _sbcs(df)

    pct_speed = df["sprint_pctile"]   # already 0–100
    pct_gain  = _league_pctrank(df["mean_gain_to_release_ft"], _GC_Q["mean_gain_to_release_ft"],
                                higher_is_better=True)

    note = " (partial season)" if year == 2026 else ""
    out_path = (OUT / str(year) / "Ground_Covered.png") if year else (OUT / "Ground_Covered.png")

    _render(df,
        headline="gain_residual_ft", headline_label="Ground vs Speed (ft)",
        headline_fmt=lambda v: f"{v:+.2f}",
        stat_cols=[("sprint_pctile", "Speed %ile", "n0"),
                   ("mean_gain_to_release_ft", "Ground Gained (ft)", "n1"),
                   ("sbcs", "SB / CS", "raw")],
        diverging_cols=[("sprint_pctile", pct_speed),
                        ("mean_gain_to_release_ft", pct_gain)],
        title=f"Ground Covered Beyond Speed{note} — {year if year else '2023–2026'}",
        subtitle=(f"Feet gained from first move to release, speed-removed. {year}{note}, top {len(df)}."
                  if year else f"Feet gained from first move to release, speed-removed. Top {n}."),
        source="Naylor Model V10  ·  Statcast lead tracking",
        out_path=out_path)

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # All-time top-15 in Tables/ root
    slow_steal_skill()
    blueprint_conversion(n=15)
    ground_covered(n=15)

    # Per-year top-50 in Tables/{year}/
    for yr in [2023, 2024, 2025, 2026]:
        slow_steal_skill(year=yr, n=50)
        blueprint_conversion(year=yr, n=50)
        ground_covered(year=yr, n=50)
