#!/usr/bin/env python3
"""
build_features.py — offline feature engineering: turn the scraped cache into the two
committed raw-data files the model reads.

Produces:
  • Data/Raw_Season.csv    — one row per runner-season: the SSSI feature set + the xSB
                             columns + a resolved team (from team_map.csv) + Statcast
                             SB run value summed from the per-attempt cache.
  • Data/Raw_Attempts.csv  — one row per tracked steal attempt, concatenated from
                             Data/leads_cache, tagged with runner_id/season and binary y.

Inputs (all already on disk — no network): Data/leads_cache/*.csv (from
scrape_statcast.py), Output/Results/DF_v7_SSSI.csv + DF_v7_xSB_Outcome.csv (the season
feature set), and Data/team_map.csv (from `scrape_statcast.py assets`).

Run:  python3 ingest/build_features.py
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "Data"
LEADS = DATA / "leads_cache"

RESULTS  = ROOT / "Output" / "Results"
SSSI     = RESULTS / "DF_v7_SSSI.csv"
XSB      = RESULTS / "DF_v7_xSB_Outcome.csv"
TEAM_MAP = DATA / "team_map.csv"
OUT_SEASON   = DATA / "Raw_Season.csv"
OUT_ATTEMPTS = DATA / "Raw_Attempts.csv"

XSB_ONLY = ["runner_id", "season", "net_sb", "z_net_sb", "z_sprint",
            "xsb_outcome", "sb_potential_gap", "quadrant"]

# Legacy season-level Steal+ (kept in Raw_Season.csv for reference only). The FLAGSHIP,
# volume-aware Steal+ is computed fresh in model/metrics.py from SB/CS + sprint speed and
# does NOT depend on these columns. sb_run_value is Statcast's own steals-above-average.
REF_ATTEMPTS = 30          # legacy scale: a full base-stealing season
MIN_ATTEMPTS = 10          # qualified runner-season (for the legacy league baseline)

LEADS_FILE = re.compile(r"(\d+)_(\d{4})\.csv$")     # leads_cache filename -> (runner_id, season)


def _iter_leads():
    """Yield (runner_id, season, dataframe) for every readable, non-empty leads file."""
    for f in sorted(LEADS.glob("*.csv")):
        m = LEADS_FILE.search(f.name)
        if not m:
            continue
        try:
            a = pd.read_csv(f)
        except Exception:
            continue
        if a.empty:
            continue
        yield int(m.group(1)), int(m.group(2)), a


def sb_run_value_map():
    """Statcast SB run value per runner-season = Σ per-attempt run_value from the cache.
    The official Statcast 'steals above average', surfaced (not reinvented) for reference."""
    rows = []
    for runner_id, season, a in _iter_leads():
        if "run_value" not in a.columns:
            continue
        rv = pd.to_numeric(a["run_value"], errors="coerce").sum()
        rows.append({"runner_id": runner_id, "season": season, "sb_run_value": round(float(rv), 3)})
    return pd.DataFrame(rows)


def add_legacy_metrics(df):
    """Attach the legacy season-level Steal+ (sb_residual x 30) and the average-runner
    net-bases baseline. Reference columns only — model/metrics.py recomputes the flagship."""
    q = df[df["sb_attempts"] >= MIN_ATTEMPTS]
    league_net_rate = float((q["SB"] - q["CS"]).sum() / max(1, q["sb_attempts"].sum()))
    df["league_net_rate"] = round(league_net_rate, 4)
    df["steal_plus"] = (df["sb_residual"] * REF_ATTEMPTS).round(2)
    df["exp_net"]    = (league_net_rate * df["sb_attempts"]).round(2)
    return df


def refresh_live_sbcs(df, years, min_att=1):
    """Overlay year-correct StatsAPI SB/CS + sprint onto the given seasons, and append any newly
    qualifying runner-seasons. Used to bring an in-progress season (e.g. 2026) current WITHOUT
    mutating the frozen DF_v7_SSSI on disk — final seasons are left exactly as published. New rows
    carry NaN for the SSSI running-mechanics columns (jump_time etc.); Steal+/Burst don't use them
    and the per-attempt model tolerates the NaNs."""
    import scrape_statcast as S
    sp = pd.read_csv(DATA / "sprint_speed.csv").rename(columns={"sprint_speed_all": "sprint_speed"})
    keep_other = df[~df["season"].isin(years)]
    updated = [keep_other]
    for y in years:
        live = S.fetch_sb_cs(y, y)                       # {pid: {sb, cs, name, team, ...}}
        cur = df[df["season"] == y].set_index("runner_id")
        rows = []
        for pid, a in live.items():
            att = a["sb"] + a["cs"]
            if att < min_att:
                continue
            base = cur.loc[pid].to_dict() if pid in cur.index else {c: np.nan for c in df.columns}
            base.update({"runner_id": pid, "season": y, "SB": a["sb"], "CS": a["cs"],
                         "sb_attempts": att, "player_name": a["name"] or base.get("player_name")})
            spd = sp[(sp.runner_id == pid) & (sp.season == y)]["sprint_speed"]
            if len(spd):
                base["sprint_speed"] = float(spd.iloc[0])
            rows.append(base)
        got = pd.DataFrame(rows)
        print(f"  refreshed {y}: {len(got)} runner-seasons (was {len(cur)}), "
              f"total SB {int(got['SB'].sum()) if len(got) else 0}")
        updated.append(got)
    return pd.concat(updated, ignore_index=True)


def build_season(refresh_years=None, refresh_min_att=10):
    """Assemble Data/Raw_Season.csv (one row per runner-season). If refresh_years is given,
    overlay live StatsAPI SB/CS for those seasons (network) so an in-progress season is current."""
    df = pd.read_csv(SSSI).merge(pd.read_csv(XSB)[XSB_ONLY], on=["runner_id", "season"], how="left")
    if refresh_years:
        df = refresh_live_sbcs(df, refresh_years, refresh_min_att)
    if TEAM_MAP.exists():
        df = df.merge(pd.read_csv(TEAM_MAP), on=["runner_id", "season"], how="left")
    else:
        df["team"] = ""
        print("note: team_map.csv missing — run `scrape_statcast.py assets` first for team logos")
    df = add_legacy_metrics(df)
    df = df.merge(sb_run_value_map(), on=["runner_id", "season"], how="left")
    df.to_csv(OUT_SEASON, index=False)
    rvn = int(df["sb_run_value"].notna().sum())
    print(f"[write] {OUT_SEASON.name}  ({len(df)} rows x {df.shape[1]} cols; "
          f"Statcast run value on {rvn} rows)")


def build_attempts():
    """Assemble Data/Raw_Attempts.csv (one row per tracked steal attempt) for the MODERN era.
    The cache now also holds classic 2015-2022 leads (for model_classic.py), so filter to >=2023
    here — otherwise the modern per-attempt model would train across the 2023 rule change."""
    frames = []
    for runner_id, season, a in _iter_leads():
        if season < 2023:
            continue
        if "runner_id" not in a.columns:            # older cache files carry no id/season columns
            a.insert(0, "runner_id", runner_id)
            a.insert(1, "season", season)
        frames.append(a)
    df = pd.concat(frames, ignore_index=True)
    df["y"] = (df["result"].astype(str).str.upper() == "SB").astype(int)
    df.to_csv(OUT_ATTEMPTS, index=False)
    print(f"[write] {OUT_ATTEMPTS.name}  ({len(df)} rows x {df.shape[1]} cols, {df['y'].mean()*100:.1f}% SB)")


if __name__ == "__main__":
    import sys
    refresh = None
    if "--refresh" in sys.argv:
        i = sys.argv.index("--refresh")
        refresh = [int(y) for y in sys.argv[i + 1].split(",")]   # e.g. --refresh 2026
    build_season(refresh_years=refresh)
    build_attempts()
