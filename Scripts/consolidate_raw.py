#!/usr/bin/env python3
"""
consolidate_raw.py — collapse the scattered working CSVs into the two committed
raw-data files V10 reads from.

Produces:
  • Data/Raw_Season.csv    — one row per runner-season: the full SSSI feature set
                             + the xSB columns + a resolved team (from team_map.csv).
  • Data/Raw_Attempts.csv  — one row per tracked steal attempt, concatenated from
                             the leads_cache, with runner_id/season and binary y.

Run AFTER fetch_assets.py (for team_map.csv).  python3 Scripts/consolidate_raw.py
"""
from pathlib import Path
import re
import pandas as pd

ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "Data"
LEADS = ROOT / "Computer Vision" / "data" / "discovery" / "leads_cache"

RESULTS  = ROOT / "Output" / "Results"
SSSI     = RESULTS / "DF_v7_SSSI.csv"
XSB      = RESULTS / "DF_v7_xSB_Outcome.csv"
TEAM_MAP = DATA / "team_map.csv"
OUT_SEASON   = DATA / "Raw_Season.csv"
OUT_ATTEMPTS = DATA / "Raw_Attempts.csv"

XSB_ONLY = ["runner_id", "season", "net_sb", "z_net_sb", "z_sprint",
            "xsb_outcome", "sb_potential_gap", "quadrant"]

def build_season():
    s = pd.read_csv(SSSI)
    x = pd.read_csv(XSB)[XSB_ONLY]
    df = s.merge(x, on=["runner_id", "season"], how="left")
    if TEAM_MAP.exists():
        tm = pd.read_csv(TEAM_MAP)
        df = df.merge(tm, on=["runner_id", "season"], how="left")
    else:
        df["team"] = ""
        print("note: team_map.csv missing — run fetch_assets.py first for team logos")
    df.to_csv(OUT_SEASON, index=False)
    print(f"wrote {OUT_SEASON}  ({len(df)} rows × {df.shape[1]} cols)")

def build_attempts():
    pat = re.compile(r"(\d+)_(\d{4})\.csv$")
    frames = []
    for f in sorted(LEADS.glob("*.csv")):
        m = pat.search(f.name)
        if not m:
            continue
        try:
            a = pd.read_csv(f)
        except Exception:
            continue
        if a.empty:
            continue
        a.insert(0, "runner_id", int(m.group(1)))
        a.insert(1, "season", int(m.group(2)))
        frames.append(a)
    df = pd.concat(frames, ignore_index=True)
    df["y"] = (df["result"].astype(str).str.upper() == "SB").astype(int)
    df.to_csv(OUT_ATTEMPTS, index=False)
    print(f"wrote {OUT_ATTEMPTS}  ({len(df)} rows × {df.shape[1]} cols, "
          f"{df['y'].mean()*100:.1f}% SB)")

if __name__ == "__main__":
    build_season()
    build_attempts()
