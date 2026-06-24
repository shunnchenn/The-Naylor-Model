#!/usr/bin/env python3
"""
fetch_assets.py — one-time network fetch of the visual assets the V10 Statcast
tables need, with everything cached to disk so later builds run offline.

Produces:
  • Output/assets/headshots/{runner_id}.png   — MLB headshot per runner
  • Data/team_map.csv                          — runner_id, season, team (logo-normalized)

Team comes from the MLB Stats API (the runner-season grain has no team column);
abbreviations are normalized to the logo filenames in Output/assets/logos/.

Run:  python3 Scripts/fetch_assets.py
"""
from pathlib import Path
import time
import pandas as pd
import requests

ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "Data"
ASSETS    = ROOT / "Output" / "assets"
HEADSHOTS = ASSETS / "headshots"
LOGOS     = ASSETS / "logos"
HEADSHOTS.mkdir(parents=True, exist_ok=True)

SEASON_MASTER = ROOT / "Output" / "Results" / "DF_v7_SSSI.csv"
TEAM_MAP_OUT  = DATA / "team_map.csv"

HEADSHOT_URL = ("https://img.mlbstatic.com/mlb-photos/image/upload/"
                "d_people:generic:headshot:67:current.png/w_213,q_auto:best/"
                "v1/people/{pid}/headshot/67/current")
STATS_URL = ("https://statsapi.mlb.com/api/v1/people/{pid}/stats"
             "?stats=season&season={yr}&group=hitting")

# MLB Stats API abbreviation → our logo filename (Output/assets/logos/*.png)
ABBR_FIX = {"ARI": "AZ", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF",
            "TBR": "TB", "WSN": "WSH", "OAK": "ATH", "AthLetics": "ATH"}

def _logo_names():
    return {p.stem for p in LOGOS.glob("*.png")}

def team_id_map(session):
    """team id → abbreviation, for every franchise the API knows (all seasons)."""
    m = {}
    r = session.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=20)
    for t in r.json().get("teams", []):
        if t.get("id") and t.get("abbreviation"):
            m[t["id"]] = t["abbreviation"]
    return m

def fetch_headshot(pid, session):
    out = HEADSHOTS / f"{pid}.png"
    if out.exists():
        return True
    try:
        r = session.get(HEADSHOT_URL.format(pid=pid), timeout=20)
        if r.ok and r.content and len(r.content) > 1000:
            out.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False

def fetch_team(pid, yr, session, id2abbr):
    """Abbreviation of the player's most recent team stint that season."""
    try:
        r = session.get(STATS_URL.format(pid=pid, yr=yr), timeout=20)
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        tid = splits[-1].get("team", {}).get("id")  # last split = latest stint
        return id2abbr.get(tid)
    except Exception:
        return None

def main():
    df = pd.read_csv(SEASON_MASTER, usecols=["runner_id", "season"]).drop_duplicates()
    logos = _logo_names()
    session = requests.Session()
    session.headers.update({"User-Agent": "naylor-model/1.0"})
    id2abbr = team_id_map(session)

    # headshots: cover the season master AND every runner on the Blueprint leaderboards
    head_ids = set(df["runner_id"].astype(int))
    bp = DATA / "Naylor Blueprint.xlsx"
    if bp.exists():
        xl = pd.ExcelFile(bp)
        for sh in ("BCS Top 25 by Season", "Ground Covered"):
            if sh in xl.sheet_names:
                head_ids |= set(xl.parse(sh)["runner_id"].dropna().astype(int))
    print(f"fetching headshots for {len(head_ids)} unique players …")
    for pid in head_ids:
        fetch_headshot(pid, session)

    rows, missing_logo = [], set()
    print(f"resolving teams for {len(df)} runner-seasons …")
    for i, (_, r) in enumerate(df.iterrows()):
        pid, yr = int(r["runner_id"]), int(r["season"])
        abbr = fetch_team(pid, yr, session, id2abbr)
        logo = ABBR_FIX.get(abbr, abbr) if abbr else None
        if logo and logo not in logos:
            missing_logo.add(logo)
        rows.append({"runner_id": pid, "season": yr, "team": logo or ""})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)} …")
        time.sleep(0.05)

    tm = pd.DataFrame(rows).drop_duplicates(["runner_id", "season"])
    tm.to_csv(TEAM_MAP_OUT, index=False)
    print(f"wrote {TEAM_MAP_OUT}  ({len(tm)} rows, {tm['team'].ne('').sum()} with team)")
    print(f"headshots cached: {len(list(HEADSHOTS.glob('*.png')))}")
    if missing_logo:
        print(f"WARNING — team abbrevs with no logo file: {sorted(missing_logo)}")

if __name__ == "__main__":
    main()
