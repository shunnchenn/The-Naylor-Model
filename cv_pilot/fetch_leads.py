#!/usr/bin/env python3
"""
fetch_leads.py  —  Per-attempt Statcast base-stealing leads for any runner/year
================================================================================

Pulls each tracked steal attempt's lead distances straight from Baseball Savant's
basestealing-running-game service (the data behind the "All Stolen Base Attempts"
drawer on https://baseballsavant.mlb.com/leaderboard/basestealing-run-value):

    GET /leaderboard/services/basestealing-running-game/{runner_id}
        ?season_start={Y}&season_end={Y}
    -> {"data": [ {per-attempt row}, ... ]}

Field mapping (verified against the public leaderboard drawer):
    r_primary_lead         -> lead_at_firstmove_ft   (lead at pitcher's first move)
    r_sec_minus_prim_lead  -> gain_to_release_ft     (lead distance gained)
    r_secondary_lead       -> lead_at_release_ft     (lead at pitch release)
    runs_stolen_on_running_act -> run_value
    runner_moved_cd        -> result (SB/CS)
    target_base            -> base (2B/3B)
    play_id                -> Film-Room GUID (feeds fetch_clips.py directly)

Also (optional) resolves each attempt's game_pk from the runner's StatsAPI gameLog
(date -> gamePk) and writes a targets CSV for fetch_clips.py --targets.

Usage
-----
python3 cv_pilot/fetch_leads.py 647304 2025 \
    --runner-name naylor \
    --out cv_pilot/Naylor_2025/naylor2025_leads.csv \
    --targets-out cv_pilot/Naylor_2025/naylor2025_targets.csv

Host is baseballsavant.mlb.com (NO hyphen). The shell sandbox has no DNS for it,
so run with dangerouslyDisableSandbox.
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests not installed.  pip install requests")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

LEADS_URL = ("https://baseballsavant.mlb.com/leaderboard/services/"
             "basestealing-running-game/{rid}?season_start={y}&season_end={y}")
GAMELOG_URL = ("https://statsapi.mlb.com/api/v1/people/{rid}/stats"
               "?stats=gameLog&group=hitting&season={y}&gameType=R")

LEADS_COLS = ["date", "play_id", "pitcher_id", "pitcher_name",
              "catcher_id", "catcher_name", "fielder_name",
              "base", "result", "run_value",
              "lead_at_firstmove_ft", "gain_to_release_ft", "lead_at_release_ft"]


def _get_json(url, tries=4):
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == tries - 1:
                print(f"  ! GET failed {url}: {e}")
                return None
            time.sleep(1.0 + i)
    return None


def _f(x, nd=None):
    try:
        v = float(x)
        return round(v, nd) if nd is not None else v
    except (TypeError, ValueError):
        return None


def fetch_leads(runner_id: int, year: int) -> list[dict]:
    d = _get_json(LEADS_URL.format(rid=runner_id, y=year))
    data = d.get("data", []) if isinstance(d, dict) else (d or [])
    rows = []
    for a in data:
        rows.append({
            "date": str(a.get("game_date", ""))[:10],
            "play_id": a.get("play_id"),
            "pitcher_id": a.get("pitcher_id"),
            "pitcher_name": a.get("pitcher_name"),
            "catcher_id": a.get("catcher_id"),
            "catcher_name": a.get("catcher_name"),
            "fielder_name": a.get("fielder_name"),
            "base": a.get("target_base"),
            "result": a.get("runner_moved_cd"),
            "run_value": _f(a.get("runs_stolen_on_running_act"), 3),
            "lead_at_firstmove_ft": _f(a.get("r_primary_lead"), 1),
            "gain_to_release_ft": _f(a.get("r_sec_minus_prim_lead"), 1),
            "lead_at_release_ft": _f(a.get("r_secondary_lead"), 1),
        })
    rows.sort(key=lambda r: (r["date"] or "", str(r["pitcher_name"])))
    return rows


def date_to_gamepks(runner_id: int, year: int) -> dict[str, list[int]]:
    d = _get_json(GAMELOG_URL.format(rid=runner_id, y=year))
    out: dict[str, list[int]] = {}
    if not d or not d.get("stats"):
        return out
    for s in d["stats"][0]["splits"]:
        dt = s.get("date")
        pk = s.get("game", {}).get("gamePk")
        if dt and pk:
            out.setdefault(dt, []).append(int(pk))
    return out


def statsapi_sb_cs(runner_id: int, year: int) -> tuple[int, int]:
    d = _get_json(GAMELOG_URL.format(rid=runner_id, y=year))
    sb = cs = 0
    if d and d.get("stats"):
        for s in d["stats"][0]["splits"]:
            st = s["stat"]
            sb += int(st.get("stolenBases", 0))
            cs += int(st.get("caughtStealing", 0))
    return sb, cs


def main():
    ap = argparse.ArgumentParser(description="Fetch per-attempt steal leads for a runner/year")
    ap.add_argument("runner_id", type=int)
    ap.add_argument("year", type=int)
    ap.add_argument("--runner-name", default="runner", help="lowercase tag for clip naming")
    ap.add_argument("--out", required=True, help="leads CSV path")
    ap.add_argument("--targets-out", help="also write a fetch_clips targets CSV (resolves game_pk)")
    args = ap.parse_args()

    rows = fetch_leads(args.runner_id, args.year)
    if not rows:
        sys.exit("No attempts returned by the leads endpoint.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEADS_COLS)
        w.writeheader()
        w.writerows(rows)
    n_sb = sum(1 for r in rows if r["result"] == "SB")
    n_cs = sum(1 for r in rows if r["result"] == "CS")
    print(f"[write] {out}  ({len(rows)} attempts: {n_sb} SB / {n_cs} CS)")

    # sanity-check coverage vs StatsAPI season totals
    sb, cs = statsapi_sb_cs(args.runner_id, args.year)
    print(f"[check] StatsAPI {args.year} totals: {sb} SB / {cs} CS  "
          f"(tracked here: {n_sb}/{n_cs}; untracked = steals of home / non-2B-3B / unmeasured)")

    if args.targets_out:
        d2g = date_to_gamepks(args.runner_id, args.year)
        trows = []
        unresolved = 0
        for r in rows:
            pks = d2g.get(r["date"], [])
            if not pks:
                unresolved += 1
            for pk in (pks or [None]):
                trows.append({
                    "game_pk": pk if pk is not None else "",
                    "play_id": r["play_id"],
                    "is_naylor": 1,
                    "clip_prefix": args.runner_name,
                    "date": r["date"],
                    "pitcher_name": r["pitcher_name"],
                    "result": r["result"],
                })
        tout = Path(args.targets_out)
        with open(tout, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["game_pk", "play_id", "is_naylor",
                                              "clip_prefix", "date", "pitcher_name", "result"])
            w.writeheader()
            w.writerows(trows)
        print(f"[write] {tout}  ({len(trows)} target rows; {unresolved} dates without a gamePk)")


if __name__ == "__main__":
    main()
