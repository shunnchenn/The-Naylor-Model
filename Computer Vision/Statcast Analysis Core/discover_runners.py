#!/usr/bin/env python3
"""
discover_runners.py  —  Automated base-stealer discovery (replaces hover-and-click)
===================================================================================

The manual workflow this replaces: open the Statcast running-game leaderboard, set
the season bounds, hover over a runner's name in a row, and click to expand that
runner's stolen-base attempts.  This script enumerates the whole leaderboard for a
season (or range) and ranks it, so the next step (fetch_leads.py, the "click") can
be pointed at any runner_id with no manual hovering.

Seed = MERGED (per the user):
  * SB / CS volume + identities  ← StatsAPI season hitting stats (year-correct).
        The Savant basestealing-run-value leaderboard CSV ignores its year param,
        so StatsAPI is the reliable year-correct source for per-runner SB/CS.
  * sprint_speed + percentile    ← Savant sprint_speed leaderboard (year-correct).
Joined on player_id, so you can sort/filter for the slow-but-prolific archetype
(low sprint speed, high attempts — Naylor / Soto), the runners this CV project
exists to study.

Usage
-----
  python3 cv_pilot/discover_runners.py --start 2025 --end 2025
  python3 cv_pilot/discover_runners.py --start 2025 --end 2025 \
      --min-attempts 8 --max-sprint-pctile 40 --top 25
  # then expand the top runners into per-attempt leads (the automated "click"):
  python3 cv_pilot/discover_runners.py --start 2025 --end 2025 --top 10 --expand

Network: StatsAPI + Savant have no sandbox DNS -> run with dangerouslyDisableSandbox.
Writes: cv_pilot/discovery/runners_<start>_<end>.csv
"""
from __future__ import annotations
import argparse
import csv
import io
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests not installed.  pip install requests")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": UA})

HERE = Path(__file__).resolve().parent

# Year-correct sources (see module docstring for why these two and not the
# basestealing-run-value leaderboard CSV, whose year param is broken).
STATS_URL = ("https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting"
             "&season={y}&gameType=R&playerPool=All&sortStat=stolenBases"
             "&order=desc&limit=3000")
SPRINT_URL = ("https://baseballsavant.mlb.com/leaderboard/sprint_speed"
              "?attempts=1&min_season={s}&max_season={e}&position=&team=&csv=true")

OUT_COLS = ["runner_id", "name", "name_tag", "team", "position",
            "sb", "cs", "attempts", "success_pct",
            "sprint_speed_ftps", "sprint_pctile", "hp_to_1b_s", "seasons"]


def _get_json(url, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, timeout=40)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == tries - 1:
                print(f"  ! GET failed {url}: {e}")
                return None
            time.sleep(1.0 + i)
    return None


def _get_csv(url, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, timeout=40)
            r.raise_for_status()
            return list(csv.DictReader(io.StringIO(r.text.lstrip("﻿"))))
        except Exception as e:
            if i == tries - 1:
                print(f"  ! GET failed {url}: {e}")
                return []
            time.sleep(1.0 + i)
    return []


def _name_tag(full_name: str) -> str:
    """Lowercase last-name tag used downstream for clip naming (fetch_leads)."""
    parts = str(full_name).split()
    last = parts[-1] if parts else "runner"
    return "".join(ch for ch in last.lower() if ch.isalnum()) or "runner"


def fetch_sb_cs(start: int, end: int) -> dict[int, dict]:
    """Per-runner SB/CS summed across the season range (year-correct, StatsAPI)."""
    agg: dict[int, dict] = {}
    for y in range(start, end + 1):
        d = _get_json(STATS_URL.format(y=y))
        if not d or not d.get("stats"):
            continue
        for s in d["stats"][0]["splits"]:
            stat, p = s["stat"], s["player"]
            sb = int(stat.get("stolenBases", 0) or 0)
            cs = int(stat.get("caughtStealing", 0) or 0)
            if sb + cs == 0:
                continue
            pid = int(p["id"])
            a = agg.setdefault(pid, {
                "runner_id": pid, "name": p["fullName"], "sb": 0, "cs": 0,
                "team": (s.get("team") or {}).get("abbreviation", ""),
                "position": (s.get("position") or {}).get("abbreviation", ""),
                "seasons": set(),
            })
            a["sb"] += sb
            a["cs"] += cs
            a["seasons"].add(y)
    return agg


def fetch_sprint(start: int, end: int) -> dict[int, dict]:
    rows = _get_csv(SPRINT_URL.format(s=start, e=end))
    out: dict[int, dict] = {}
    speeds = []
    for r in rows:
        try:
            pid = int(r["player_id"])
            spd = float(r["sprint_speed"])
        except (KeyError, TypeError, ValueError):
            continue
        out[pid] = {"sprint_speed": spd, "hp_to_1b": r.get("hp_to_1b", ""),
                    "team": r.get("team", ""), "position": r.get("position", "")}
        speeds.append((pid, spd))
    # percentile within the full sprint population (low pct = slow)
    speeds.sort(key=lambda x: x[1])
    n = len(speeds)
    for rank, (pid, _) in enumerate(speeds):
        out[pid]["pctile"] = round(100.0 * rank / (n - 1), 1) if n > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description="Discover & rank base-stealers for a season/range")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--min-attempts", type=int, default=1,
                    help="keep runners with SB+CS >= this (default 1)")
    ap.add_argument("--max-sprint-pctile", type=float, default=None,
                    help="keep only runners at/below this sprint percentile (slow filter)")
    ap.add_argument("--top", type=int, default=None, help="keep only the top-N rows after sort")
    ap.add_argument("--sort", choices=["attempts", "slow", "run_value_proxy"], default="attempts",
                    help="attempts (default) | slow = prolific-AND-slow | run_value_proxy")
    ap.add_argument("--out", default=None, help="override output CSV path")
    ap.add_argument("--expand", action="store_true",
                    help="after ranking, run fetch_leads.py for each kept runner (the 'click')")
    args = ap.parse_args()

    print(f"[discover] seasons {args.start}-{args.end}: pulling StatsAPI SB/CS + sprint_speed …")
    sbcs = fetch_sb_cs(args.start, args.end)
    sprint = fetch_sprint(args.start, args.end)
    print(f"[discover] {len(sbcs)} runners with >=1 attempt; {len(sprint)} sprint rows")

    rows = []
    for pid, a in sbcs.items():
        att = a["sb"] + a["cs"]
        sp = sprint.get(pid, {})
        rows.append({
            "runner_id": pid,
            "name": a["name"],
            "name_tag": _name_tag(a["name"]),
            "team": a["team"] or sp.get("team", ""),
            "position": a["position"] or sp.get("position", ""),
            "sb": a["sb"],
            "cs": a["cs"],
            "attempts": att,
            "success_pct": round(100.0 * a["sb"] / att, 1) if att else "",
            "sprint_speed_ftps": sp.get("sprint_speed", ""),
            "sprint_pctile": sp.get("pctile", ""),
            "hp_to_1b_s": sp.get("hp_to_1b", ""),
            "seasons": "/".join(str(y) for y in sorted(a["seasons"])),
        })

    # filters
    rows = [r for r in rows if r["attempts"] >= args.min_attempts]
    if args.max_sprint_pctile is not None:
        rows = [r for r in rows
                if r["sprint_pctile"] != "" and r["sprint_pctile"] <= args.max_sprint_pctile]

    # sort
    def slow_key(r):
        # prolific AND slow: many attempts, low sprint percentile.
        pct = r["sprint_pctile"] if r["sprint_pctile"] != "" else 100.0
        return (r["attempts"], -pct)
    if args.sort == "attempts":
        rows.sort(key=lambda r: (r["attempts"], r["sb"]), reverse=True)
    elif args.sort == "slow":
        rows.sort(key=slow_key, reverse=True)
    else:  # run_value_proxy ~ attempts * success (no per-attempt run value here)
        rows.sort(key=lambda r: r["attempts"] * (r["success_pct"] or 0), reverse=True)

    if args.top:
        rows = rows[:args.top]

    out = Path(args.out) if args.out else (HERE / "discovery" /
                                           f"runners_{args.start}_{args.end}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {out}  ({len(rows)} runners)")

    # preview
    print(f"\n{'runner_id':>9}  {'name':22s} {'att':>3} {'SB':>3} {'CS':>2} "
          f"{'succ%':>5} {'sprint':>6} {'pct':>5}")
    for r in rows[:20]:
        print(f"{r['runner_id']:>9}  {r['name'][:22]:22s} {r['attempts']:>3} "
              f"{r['sb']:>3} {r['cs']:>2} {str(r['success_pct']):>5} "
              f"{str(r['sprint_speed_ftps']):>6} {str(r['sprint_pctile']):>5}")

    if args.expand:
        print(f"\n[expand] fetching per-attempt leads for {len(rows)} runners (the 'click') …")
        for r in rows:
            tag = r["name_tag"]
            d = HERE / "discovery" / f"{tag}_{args.start}_{args.end}"
            for y in range(args.start, args.end + 1):
                cmd = [sys.executable, str(HERE / "fetch_leads.py"),
                       str(r["runner_id"]), str(y),
                       "--runner-name", tag,
                       "--out", str(d / f"{tag}{y}_leads.csv"),
                       "--targets-out", str(d / f"{tag}{y}_targets.csv")]
                print("   $", " ".join(cmd[1:]))
                subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
