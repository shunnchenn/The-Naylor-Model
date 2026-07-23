#!/usr/bin/env python3
"""
scrape_statcast.py — the project's ONLY web-scraper (Baseball Savant + MLB StatsAPI).

Two subcommands, run in this order to (re)build the raw inputs:

  discover   Rank the league's base-stealers for a season range, joining year-correct
             SB/CS (StatsAPI) with sprint speed + percentile (Savant). Writes one
             runners_<start>_<end>.csv. With --expand it then pulls per-attempt leads
             for every runner it kept.
                 python3 Scripts/scrape_statcast.py discover --start 2023 --end 2026 --expand

  leads      Pull one runner-season's per-attempt steal data (the lead distances that
             become Burst) from Savant's basestealing-running-game service, and print a
             coverage check against that runner's official StatsAPI SB/CS total.
                 python3 Scripts/scrape_statcast.py leads 647304 2025

Why these two sources and not the obvious one: Savant's basestealing-run-value LEADERBOARD
export silently ignores its year parameter (it returns the latest season regardless), so
year-correct SB/CS come from StatsAPI instead. Savant is used only where it IS year-correct:
the sprint-speed leaderboard and the PER-ATTEMPT drawer (the running-game service below).

Field mapping for the per-attempt data (verified against the public leaderboard drawer):
    r_primary_lead         -> lead_at_firstmove_ft   lead (ft) at the pitcher's first move
    r_secondary_lead       -> lead_at_release_ft     secondary lead (ft): pitch reaching the catcher
    r_sec_minus_prim_lead  -> gain_to_release_ft     ground gained between those two -> feeds Burst
    runner_moved_cd        -> result (SB / CS)
    runs_stolen_on_running_act -> run_value  (Statcast's own steals-above-average credit)

Network: Savant/StatsAPI have no DNS inside the sandbox -> run with dangerouslyDisableSandbox.
Writes: Data/leads_cache/<id>_<year>.csv  and  Data/discovery/runners_<start>_<end>.csv
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests not installed.  pip install requests")

ROOT       = Path(__file__).resolve().parent.parent
LEADS_DIR  = ROOT / "Data" / "leads_cache"
DISC_DIR   = ROOT / "Data" / "discovery"
HEADSHOTS  = ROOT / "Output" / "assets" / "headshots"
LOGOS      = ROOT / "Output" / "assets" / "logos"
SEASON_MASTER = ROOT / "Output" / "Results" / "DF_v7_SSSI.csv"
TEAM_MAP_OUT  = ROOT / "Data" / "team_map.csv"
PBP_DIR       = ROOT / "Data" / "pbp_cache"          # one small json per game (resumable)
OPPS_DIR      = ROOT / "Data" / "pbp_opps_cache"     # v13: separate cache — richer schema
OPPS_OUT      = ROOT / "Data" / "Raw_Opportunities.csv"
POPTIME_OUT   = ROOT / "Data" / "poptime.csv"
SPRINT_OUT    = ROOT / "Data" / "sprint_speed.csv"
CONTEXT_OUT   = ROOT / "Data" / "Raw_Attempt_Context.csv"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

# Per-attempt lead drawer (year-correct). Season SB/CS + sprint speed leaderboards below it.
LEADS_URL   = ("https://baseballsavant.mlb.com/leaderboard/services/"
               "basestealing-running-game/{rid}?season_start={y}&season_end={y}")
GAMELOG_URL = ("https://statsapi.mlb.com/api/v1/people/{rid}/stats"
               "?stats=gameLog&group=hitting&season={y}&gameType=R")
STATS_URL   = ("https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting"
               "&season={y}&gameType=R&playerPool=All&sortStat=stolenBases&order=desc&limit=3000")
# v12 context: pitch/count/situation joined on play_id, and catcher pop time + arm strength.
# `fields` shrinks playByPlay ~9x (490KB -> 55KB) while keeping everything we use.
PBP_URL     = ("https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay?fields="
               "allPlays,playEvents,playId,details,type,code,count,balls,strikes,outs,"
               "about,inning,halfInning,matchup,pitchHand,batSide,result,awayScore,homeScore")
SCHED_URL   = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
# v13 opportunity feed. `fields` is a FLAT key-name whitelist applied at every depth, so each
# descendant key must be named individually. Carries runner movement so base state can be
# replayed, and pitch ordering so the TRUE pre-pitch count is recoverable.
PBP_OPPS_URL = ("https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay?fields="
                "allPlays,atBatIndex,about,inning,halfInning,playEvents,playId,isPitch,index,"
                "pitchNumber,details,type,code,count,balls,strikes,outs,matchup,pitchHand,"
                "batSide,postOnFirst,postOnSecond,postOnThird,id,runners,movement,originBase,"
                "start,end,outBase,isOut,event,eventType,playIndex,runner,result,awayScore,homeScore")
SCHED_R_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R&date={date}"
POPTIME_URL = ("https://baseballsavant.mlb.com/leaderboard/poptime"
               "?year={y}&team=&min2sb=1&min3b=0&csv=true")
SPRINT_URL  = ("https://baseballsavant.mlb.com/leaderboard/sprint_speed"
               "?attempts=1&min_season={s}&max_season={e}&position=&team=&csv=true")

LEADS_COLS = ["runner_id", "season", "date", "play_id", "pitcher_id", "pitcher_name",
              "catcher_id", "catcher_name", "fielder_name", "base", "result", "run_value",
              "lead_at_firstmove_ft", "gain_to_release_ft", "lead_at_release_ft"]
DISC_COLS  = ["runner_id", "name", "name_tag", "team", "position", "sb", "cs", "attempts",
              "success_pct", "sprint_speed_ftps", "sprint_pctile", "seasons"]


# ── shared fetch helpers (retry with linear backoff) ─────────────────────────
def get_json(url, tries=4):
    """GET a URL and parse JSON; return None after `tries` failed attempts."""
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=40)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == tries - 1:
                print(f"  ! GET failed {url}: {e}")
                return None
            time.sleep(1.0 + attempt)
    return None


def get_csv_rows(url, tries=4):
    """GET a Savant CSV export and return it as a list of dict rows (BOM-stripped)."""
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=40)
            r.raise_for_status()
            return list(csv.DictReader(io.StringIO(r.text.lstrip("﻿"))))
        except Exception as e:
            if attempt == tries - 1:
                print(f"  ! GET failed {url}: {e}")
                return []
            time.sleep(1.0 + attempt)
    return []


def as_float(x, ndigits=None):
    """Parse to float (rounded if ndigits given); return None on non-numeric input."""
    try:
        v = float(x)
        return round(v, ndigits) if ndigits is not None else v
    except (TypeError, ValueError):
        return None


def last_name_tag(full_name: str) -> str:
    """Lowercase alphanumeric last-name tag (e.g. 'Josh Naylor' -> 'naylor')."""
    parts = str(full_name).split()
    last = parts[-1] if parts else "runner"
    return "".join(ch for ch in last.lower() if ch.isalnum()) or "runner"


# ── per-attempt leads (feeds Burst) ──────────────────────────────────────────
def fetch_leads(runner_id: int, year: int) -> list[dict]:
    """One row per tracked steal attempt for a runner-season, from Savant's drawer."""
    d = get_json(LEADS_URL.format(rid=runner_id, y=year))
    data = d.get("data", []) if isinstance(d, dict) else (d or [])
    rows = []
    for a in data:
        rows.append({
            "runner_id": runner_id, "season": year,
            "date": str(a.get("game_date", ""))[:10],
            "play_id": a.get("play_id"),
            "pitcher_id": a.get("pitcher_id"), "pitcher_name": a.get("pitcher_name"),
            "catcher_id": a.get("catcher_id"), "catcher_name": a.get("catcher_name"),
            "fielder_name": a.get("fielder_name"),
            "base": a.get("target_base"), "result": a.get("runner_moved_cd"),
            "run_value": as_float(a.get("runs_stolen_on_running_act"), 3),
            "lead_at_firstmove_ft": as_float(a.get("r_primary_lead"), 1),
            "gain_to_release_ft": as_float(a.get("r_sec_minus_prim_lead"), 1),
            "lead_at_release_ft": as_float(a.get("r_secondary_lead"), 1),
        })
    rows.sort(key=lambda r: (r["date"] or "", str(r["pitcher_name"])))
    return rows


def statsapi_sb_cs(runner_id: int, year: int) -> tuple[int, int]:
    """A runner's official SB / CS season total from StatsAPI (the coverage-check baseline)."""
    d = get_json(GAMELOG_URL.format(rid=runner_id, y=year))
    sb = cs = 0
    if d and d.get("stats"):
        for s in d["stats"][0]["splits"]:
            st = s["stat"]
            sb += int(st.get("stolenBases", 0) or 0)
            cs += int(st.get("caughtStealing", 0) or 0)
    return sb, cs


def write_leads(runner_id: int, year: int, out: Path) -> int:
    """Fetch + write one runner-season's leads; print a StatsAPI coverage check. Returns row count."""
    rows = fetch_leads(runner_id, year)
    if not rows:
        print(f"  ! no attempts returned for {runner_id} {year}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEADS_COLS)
        w.writeheader()
        w.writerows(rows)
    n_sb = sum(1 for r in rows if r["result"] == "SB")
    n_cs = sum(1 for r in rows if r["result"] == "CS")
    sb, cs = statsapi_sb_cs(runner_id, year)
    print(f"[write] {out.name}  tracked {n_sb} SB / {n_cs} CS  |  "
          f"StatsAPI {year} total {sb} SB / {cs} CS  (gap = steals of home / non-2B-3B / untracked)")
    return len(rows)


# ── discovery (who to scrape) ────────────────────────────────────────────────
def fetch_sb_cs(start: int, end: int) -> dict:
    """Per-runner SB/CS summed across the season range (year-correct, StatsAPI)."""
    agg: dict = {}
    for y in range(start, end + 1):
        d = get_json(STATS_URL.format(y=y))
        if not d or not d.get("stats"):
            continue
        for s in d["stats"][0]["splits"]:
            stat, p = s["stat"], s["player"]
            sb = int(stat.get("stolenBases", 0) or 0)
            cs = int(stat.get("caughtStealing", 0) or 0)
            if sb + cs == 0:
                continue
            pid = int(p["id"])
            a = agg.setdefault(pid, {"runner_id": pid, "name": p["fullName"], "sb": 0, "cs": 0,
                                     "team": (s.get("team") or {}).get("abbreviation", ""),
                                     "position": (s.get("position") or {}).get("abbreviation", ""),
                                     "seasons": set()})
            a["sb"] += sb; a["cs"] += cs; a["seasons"].add(y)
    return agg


def fetch_sprint(start: int, end: int) -> dict:
    """Sprint speed + within-population percentile per runner (Savant, year-correct)."""
    rows = get_csv_rows(SPRINT_URL.format(s=start, e=end))
    out: dict = {}
    speeds = []
    for r in rows:
        try:
            pid, spd = int(r["player_id"]), float(r["sprint_speed"])
        except (KeyError, TypeError, ValueError):
            continue
        out[pid] = {"sprint_speed": spd, "team": r.get("team", ""), "position": r.get("position", "")}
        speeds.append((pid, spd))
    speeds.sort(key=lambda x: x[1])                      # low percentile = slow
    n = len(speeds)
    for rank, (pid, _) in enumerate(speeds):
        out[pid]["pctile"] = round(100.0 * rank / (n - 1), 1) if n > 1 else 0.0
    return out


def discover(start: int, end: int, min_attempts=1, max_sprint_pctile=None, top=None, sort="attempts"):
    """Rank base-stealers for the range; return the kept rows (also the --expand seed list)."""
    sbcs, sprint = fetch_sb_cs(start, end), fetch_sprint(start, end)
    rows = []
    for pid, a in sbcs.items():
        att = a["sb"] + a["cs"]
        sp = sprint.get(pid, {})
        rows.append({
            "runner_id": pid, "name": a["name"], "name_tag": last_name_tag(a["name"]),
            "team": a["team"] or sp.get("team", ""), "position": a["position"] or sp.get("position", ""),
            "sb": a["sb"], "cs": a["cs"], "attempts": att,
            "success_pct": round(100.0 * a["sb"] / att, 1) if att else "",
            "sprint_speed_ftps": sp.get("sprint_speed", ""), "sprint_pctile": sp.get("pctile", ""),
            "seasons": "/".join(str(y) for y in sorted(a["seasons"])),
        })
    rows = [r for r in rows if r["attempts"] >= min_attempts]
    if max_sprint_pctile is not None:
        rows = [r for r in rows if r["sprint_pctile"] != "" and r["sprint_pctile"] <= max_sprint_pctile]

    def slow_key(r):                                     # prolific AND slow
        pct = r["sprint_pctile"] if r["sprint_pctile"] != "" else 100.0
        return (r["attempts"], -pct)
    if sort == "slow":
        rows.sort(key=slow_key, reverse=True)
    else:
        rows.sort(key=lambda r: (r["attempts"], r["sb"]), reverse=True)
    return rows[:top] if top else rows


# ── visual assets (headshots + team map) ─────────────────────────────────────
# MLB Stats API abbreviation -> our logo filename in Output/assets/logos/*.png
ABBR_FIX = {"ARI": "AZ", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF",
            "TBR": "TB", "WSN": "WSH", "OAK": "ATH", "AthLetics": "ATH"}
HEADSHOT_URL = ("https://img.mlbstatic.com/mlb-photos/image/upload/"
                "d_people:generic:headshot:67:current.png/w_213,q_auto:best/"
                "v1/people/{pid}/headshot/67/current")
PERSON_STATS_URL = ("https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                    "?stats=season&season={yr}&group=hitting")



# ── v12: catcher pop time / arm strength (4 requests, one per season) ────────
def fetch_poptime(start: int, end: int) -> None:
    """Savant pop-time leaderboard per season -> Data/poptime.csv (joins on catcher_id)."""
    keep = ["pop_2b_sba", "pop_3b_sba", "maxeff_arm_2b_3b_sba", "exchange_2b_3b_sba",
            "pop_2b_sba_count"]
    out = []
    for y in range(start, end + 1):
        rows = get_csv_rows(POPTIME_URL.format(y=y))
        for r in rows:
            rec = {"catcher_id": r.get("entity_id"), "season": y,
                   "catcher_name": r.get("entity_name")}
            for k in keep:
                rec[k] = as_float(r.get(k), 3)
            out.append(rec)
        print(f"  poptime {y}: {len(rows)} catchers")
    POPTIME_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(POPTIME_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["catcher_id", "season", "catcher_name"] + keep)
        w.writeheader(); w.writerows(out)
    print(f"[write] {POPTIME_OUT.name}  ({len(out)} catcher-seasons)")



# ── v13: sprint speed for EVERY runner-season, not just qualified ones ───────
# The committed DF_v7_SSSI.csv only covers runner-seasons with >=10 attempts, which is why the
# web calculator could only be fit on 6,712 of 10,366 attempts. This pulls the full leaderboard
# (one small request per season) so speed exists for everyone; Burst is then recomputed offline.
# NOTE: named distinctly from the discover() helper `fetch_sprint` (a dict), which it must not shadow.
def fetch_sprint_leaderboard(start: int, end: int) -> None:
    """Full sprint-speed leaderboard per season -> Data/sprint_speed.csv."""
    out = []
    for y in range(start, end + 1):
        rows = get_csv_rows(SPRINT_URL.format(s=y, e=y))
        for r in rows:
            out.append({"runner_id": r.get("player_id"), "season": y,
                        "sprint_speed_all": as_float(r.get("sprint_speed"), 1)})
        print(f"  sprint speed {y}: {len(rows)} players")
    SPRINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SPRINT_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["runner_id", "season", "sprint_speed_all"])
        w.writeheader(); w.writerows(out)
    print(f"[write] {SPRINT_OUT.name}  ({len(out)} player-seasons)")



# ── v13: opportunity denominator (every pitch with a runner on 1B, 2B empty) ─
BASES = ("1B", "2B", "3B")


def _apply_movement(state, r):
    """Apply one runner movement to the base state (outs come from the feed's own count)."""
    mv = r.get("movement") or {}
    det = r.get("details") or {}
    rid = (det.get("runner") or {}).get("id")
    start, end = mv.get("originBase") or mv.get("start"), mv.get("end")
    if mv.get("isOut"):
        for b in BASES:
            if state[b] == rid:
                state[b] = None
        return
    if end and end != start:
        for b in BASES:
            if state[b] == rid:
                state[b] = None
        if end in BASES:
            state[end] = rid


def _game_opportunities(pk):
    """One row per pitch with a runner on 1B and 2B empty, carrying the base state and TRUE
    pre-pitch count. Returns (rows, plate_appearances_checked, mismatches) so the caller can
    gate on how well the base-state replay agrees with MLB's own end-of-PA state.

    Two things the feed does that are easy to get wrong:
      * a steal is its own non-pitch ACTION event inserted AFTER the pitch it happened on, so
        runners[].details.playIndex points past the pitch — attribute it to the preceding pitch;
      * playEvents[].count is the count AFTER that event, so the pre-pitch count (and outs) is
        the previous event's count, and 0-0 at the start of each plate appearance.
    """
    OPPS_DIR.mkdir(parents=True, exist_ok=True)
    cache = OPPS_DIR / f"{pk}.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text())
            return d["rows"], d["checked"], d["bad"]
        except Exception:
            pass

    d = get_json(PBP_OPPS_URL.format(pk=pk)) or {}
    rows, checked, bad = [], 0, 0
    state = {b: None for b in BASES}
    outs_now, cur_half = 0, None

    for play in d.get("allPlays", []):
        about = play.get("about") or {}
        half, inning = about.get("halfInning"), about.get("inning")
        if (half, inning) != cur_half:
            state = {b: None for b in BASES}
            outs_now, cur_half = 0, (half, inning)

        mu, res = play.get("matchup") or {}, play.get("result") or {}
        away, home = res.get("awayScore"), res.get("homeScore")
        diff = None if away is None or home is None else ((away - home) if half == "top" else (home - away))
        is_lhp = 1 if (mu.get("pitchHand") or {}).get("code") == "L" else 0
        bat_r = 1 if (mu.get("batSide") or {}).get("code") == "R" else 0

        moves = {}
        for r in play.get("runners", []):
            moves.setdefault((r.get("details") or {}).get("playIndex"), []).append(r)
        seen = set()

        balls = strikes = 0
        last_row = None                          # the most recent pitch row, for steal attribution
        for ev in play.get("playEvents", []):
            i = ev.get("index")
            if ev.get("isPitch"):
                on1, on2 = state["1B"], state["2B"]
                if on1 and not on2:
                    last_row = {"game_pk": pk, "at_bat": play.get("atBatIndex"),
                                "half": half, "inning": inning, "pitch_index": i,
                                "play_id": ev.get("playId"), "runner_1b": on1,
                                "outs": outs_now, "balls": balls, "strikes": strikes,
                                "score_diff": diff, "is_lhp": is_lhp, "bat_side_r": bat_r,
                                "pitch_code": ((ev.get("details") or {}).get("type") or {}).get("code"),
                                "attempt": 0, "attempt_type": ""}
                    rows.append(last_row)
                else:
                    last_row = None

            for r in moves.get(i, []):
                seen.add(i)
                et = ((r.get("details") or {}).get("eventType") or "")
                # steals/pickoffs belong to the pitch just thrown, not to this action event
                if et in ("stolen_base_2b", "caught_stealing_2b", "pickoff_1b",
                          "pickoff_caught_stealing_2b") and last_row is not None:
                    last_row["attempt_type"] = et
                    if et.startswith(("stolen_base", "caught_stealing")):
                        last_row["attempt"] = 1
                _apply_movement(state, r)

            c = ev.get("count") or {}            # authoritative post-event count / outs
            balls = c.get("balls", balls)
            strikes = c.get("strikes", strikes)
            outs_now = c.get("outs", outs_now)

        for idx in sorted((k for k in moves if k not in seen), key=lambda v: (v is None, v)):
            for r in moves[idx]:
                _apply_movement(state, r)

        # a third out ends the inning, so the bases are empty regardless of who reached
        pa_outs = (play.get("count") or {}).get("outs", outs_now)
        if pa_outs is not None and pa_outs >= 3:
            state = {b: None for b in BASES}

        checked += 1
        post = {b: (mu.get("postOn" + w) or {}).get("id")
                for b, w in zip(BASES, ("First", "Second", "Third"))}
        if any(state[b] != post[b] for b in BASES):
            bad += 1
            state = dict(post)                   # resync so one bad PA cannot poison the inning
        outs_now = (play.get("count") or {}).get("outs", outs_now)

    cache.write_text(json.dumps({"rows": rows, "checked": checked, "bad": bad},
                                separators=(",", ":")))
    return rows, checked, bad


def fetch_opportunities() -> None:
    """Build Data/Raw_Opportunities.csv — the denominator the success models never had."""
    src = ROOT / "Data" / "Raw_Attempts.csv"
    if not src.exists():
        sys.exit("Data/Raw_Attempts.csv missing — run build_features.py first")
    dates = sorted({r["date"][:10] for r in csv.DictReader(open(src)) if r.get("date")})
    print(f"walking {len(dates)} dates for regular-season games")

    pks = []
    for i, dt in enumerate(dates, 1):
        sched = get_json(SCHED_R_URL.format(date=dt)) or {}
        pks.extend(g.get("gamePk") for d_ in sched.get("dates", []) for g in d_.get("games", []))
        if i % 100 == 0:
            print(f"  schedule {i}/{len(dates)} -> {len(set(pks))} games")
    pks = sorted({p for p in pks if p})
    print(f"{len(pks)} regular-season games to read")

    cols = ["game_pk", "at_bat", "half", "inning", "pitch_index", "play_id", "runner_1b",
            "outs", "balls", "strikes", "score_diff", "is_lhp", "bat_side_r", "pitch_code",
            "attempt", "attempt_type"]
    n, checked, bad = 0, 0, 0
    with open(OPPS_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for i, pk in enumerate(pks, 1):
            rows, c, b = _game_opportunities(pk)
            checked += c; bad += b; n += len(rows)
            w.writerows(rows)
            if i % 250 == 0:
                acc = 100 * (1 - bad / max(1, checked))
                print(f"  games {i}/{len(pks)}  opportunities {n:,}  base-state accuracy {acc:.2f}%")
    acc = 100 * (1 - bad / max(1, checked))
    print(f"[write] {OPPS_OUT.name}  ({n:,} opportunity pitches from {len(pks)} games)")
    print(f"base-state reconstruction: {acc:.2f}% of {checked:,} plate appearances matched MLB's "
          f"own end-of-PA state  ({'PASS' if acc >= 99 else 'FAIL — investigate before modelling'})")


# ── v12: per-pitch context (handedness / pitch type / count / situation) ─────
def _game_context(pk: int) -> dict:
    """play_id -> pitch+situation for one game. Cached, so re-runs are cheap and resumable."""
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    cache = PBP_DIR / f"{pk}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    d = get_json(PBP_URL.format(pk=pk)) or {}
    ctx = {}
    for play in d.get("allPlays", []):
        ab, mu, res = play.get("about", {}), play.get("matchup", {}), play.get("result", {})
        half = ab.get("halfInning")
        away, home = res.get("awayScore"), res.get("homeScore")
        # score from the BATTING team's view: the runner's side is the batting team
        if away is not None and home is not None:
            diff = (away - home) if half == "top" else (home - away)
        else:
            diff = None
        for ev in play.get("playEvents", []):
            pid = ev.get("playId")
            if not pid:
                continue
            cnt = ev.get("count", {}) or {}
            ctx[pid] = {"is_lhp": 1 if (mu.get("pitchHand", {}) or {}).get("code") == "L" else 0,
                        "bat_side_r": 1 if (mu.get("batSide", {}) or {}).get("code") == "R" else 0,
                        "pitch_code": ((ev.get("details", {}) or {}).get("type", {}) or {}).get("code"),
                        "balls": cnt.get("balls"), "strikes": cnt.get("strikes"),
                        "outs": cnt.get("outs"), "inning": ab.get("inning"),
                        "score_diff": diff}
    cache.write_text(json.dumps(ctx, separators=(",", ":")))
    return ctx


def fetch_context() -> None:
    """Resolve every tracked attempt's pitch context by joining Savant play_id to the MLB
    play-by-play feed. Walks the schedule for each date our attempts fall on, caches one
    slim json per game, then writes Data/Raw_Attempt_Context.csv."""
    src = ROOT / "Data" / "Raw_Attempts.csv"
    if not src.exists():
        sys.exit("Data/Raw_Attempts.csv missing — run build_features.py first")
    att = list(csv.DictReader(open(src)))
    dates = sorted({r["date"][:10] for r in att if r.get("date")})
    print(f"resolving context for {len(att)} attempts across {len(dates)} dates")

    pks = []
    for i, d in enumerate(dates, 1):
        sched = get_json(SCHED_URL.format(date=d)) or {}
        day = [g.get("gamePk") for dt in sched.get("dates", []) for g in dt.get("games", [])]
        pks.extend(p for p in day if p)
        if i % 50 == 0:
            print(f"  schedule {i}/{len(dates)} dates -> {len(pks)} games so far")
    pks = sorted(set(pks))
    print(f"{len(pks)} games to read")

    ctx = {}
    for i, pk in enumerate(pks, 1):
        ctx.update(_game_context(pk))
        if i % 250 == 0:
            print(f"  games {i}/{len(pks)}  ({len(ctx)} pitches indexed)")

    cols = ["play_id", "is_lhp", "bat_side_r", "pitch_code", "balls", "strikes",
            "outs", "inning", "score_diff"]
    hit = 0
    with open(CONTEXT_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in att:
            c = ctx.get(r.get("play_id"))
            if not c:
                continue
            hit += 1
            w.writerow({"play_id": r["play_id"], **{k: c.get(k) for k in cols[1:]}})
    print(f"[write] {CONTEXT_OUT.name}  ({hit}/{len(att)} attempts matched, "
          f"{100*hit/max(1,len(att)):.1f}%)")


def fetch_assets():
    """Cache MLB headshots per runner and resolve each runner-season's team ->
    Data/team_map.csv (team abbreviations normalized to the logo filenames)."""
    import pandas as pd
    HEADSHOTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SEASON_MASTER, usecols=["runner_id", "season"]).drop_duplicates()
    logo_names = {p.stem for p in LOGOS.glob("*.png")}
    id2abbr = {t["id"]: t["abbreviation"]
               for t in (get_json("https://statsapi.mlb.com/api/v1/teams?sportId=1") or {}).get("teams", [])
               if t.get("id") and t.get("abbreviation")}

    for pid in set(df["runner_id"].astype(int)):
        out = HEADSHOTS / f"{pid}.png"
        if out.exists():
            continue
        try:
            r = SESSION.get(HEADSHOT_URL.format(pid=pid), timeout=20)
            if r.ok and r.content and len(r.content) > 1000:
                out.write_bytes(r.content)
        except Exception:
            pass

    rows, missing = [], set()
    for pid, yr in df[["runner_id", "season"]].astype(int).itertuples(index=False):
        d = get_json(PERSON_STATS_URL.format(pid=pid, yr=yr))
        splits = (d or {}).get("stats", [{}])[0].get("splits", []) if d else []
        abbr = splits[-1].get("team", {}).get("id") if splits else None      # last stint = latest team
        abbr = id2abbr.get(abbr)
        logo = ABBR_FIX.get(abbr, abbr) if abbr else None
        if logo and logo not in logo_names:
            missing.add(logo)
        rows.append({"runner_id": pid, "season": yr, "team": logo or ""})
        time.sleep(0.05)

    tm = pd.DataFrame(rows).drop_duplicates(["runner_id", "season"])
    tm.to_csv(TEAM_MAP_OUT, index=False)
    print(f"[write] {TEAM_MAP_OUT.name}  ({len(tm)} rows, {tm['team'].ne('').sum()} with team; "
          f"{len(list(HEADSHOTS.glob('*.png')))} headshots cached)")
    if missing:
        print(f"WARNING — team abbrevs with no logo file: {sorted(missing)}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("leads", help="pull one runner-season's per-attempt leads")
    pl.add_argument("runner_id", type=int)
    pl.add_argument("year", type=int)
    pl.add_argument("--out", help="output CSV (default Data/leads_cache/<id>_<year>.csv)")

    pd_ = sub.add_parser("discover", help="rank base-stealers for a season range")
    pd_.add_argument("--start", type=int, required=True)
    pd_.add_argument("--end", type=int, required=True)
    pd_.add_argument("--min-attempts", type=int, default=1)
    pd_.add_argument("--max-sprint-pctile", type=float, default=None)
    pd_.add_argument("--top", type=int, default=None)
    pd_.add_argument("--sort", choices=["attempts", "slow"], default="attempts")
    pd_.add_argument("--expand", action="store_true", help="then pull leads for every kept runner")

    sub.add_parser("assets", help="cache headshots + resolve Data/team_map.csv")

    pt = sub.add_parser("poptime", help="v12: catcher pop time + arm strength -> Data/poptime.csv")
    pt.add_argument("--start", type=int, default=2023); pt.add_argument("--end", type=int, default=2026)

    sp = sub.add_parser("sprint", help="v13: full sprint-speed leaderboard -> Data/sprint_speed.csv")
    sp.add_argument("--start", type=int, default=2023); sp.add_argument("--end", type=int, default=2026)

    sub.add_parser("opportunities", help="v13: every pitch with a runner on 1B -> Data/Raw_Opportunities.csv")

    sub.add_parser("context", help="v12: per-pitch handedness/count/situation -> Data/Raw_Attempt_Context.csv")

    args = ap.parse_args()

    if args.cmd == "assets":
        fetch_assets()
        return

    if args.cmd == "poptime":
        fetch_poptime(args.start, args.end)
        return

    if args.cmd == "sprint":
        fetch_sprint_leaderboard(args.start, args.end)
        return

    if args.cmd == "opportunities":
        fetch_opportunities()
        return

    if args.cmd == "context":
        fetch_context()
        return

    if args.cmd == "leads":
        out = Path(args.out) if args.out else LEADS_DIR / f"{args.runner_id}_{args.year}.csv"
        write_leads(args.runner_id, args.year, out)
        return

    rows = discover(args.start, args.end, args.min_attempts,
                    args.max_sprint_pctile, args.top, args.sort)
    DISC_DIR.mkdir(parents=True, exist_ok=True)
    out = DISC_DIR / f"runners_{args.start}_{args.end}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DISC_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {out.name}  ({len(rows)} runners)")
    for r in rows[:20]:
        print(f"{r['runner_id']:>9}  {r['name'][:22]:22s} att {r['attempts']:>3}  "
              f"{r['sb']:>3} SB / {r['cs']:>2} CS  sprint {str(r['sprint_speed_ftps']):>5} "
              f"(pct {str(r['sprint_pctile']):>5})")

    if args.expand:
        for r in rows:
            for y in range(args.start, args.end + 1):
                write_leads(r["runner_id"], y, LEADS_DIR / f"{r['runner_id']}_{y}.csv")


if __name__ == "__main__":
    main()
