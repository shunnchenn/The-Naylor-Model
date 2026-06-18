#!/usr/bin/env python3
"""
Josh Naylor 2025 Statcast base-stealing reference + head-to-head vs our CV metric.

IMPORTANT — Savant data sourcing (year correctness):
  * The basestealing-run-value LEADERBOARD CSV export IGNORES its ?year= param
    (it serves the current-season default regardless), so it CANNOT be used for a
    past-season reference. Verified: year=2025 and year=2026 return byte-identical
    rows (Naylor 9 SB / 1 CS both times).
  * The PER-ATTEMPT service
        /leaderboard/services/basestealing-running-game/{rid}?season_start=Y&season_end=Y
    DOES honor the year (2024->3, 2025->23, 2026->10 attempts). It is the authoritative
    source, and it is exactly what naylor2025_leads.csv was built from. So we compute
    the season SB-tracking anchors (SB/CS, run value, leads) from the leads CSV.
  * The sprint_speed leaderboard DOES honor ?min_season/?max_season, so sprint speed
    and home-to-1B are pulled live from there for the correct year.

Writes:
  statcast_ref_2025.csv         Naylor's season SB metrics (one row, year-correct)
  metric_vs_statcast_2025.csv   per-play velocity joined w/ season anchors

Network: Savant has no sandbox DNS -> run with dangerouslyDisableSandbox.
"""
import csv, io, os, statistics as st, requests

HERE = os.path.dirname(os.path.abspath(__file__))
NAYLOR = 647304
YEAR = 2025
S = requests.Session(); S.headers.update({"User-Agent": "Mozilla/5.0"})

SPRINT = ("https://baseballsavant.mlb.com/leaderboard/sprint_speed"
          f"?attempts=1&min_season={YEAR}&max_season={YEAR}&position=&team=&csv=true")


def getcsv(u):
    return list(csv.DictReader(io.StringIO(S.get(u, timeout=30).text.lstrip("﻿"))))


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def main():
    # ---- sprint speed (year-honored leaderboard) ----
    sprint = getcsv(SPRINT)
    s = next(r for r in sprint if r.get("player_id") == str(NAYLOR))

    # ---- season SB-tracking anchors computed from per-attempt leads (year-correct) ----
    leads = list(csv.DictReader(open(os.path.join(HERE, "naylor2025_leads.csv"))))
    sb_rows = [r for r in leads if r["result"] == "SB"]
    cs_rows = [r for r in leads if r["result"] == "CS"]
    n_sb, n_cs = len(sb_rows), len(cs_rows)
    fm_all = [fnum(r["lead_at_firstmove_ft"]) for r in leads]
    rel_all = [fnum(r["lead_at_release_ft"]) for r in leads]
    gain_all = [fnum(r["gain_to_release_ft"]) for r in leads]
    fm_sb = [fnum(r["lead_at_firstmove_ft"]) for r in sb_rows]
    rel_sb = [fnum(r["lead_at_release_ft"]) for r in sb_rows]

    ref = {
        "player": "Josh Naylor", "season": YEAR,
        "sprint_speed_ftps": s["sprint_speed"],
        "hp_to_1b_s": s["hp_to_1b"],
        "competitive_runs": s["competitive_runs"],
        "n_tracked_attempts": len(leads),
        "n_sb": n_sb, "n_cs": n_cs,
        "sb_success_pct": round(100 * n_sb / (n_sb + n_cs), 1),
        "steal_run_value": round(sum(fnum(r["run_value"]) for r in leads), 3),
        "primary_lead_ft": round(st.mean(fm_all), 2),
        "secondary_lead_ft": round(st.mean(rel_all), 2),
        "sec_minus_prim_lead_ft": round(st.mean(gain_all), 2),
        "primary_lead_sbx_ft": round(st.mean(fm_sb), 2),
        "secondary_lead_sbx_ft": round(st.mean(rel_sb), 2),
        "source_note": "SB anchors from per-attempt basestealing-running-game (year-correct); "
                       "sprint from sprint_speed leaderboard; leaderboard CSV year-param is broken",
    }
    with open(os.path.join(HERE, "statcast_ref_2025.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ref)); w.writeheader(); w.writerow(ref)

    # ---- our per-play metric ----
    dv = list(csv.DictReader(open(os.path.join(HERE, "delivery_velocity_2025.csv"))))
    usable = [r for r in dv if r["avg_velocity_ftps"] != ""]
    vel = [float(r["avg_velocity_ftps"]) for r in usable]
    lfm = [float(r["lead_at_firstmove_ft"]) for r in dv]
    lrel = [float(r["lead_at_release_ft"]) for r in dv]
    sprint_v = fnum(ref["sprint_speed_ftps"])

    print("=" * 78)
    print(f"JOSH NAYLOR {YEAR} — SB-TRACKING METRICS  (Statcast season + our CV metric)")
    print("=" * 78)
    print("SEASON STATCAST (year-correct; SB anchors from per-attempt service):")
    print(f"  sprint speed          {ref['sprint_speed_ftps']:>6} ft/s")
    print(f"  home-to-1B            {ref['hp_to_1b_s']:>6} s")
    print(f"  tracked attempts      {ref['n_tracked_attempts']:>6}")
    print(f"  primary lead (all)    {ref['primary_lead_ft']:>6} ft")
    print(f"  secondary lead (all)  {ref['secondary_lead_ft']:>6} ft")
    print(f"  sec-minus-prim gain   {ref['sec_minus_prim_lead_ft']:>6} ft")
    print(f"  primary lead (steals) {ref['primary_lead_sbx_ft']:>6} ft")
    print(f"  secondary lead(steals){ref['secondary_lead_sbx_ft']:>6} ft")
    print(f"  SB / CS               {ref['n_sb']} / {ref['n_cs']}  "
          f"({ref['sb_success_pct']}% success)")
    print(f"  steal run value       {ref['steal_run_value']:>+6} runs")
    print()
    print("OUR CV METRIC (delivery-window velocity, per-attempt — varies by matchup):")
    print(f"  mean {st.mean(vel):.2f}  median {st.median(vel):.2f}  "
          f"range {min(vel):.2f}-{max(vel):.2f} ft/s   (n={len(vel)})")
    print()
    print("-" * 78)
    print("CROSS-VALIDATION  (these are the same per-attempt rows, sanity check):")
    print(f"  mean lead@firstmove = {st.mean(lfm):5.2f} ft   (= primary lead all = {ref['primary_lead_ft']})")
    print(f"  mean lead@release   = {st.mean(lrel):5.2f} ft   (= secondary lead all = {ref['secondary_lead_ft']})")
    print()
    print("-" * 78)
    print(f"HEAD-TO-HEAD — what each metric can/cannot do for these {len(dv)} attempts:")
    print(f"  sprint speed ({sprint_v:.1f} ft/s): runner-only, SAME every play -> 0 per-play")
    print(f"     resolution; aggregates can't tell a good matchup from a bad one.")
    print(f"  our velocity (mean {st.mean(vel):.1f} ft/s): runner x PITCHER interaction ->")
    print(f"     varies {min(vel):.1f}-{max(vel):.1f} ft/s across matchups; encodes the delivery he")
    print(f"     exploited. The lone CS (Littell) is the LOWEST velocity in the set.")
    print(f"  ratio our-vel / sprint-speed = {st.mean(vel)/sprint_v:.2f}  "
          f"(~{100*st.mean(vel)/sprint_v:.0f}% of top speed during the delivery window)")
    print("=" * 78)

    rows = [(float(r["avg_velocity_ftps"]) if r["avg_velocity_ftps"] != "" else -1, r) for r in dv]
    rows.sort(key=lambda x: -x[0])
    out = []
    for v, r in rows:
        out.append({
            "date": r["date"], "pitcher_name": r["pitcher_name"], "result": r["result"],
            "avg_velocity_ftps": r["avg_velocity_ftps"],
            "lead_at_release_ft": r["lead_at_release_ft"],
            "delivery_s": r["delivery_s"], "analysis_method": r["analysis_method"],
            "naylor_sprint_speed_ftps": ref["sprint_speed_ftps"],
            "naylor_secondary_lead_sbx_ft": ref["secondary_lead_sbx_ft"],
        })
    with open(os.path.join(HERE, "metric_vs_statcast_2025.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("\nwrote statcast_ref_2025.csv + metric_vs_statcast_2025.csv")


if __name__ == "__main__":
    main()
