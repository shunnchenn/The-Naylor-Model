#!/usr/bin/env python3
"""
Pull Josh Naylor's 2026 Statcast base-stealing reference metrics from Baseball
Savant and rank our delivery-window velocity metric head-to-head against them.

Savant hosts (note: NO hyphen):
  - basestealing-run-value leaderboard (runner view) -> leads, run value, SB/CS
  - sprint_speed leaderboard                          -> sprint speed, hp_to_1b

Writes:
  statcast_ref_2026.csv         Naylor's season SB metrics (one row)
  metric_vs_statcast_2026.csv   per-play velocity joined w/ season anchors
Prints the head-to-head read-out.
"""
import csv, io, os, statistics as st, requests

HERE = os.path.dirname(os.path.abspath(__file__))
NAYLOR = 647304
UA = {"User-Agent": "Mozilla/5.0"}
S = requests.Session(); S.headers.update(UA)

BSRV = ("https://baseballsavant.mlb.com/leaderboard/basestealing-run-value"
        "?type=runner&year=2026&team=&min=0&sortColumn=runner_runs_swiped_total"
        "&sortDirection=desc&csv=true")
SPRINT = ("https://baseballsavant.mlb.com/leaderboard/sprint_speed"
          "?attempts=1&min_season=2026&max_season=2026&position=&team=&csv=true")


def getcsv(u):
    t = S.get(u, timeout=30).text.lstrip("﻿")
    return list(csv.DictReader(io.StringIO(t)))


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def main():
    bsrv = getcsv(BSRV)
    sprint = getcsv(SPRINT)
    b = next(r for r in bsrv if r.get("player_id") == str(NAYLOR))
    s = next(r for r in sprint if r.get("player_id") == str(NAYLOR))

    ref = {
        "player": "Josh Naylor", "season": 2026,
        "sprint_speed_ftps": s["sprint_speed"],
        "hp_to_1b_s": s["hp_to_1b"],
        "competitive_runs": s["competitive_runs"],
        "n_sb": b["n_sb"], "n_cs": b["n_cs"],
        "sb_success_pct": round(100 * int(b["n_sb"]) / (int(b["n_sb"]) + int(b["n_cs"])), 1),
        "steal_run_value": round(fnum(b["runs_stolen_on_running_act"]), 3),
        "attempt_rate_sbx": b["rate_sbx"],
        "n_init": b["n_init"],
        "primary_lead_ft": round(fnum(b["r_primary_lead"]), 2),
        "secondary_lead_ft": round(fnum(b["r_secondary_lead"]), 2),
        "sec_minus_prim_lead_ft": round(fnum(b["r_sec_minus_prim_lead"]), 2),
        "primary_lead_sbx_ft": round(fnum(b["r_primary_lead_sbx"]), 2),
        "secondary_lead_sbx_ft": round(fnum(b["r_secondary_lead_sbx"]), 2),
    }
    with open(os.path.join(HERE, "statcast_ref_2026.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ref)); w.writeheader(); w.writerow(ref)

    # ---- our per-play metric ----
    dv = list(csv.DictReader(open(os.path.join(HERE, "delivery_velocity_2026.csv"))))
    usable = [r for r in dv if r["avg_velocity_ftps"] != ""]
    vel = [float(r["avg_velocity_ftps"]) for r in usable]
    lfm = [float(r["lead_at_firstmove_ft"]) for r in dv]
    lrel = [float(r["lead_at_release_ft"]) for r in dv]

    sprint_v = fnum(ref["sprint_speed_ftps"])

    print("=" * 78)
    print("JOSH NAYLOR 2026 — SB-TRACKING METRICS  (Statcast season + our CV metric)")
    print("=" * 78)
    print("SEASON STATCAST (runner-only, constant across all attempts):")
    print(f"  sprint speed          {ref['sprint_speed_ftps']:>6} ft/s   "
          f"(MLB avg ~27.0 -> Naylor is SLOW, ~12th pctile)")
    print(f"  home-to-1B            {ref['hp_to_1b_s']:>6} s")
    print(f"  primary lead          {ref['primary_lead_ft']:>6} ft")
    print(f"  secondary lead        {ref['secondary_lead_ft']:>6} ft")
    print(f"  sec-minus-prim gain   {ref['sec_minus_prim_lead_ft']:>6} ft")
    print(f"  primary lead (steals) {ref['primary_lead_sbx_ft']:>6} ft")
    print(f"  secondary lead(steals){ref['secondary_lead_sbx_ft']:>6} ft")
    print(f"  SB / CS               {ref['n_sb']} / {ref['n_cs']}  "
          f"({ref['sb_success_pct']}% success)")
    print(f"  steal run value       {ref['steal_run_value']:>+6} runs")
    print(f"  attempt rate          {ref['attempt_rate_sbx']}  (of {ref['n_init']} opportunities)")
    print()
    print("OUR CV METRIC (delivery-window velocity, per-attempt — varies by matchup):")
    print(f"  mean {st.mean(vel):.2f}  median {st.median(vel):.2f}  "
          f"range {min(vel):.2f}-{max(vel):.2f} ft/s   (n={len(vel)})")
    print()
    print("-" * 78)
    print("CROSS-VALIDATION  (our per-play lead data vs Savant season anchors):")
    print(f"  our mean lead@firstmove = {st.mean(lfm):5.2f} ft   "
          f"Savant primary-lead(steals) = {ref['primary_lead_sbx_ft']} ft")
    print(f"  our mean lead@release   = {st.mean(lrel):5.2f} ft   "
          f"Savant secondary-lead(steals) = {ref['secondary_lead_sbx_ft']} ft  <- matches")
    print()
    print("-" * 78)
    print("HEAD-TO-HEAD — what each metric can and cannot do for these 10 attempts:")
    print(f"  sprint speed (24.5 ft/s): runner-only, SAME every play -> 0 per-play")
    print(f"     resolution; would predict Naylor a BAD base-stealer, yet he's 90% / +1.19 runs.")
    print(f"  our velocity (mean {st.mean(vel):.1f} ft/s): runner x PITCHER interaction ->")
    print(f"     varies {min(vel):.1f}-{max(vel):.1f} ft/s across matchups; encodes the delivery he")
    print(f"     exploited. This is the resolution sprint speed/lead aggregates lack.")
    print(f"  ratio our-vel / sprint-speed = {st.mean(vel)/sprint_v:.2f}  "
          f"(he reaches ~{100*st.mean(vel)/sprint_v:.0f}% of top speed during the delivery window)")
    print("=" * 78)

    # per-play table sorted by velocity, flag the CS
    rows = []
    for r in dv:
        v = r["avg_velocity_ftps"]
        rows.append((float(v) if v != "" else -1, r))
    rows.sort(key=lambda x: -x[0])
    print(f"{'pitcher':20} {'res':4} {'vel':>6} {'lead@rel':>9} {'deliv':>6}")
    out = []
    for v, r in rows:
        vs = f"{v:6.2f}" if v >= 0 else "   n/a"
        print(f"{r['pitcher_name']:20} {r['result']:4} {vs} "
              f"{float(r['lead_at_release_ft']):9.1f} {r['delivery_s'] or 'n/a':>6}")
        out.append({
            "pitcher_name": r["pitcher_name"], "result": r["result"],
            "avg_velocity_ftps": r["avg_velocity_ftps"],
            "lead_at_release_ft": r["lead_at_release_ft"],
            "delivery_s": r["delivery_s"],
            "naylor_sprint_speed_ftps": ref["sprint_speed_ftps"],
            "naylor_secondary_lead_sbx_ft": ref["secondary_lead_sbx_ft"],
        })
    with open(os.path.join(HERE, "metric_vs_statcast_2026.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("\nwrote statcast_ref_2026.csv + metric_vs_statcast_2026.csv")


if __name__ == "__main__":
    main()
