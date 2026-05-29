#!/usr/bin/env python3
"""
Generalized Statcast base-stealing reference + head-to-head vs our CV metric
(any runner / year). Replaces the per-folder statcast_ref_<year>.py copies.

IMPORTANT — Savant data sourcing (year correctness):
  * The basestealing-run-value LEADERBOARD CSV export IGNORES its ?year= param, so it
    CANNOT be used for a past-season reference. The PER-ATTEMPT service
        /leaderboard/services/basestealing-running-game/{rid}?season_start=Y&season_end=Y
    DOES honor the year and is what <leads>.csv was built from, so the season
    SB-tracking anchors are computed from the leads CSV.
  * The sprint_speed leaderboard DOES honor ?min_season/?max_season.

Usage:
  python3 cv_pilot/statcast_ref.py --dir cv_pilot/Vladdy_2025 \
      --leads vladdy2025_leads.csv --year 2025 \
      --runner-id 665489 --runner "Vladimir Guerrero Jr."

Network: Savant has no sandbox DNS -> run with dangerouslyDisableSandbox.
Writes: <dir>/statcast_ref_<year>.csv, <dir>/metric_vs_statcast_<year>.csv
"""
import argparse, csv, io, os, statistics as st, requests

S = requests.Session(); S.headers.update({"User-Agent": "Mozilla/5.0"})


def getcsv(u):
    return list(csv.DictReader(io.StringIO(S.get(u, timeout=30).text.lstrip("﻿"))))


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--leads", required=True)
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--runner-id", required=True)
    ap.add_argument("--runner", default="runner")
    args = ap.parse_args()

    HERE = args.dir
    LEADS = os.path.join(HERE, args.leads)
    DV = os.path.join(HERE, f"delivery_velocity_{args.year}.csv")
    REF_OUT = os.path.join(HERE, f"statcast_ref_{args.year}.csv")
    H2H_OUT = os.path.join(HERE, f"metric_vs_statcast_{args.year}.csv")
    rid = str(args.runner_id)

    SPRINT = ("https://baseballsavant.mlb.com/leaderboard/sprint_speed"
              f"?attempts=1&min_season={args.year}&max_season={args.year}"
              "&position=&team=&csv=true")
    sprint = getcsv(SPRINT)
    s = next((r for r in sprint if r.get("player_id") == rid), None)

    leads = list(csv.DictReader(open(LEADS)))
    sb_rows = [r for r in leads if r["result"] == "SB"]
    cs_rows = [r for r in leads if r["result"] == "CS"]
    n_sb, n_cs = len(sb_rows), len(cs_rows)
    lead_rows = [r for r in leads if fnum(r["lead_at_firstmove_ft"]) is not None]
    fm_all = [fnum(r["lead_at_firstmove_ft"]) for r in lead_rows]
    rel_all = [fnum(r["lead_at_release_ft"]) for r in lead_rows]
    gain_all = [fnum(r["gain_to_release_ft"]) for r in lead_rows]
    fm_sb = [fnum(r["lead_at_firstmove_ft"]) for r in sb_rows if fnum(r["lead_at_firstmove_ft"]) is not None]
    rel_sb = [fnum(r["lead_at_release_ft"]) for r in sb_rows if fnum(r["lead_at_release_ft"]) is not None]

    ref = {
        "player": args.runner, "season": args.year,
        "sprint_speed_ftps": (s or {}).get("sprint_speed", ""),
        "hp_to_1b_s": (s or {}).get("hp_to_1b", ""),
        "competitive_runs": (s or {}).get("competitive_runs", ""),
        "n_tracked_attempts": len(leads), "n_sb": n_sb, "n_cs": n_cs,
        "sb_success_pct": round(100 * n_sb / (n_sb + n_cs), 1) if (n_sb + n_cs) else "",
        "steal_run_value": round(sum(fnum(r["run_value"]) for r in leads if fnum(r["run_value"]) is not None), 3),
        "primary_lead_ft": round(st.mean(fm_all), 2) if fm_all else "",
        "secondary_lead_ft": round(st.mean(rel_all), 2) if rel_all else "",
        "sec_minus_prim_lead_ft": round(st.mean(gain_all), 2) if gain_all else "",
        "primary_lead_sbx_ft": round(st.mean(fm_sb), 2) if fm_sb else "",
        "secondary_lead_sbx_ft": round(st.mean(rel_sb), 2) if rel_sb else "",
        "source_note": "SB anchors from per-attempt basestealing-running-game (year-correct); "
                       "sprint from sprint_speed leaderboard; leaderboard CSV year-param is broken",
    }
    with open(REF_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ref)); w.writeheader(); w.writerow(ref)

    dv = list(csv.DictReader(open(DV)))
    usable = [r for r in dv if r["avg_velocity_ftps"] != ""]
    vel = [float(r["avg_velocity_ftps"]) for r in usable]
    lfm = [float(r["lead_at_firstmove_ft"]) for r in dv if r["lead_at_firstmove_ft"]]
    lrel = [float(r["lead_at_release_ft"]) for r in dv if r["lead_at_release_ft"]]
    sprint_v = fnum(ref["sprint_speed_ftps"])

    print("=" * 78)
    print(f"{args.runner.upper()} {args.year} — SB-TRACKING METRICS  (Statcast season + our CV metric)")
    print("=" * 78)
    print("SEASON STATCAST (year-correct; SB anchors from per-attempt service):")
    print(f"  sprint speed          {ref['sprint_speed_ftps']:>6} ft/s")
    print(f"  home-to-1B            {ref['hp_to_1b_s']:>6} s")
    print(f"  tracked attempts      {ref['n_tracked_attempts']:>6}")
    print(f"  primary lead (all)    {ref['primary_lead_ft']:>6} ft")
    print(f"  secondary lead (all)  {ref['secondary_lead_ft']:>6} ft")
    print(f"  sec-minus-prim gain   {ref['sec_minus_prim_lead_ft']:>6} ft")
    print(f"  SB / CS               {ref['n_sb']} / {ref['n_cs']}  "
          f"({ref['sb_success_pct']}% success)")
    print(f"  steal run value       {ref['steal_run_value']:>+6} runs")
    print()
    if vel:
        print("OUR CV METRIC (delivery-window velocity, per-attempt — varies by matchup):")
        print(f"  mean {st.mean(vel):.2f}  median {st.median(vel):.2f}  "
              f"range {min(vel):.2f}-{max(vel):.2f} ft/s   (n={len(vel)})")
        print()
        print("CROSS-VALIDATION  (same per-attempt rows, sanity check):")
        if lfm:
            print(f"  mean lead@firstmove = {st.mean(lfm):5.2f} ft   (= primary lead all = {ref['primary_lead_ft']})")
        if lrel:
            print(f"  mean lead@release   = {st.mean(lrel):5.2f} ft   (= secondary lead all = {ref['secondary_lead_ft']})")
        if sprint_v:
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
            "runner_sprint_speed_ftps": ref["sprint_speed_ftps"],
            "runner_secondary_lead_sbx_ft": ref["secondary_lead_sbx_ft"],
        })
    if out:
        with open(H2H_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"\nwrote {os.path.basename(REF_OUT)} + {os.path.basename(H2H_OUT)}")


if __name__ == "__main__":
    main()
