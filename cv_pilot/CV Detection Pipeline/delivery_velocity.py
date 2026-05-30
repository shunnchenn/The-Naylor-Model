#!/usr/bin/env python3
"""
Generalized delivery-window runner velocity metric (any runner / year).

    avg_velocity_ftps = gain_to_release_ft / delivery_s

gain_to_release_ft = lead_at_release_ft - lead_at_firstmove_ft   (Statcast lead data)
delivery_s         = CV-measured first-move -> ball-release time  (cv_pilot detector)

Joins leads ⟕ delivery on play_id (the Savant GUID), robust to duplicate
(date, pitcher) pairs. Replaces the per-folder delivery_velocity_<year>.py copies.

Usage:
  python3 cv_pilot/delivery_velocity.py --dir cv_pilot/Vladdy_2025 \
      --leads vladdy2025_leads.csv --year 2025 --runner "Vladimir Guerrero Jr."

Output: <dir>/delivery_velocity_<year>.csv
"""
import argparse, csv, os, statistics as st


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--leads", required=True, help="leads filename inside --dir")
    ap.add_argument("--year", required=True)
    ap.add_argument("--runner", default="runner")
    args = ap.parse_args()

    HERE = args.dir
    LEADS = os.path.join(HERE, args.leads)
    DELIV = os.path.join(HERE, f"pitcher_delivery_{args.year}.csv")
    OUT = os.path.join(HERE, f"delivery_velocity_{args.year}.csv")

    leads = load_csv(LEADS)
    deliv = {r["play_id"]: r for r in load_csv(DELIV)}

    rows = []
    for L in leads:
        d = deliv.get(L["play_id"])
        if d is None:
            continue
        delivery_s = fnum(d.get("delivery_s"))
        usable = str(d.get("usable", "")).strip().lower() == "true"
        gain = fnum(L.get("gain_to_release_ft"))
        avg_vel = None
        if usable and delivery_s and delivery_s > 0 and gain is not None:
            avg_vel = round(gain / delivery_s, 3)
        rows.append({
            "date": L["date"], "pitcher_name": L["pitcher_name"],
            "pitcher_id": L["pitcher_id"], "p_throws": d.get("p_throws", ""),
            "base": L["base"], "result": L["result"], "run_value": L["run_value"],
            "lead_at_firstmove_ft": L["lead_at_firstmove_ft"],
            "lead_at_release_ft": L["lead_at_release_ft"],
            "gain_to_release_ft": L["gain_to_release_ft"],
            "delivery_s": d.get("delivery_s", ""), "release_conf": d.get("release_conf", ""),
            "usable": d.get("usable", ""), "analysis_method": d.get("analysis_method", ""),
            "play_id": L["play_id"],
            "avg_velocity_ftps": "" if avg_vel is None else avg_vel,
        })

    cols = ["date", "pitcher_name", "pitcher_id", "p_throws", "base", "result",
            "run_value", "lead_at_firstmove_ft", "lead_at_release_ft",
            "gain_to_release_ft", "delivery_s", "release_conf", "usable",
            "analysis_method", "play_id", "avg_velocity_ftps"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    usable_rows = [r for r in rows if r["avg_velocity_ftps"] != ""]
    vals = [float(r["avg_velocity_ftps"]) for r in usable_rows]

    print("=" * 74)
    print(f"DELIVERY-WINDOW RUNNER VELOCITY  ({args.runner}, {args.year})")
    print("  avg_velocity_ftps = gain_to_release_ft / delivery_s")
    print("=" * 74)
    hdr = f"{'date':10} {'pitcher':22} {'res':4} {'gain':>5} {'deliv':>6} {'vel':>7} {'meth':>9}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["avg_velocity_ftps"] == "",
                    -(float(x["avg_velocity_ftps"]) if x["avg_velocity_ftps"] != "" else 0))):
        vel = r["avg_velocity_ftps"]
        vel_s = f"{float(vel):7.2f}" if vel != "" else "    n/a"
        dv = r["delivery_s"] or "n/a"
        g = fnum(r["gain_to_release_ft"])
        g_s = f"{g:5.1f}" if g is not None else "  n/a"
        print(f"{r['date']:10} {r['pitcher_name']:22} {r['result']:4} "
              f"{g_s} {str(dv):>6} {vel_s} {r['analysis_method']:>9}")
    print("-" * len(hdr))
    print(f"usable plays: {len(usable_rows)}/{len(rows)}")
    if vals:
        print(f"velocity ft/s  mean={st.mean(vals):.2f}  median={st.median(vals):.2f}  "
              f"min={min(vals):.2f}  max={max(vals):.2f}  "
              f"std={st.pstdev(vals):.2f}")

    sb = [float(r["avg_velocity_ftps"]) for r in usable_rows if r["result"] == "SB"]
    cs = [float(r["avg_velocity_ftps"]) for r in usable_rows if r["result"] == "CS"]
    print()
    if sb:
        print(f"  SB (n={len(sb)})  mean vel = {st.mean(sb):.2f} ft/s")
    if cs:
        print(f"  CS (n={len(cs)})  mean vel = {st.mean(cs):.2f} ft/s")
    print("=" * 74)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
