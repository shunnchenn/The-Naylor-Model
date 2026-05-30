#!/usr/bin/env python3
"""
Generalized per-attempt delivery-table builder (any runner / year).

Assembles <dir>/pitcher_delivery_<year>.csv (the schema delivery_velocity_*.py reads)
from:
  - <dir>/pilot_results.csv       full detector pass over the clips
  - /tmp/rec_<clip>.csv           optional per-clip 6s-cap recovery runs
  - <dir>/<leads>.csv             date / base / result join keys
  - a small reuse-fallback dict   (same-pitcher measured delivery) when a clip is
                                  un-measurable even after the cap

Usage:
  python3 cv_pilot/build_delivery.py --dir cv_pilot/Soto_2025 \
      --leads soto2025_leads.csv --year 2025 \
      [--recovered NAYLOR_x,NAYLOR_y] [--reuse play_id=delivery_s:note,...]

Selection per attempt:
  1. full pass usable=True            -> analysis_method=full
  2. else 6s-cap recovery usable=True -> analysis_method=capped_6s
  3. else reuse-fallback if provided  -> analysis_method=reused
  4. else keep failed number, usable=False
"""
import argparse, csv, os


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def truthy(x):
    return str(x).strip().lower() == "true"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--leads", required=True, help="leads filename inside --dir")
    ap.add_argument("--year", required=True)
    ap.add_argument("--recovered", default="", help="comma-separated clip_ids re-run with 6s cap")
    ap.add_argument("--reuse", default="",
                    help="comma-separated play_id=delivery_s:note entries")
    args = ap.parse_args()

    HERE = args.dir
    PILOT = os.path.join(HERE, "pilot_results.csv")
    LEADS = os.path.join(HERE, args.leads)
    OUT = os.path.join(HERE, f"pitcher_delivery_{args.year}.csv")

    recovered = [c for c in args.recovered.split(",") if c.strip()]
    reuse = {}
    for ent in args.reuse.split(","):
        ent = ent.strip()
        if not ent:
            continue
        pid, rest = ent.split("=", 1)
        ds, _, note = rest.partition(":")
        reuse[pid] = (float(ds), note or "reused")

    pilot = {r["play_id"]: r for r in load(PILOT)}
    leads = {r["play_id"]: r for r in load(LEADS)}

    rec = {}
    for c in recovered:
        p = f"/tmp/rec_{c}.csv"
        if os.path.exists(p):
            row = load(p)[0]
            rec[row["play_id"]] = row

    out_rows = []
    for play_id, L in leads.items():
        base = L["base"]; result = L["result"]
        event = ("Caught Stealing " if result == "CS" else "Stolen Base ") + base
        p = pilot.get(play_id, {})
        r = rec.get(play_id)
        delivery_s = usable = release_conf = method = note = None

        if p and truthy(p.get("usable")):
            delivery_s = p.get("delivery_s"); usable = True
            release_conf = p.get("release_kpt_conf"); method = "full"
        elif r and truthy(r.get("usable")):
            delivery_s = r.get("delivery_s"); usable = True
            release_conf = r.get("release_kpt_conf"); method = "capped_6s"
            note = "recovered via 6s analysis cap (broadcast follows runner)"
        elif play_id in reuse:
            delivery_s, note = reuse[play_id]
            usable = True; release_conf = ""; method = "reused"
        else:
            delivery_s = p.get("delivery_s", ""); usable = False
            release_conf = p.get("release_kpt_conf", ""); method = "full"
            note = "un-measurable; no same-pitcher fallback available"

        out_rows.append({
            "date": L["date"], "pitcher_id": L["pitcher_id"],
            "pitcher_name": L["pitcher_name"],
            "p_throws": p.get("p_throws", r.get("p_throws", "") if r else ""),
            "event": event, "catcher_name": L["catcher_name"],
            "clip_id": p.get("clip_id", r.get("clip_id", "") if r else ""),
            "play_id": play_id, "fps": p.get("fps", ""),
            "delivery_s": delivery_s, "usable": usable,
            "release_conf": release_conf, "analysis_method": method,
            "note": note or "",
        })

    out_rows.sort(key=lambda x: x["date"])
    cols = ["date", "pitcher_id", "pitcher_name", "p_throws", "event",
            "catcher_name", "clip_id", "play_id", "fps", "delivery_s",
            "usable", "release_conf", "analysis_method", "note"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(out_rows)

    n_us = sum(1 for r in out_rows if r["usable"])
    n_re = sum(1 for r in out_rows if r["analysis_method"] == "reused")
    n_cap = sum(1 for r in out_rows if r["analysis_method"] == "capped_6s")
    print(f"[write] {OUT}  ({len(out_rows)} attempts; {n_us} usable; "
          f"{n_cap} capped_6s; {n_re} reused)")
    for r in out_rows:
        if not r["usable"] or r["analysis_method"] != "full":
            print(f"  {r['date']} {r['pitcher_name']:22} {r['analysis_method']:9} "
                  f"d={r['delivery_s']} usable={r['usable']}  {r['note']}")


if __name__ == "__main__":
    main()
