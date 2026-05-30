#!/usr/bin/env python3
"""
Assemble pitcher_delivery_2025.csv (mirrors the 2026 schema) from:
  - pilot_results.csv         full detector pass over all 23 clips
  - /tmp/rec_<clip>.csv        per-clip 6s-cap recovery runs (broadcast-follows-runner)
  - naylor2025_leads.csv       date / base / result / run_value join keys
  - reuse-fallback             a same-pitcher measured delivery when a 2025 clip is
                               un-measurable even after the cap

Selection per clip:
  1. if the full pass produced usable=True -> use it (analysis_method=full)
  2. else if the 6s-cap recovery produced usable=True -> use it (analysis_method=capped_6s)
  3. else apply reuse-fallback (analysis_method=reused, note explains the source)

The reuse table is intentionally explicit (small, auditable) rather than auto-joined.
"""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, "pilot_results.csv")
LEADS = os.path.join(HERE, "naylor2025_leads.csv")
OUT = os.path.join(HERE, "pitcher_delivery_2025.csv")

# clips re-run with the 6s analysis cap (whether or not they recovered)
RECOVERED = [
    "NAYLOR_03ee0445_kerkering_R", "NAYLOR_04855f81_gausman_R",
    "NAYLOR_09e58c6b_lopez_R", "NAYLOR_0a8f1dbd_eisert_L",
    "NAYLOR_39a80ca9_martin_R", "NAYLOR_8f12e2af_hendricks_R",
    "NAYLOR_af7bfe85_littell_R",
]

# reuse-fallback: play_id -> (delivery_s, note)  for clips un-measurable even after cap
REUSE = {
    # Gausman 2025 clip un-measurable (camera follow); reuse 20-pitcher SB pool median
    "04855f81-6b86-3b37-a53a-c5df4362065d": (1.0932, "reused: Gausman pool median (cv_pilot/pitcher_delivery_sb.csv)"),
    # Hendricks 8f12e2af un-measurable; reuse his OTHER 2025 clip (718944e7) measured 0.9322
    "8f12e2af-0a21-3542-8ce5-1bf7860cf0a3": (0.9322, "reused: Hendricks 2025 same-pitcher clip 718944e7"),
}


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def truthy(x):
    return str(x).strip().lower() == "true"


def main():
    pilot = {r["play_id"]: r for r in load(PILOT)}
    leads = {r["play_id"]: r for r in load(LEADS)}

    rec = {}
    for c in RECOVERED:
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
        elif play_id in REUSE:
            delivery_s, note = REUSE[play_id]
            usable = True; release_conf = ""; method = "reused"
        else:
            # no measurement and no fallback -> keep the (failed) full-pass number, mark unusable
            delivery_s = p.get("delivery_s", ""); usable = False
            release_conf = p.get("release_kpt_conf", ""); method = "full"
            note = "un-measurable; no same-pitcher fallback available"

        out_rows.append({
            "date": L["date"],
            "pitcher_id": L["pitcher_id"],
            "pitcher_name": L["pitcher_name"],
            "p_throws": p.get("p_throws", r.get("p_throws", "") if r else ""),
            "event": event,
            "catcher_name": L["catcher_name"],
            "clip_id": p.get("clip_id", r.get("clip_id", "") if r else ""),
            "play_id": play_id,
            "fps": p.get("fps", ""),
            "delivery_s": delivery_s,
            "usable": usable,
            "release_conf": release_conf,
            "analysis_method": method,
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
