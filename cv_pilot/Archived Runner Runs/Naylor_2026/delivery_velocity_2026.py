#!/usr/bin/env python3
"""
Delivery-window runner velocity metric (Josh Naylor, 2026).

Metric:
    avg_velocity_ftps = gain_to_release_ft / delivery_s

where
    gain_to_release_ft = lead_at_release_ft - lead_at_firstmove_ft   (Statcast lead data)
    delivery_s         = CV-measured first-move -> ball-release time  (cv_pilot detector)

This is the average ground speed (ft/s) the runner builds during the window the
pitcher is committed to the plate (first move -> release). It blends the runner's
secondary-lead burst with how much time the pitcher's delivery gives him. A high
value means the runner is covering a lot of ground in the exact window where the
battery can no longer hold him -- the cleanest single-number "free distance" proxy
we can build from the clips we have.

Inputs (same folder):
    naylor2026_leads.csv          Statcast lead distances + outcome/run value
    pitcher_delivery_2026.csv     CV per-clip delivery time + confidence

Output:
    delivery_velocity_2026.csv    joined rows + avg_velocity_ftps
Prints a performance read-out to stdout.
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LEADS = os.path.join(HERE, "naylor2026_leads.csv")
DELIV = os.path.join(HERE, "pitcher_delivery_2026.csv")
OUT = os.path.join(HERE, "delivery_velocity_2026.csv")


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    leads = load_csv(LEADS)
    deliv = load_csv(DELIV)

    # key on (date, pitcher_name): unique per play in this sample
    dkey = {(r["date"], r["pitcher_name"]): r for r in deliv}

    rows = []
    for L in leads:
        d = dkey.get((L["date"], L["pitcher_name"]))
        if d is None:
            continue
        delivery_s = fnum(d.get("delivery_s"))
        usable = (d.get("usable", "").strip().lower() == "true")
        gain = fnum(L.get("gain_to_release_ft"))
        avg_vel = None
        if usable and delivery_s and delivery_s > 0 and gain is not None:
            avg_vel = round(gain / delivery_s, 3)
        rows.append({
            "date": L["date"],
            "pitcher_name": L["pitcher_name"],
            "p_throws": d.get("p_throws", ""),
            "base": L["base"],
            "result": L["result"],
            "run_value": L["run_value"],
            "lead_at_firstmove_ft": L["lead_at_firstmove_ft"],
            "lead_at_release_ft": L["lead_at_release_ft"],
            "gain_to_release_ft": L["gain_to_release_ft"],
            "delivery_s": d.get("delivery_s", ""),
            "release_conf": d.get("release_conf", ""),
            "usable": d.get("usable", ""),
            "analysis_method": d.get("analysis_method", ""),
            "avg_velocity_ftps": "" if avg_vel is None else avg_vel,
        })

    cols = ["date", "pitcher_name", "p_throws", "base", "result", "run_value",
            "lead_at_firstmove_ft", "lead_at_release_ft", "gain_to_release_ft",
            "delivery_s", "release_conf", "usable", "analysis_method",
            "avg_velocity_ftps"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---------- performance read-out ----------
    usable_rows = [r for r in rows if r["avg_velocity_ftps"] != ""]
    vals = [float(r["avg_velocity_ftps"]) for r in usable_rows]

    print("=" * 74)
    print("DELIVERY-WINDOW RUNNER VELOCITY  (Josh Naylor, 2026)")
    print("  avg_velocity_ftps = gain_to_release_ft / delivery_s")
    print("=" * 74)
    hdr = f"{'date':10} {'pitcher':20} {'res':4} {'gain':>5} {'deliv':>6} {'vel':>7} {'conf':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["avg_velocity_ftps"] == "",
                                         -(float(x["avg_velocity_ftps"]) if x["avg_velocity_ftps"] != "" else 0))):
        vel = r["avg_velocity_ftps"]
        vel_s = f"{float(vel):7.2f}" if vel != "" else "    n/a"
        dv = r["delivery_s"] or "n/a"
        print(f"{r['date']:10} {r['pitcher_name']:20} {r['result']:4} "
              f"{float(r['gain_to_release_ft']):5.1f} {str(dv):>6} {vel_s} "
              f"{r['release_conf']:>5}")

    print("-" * len(hdr))
    print(f"usable plays: {len(usable_rows)}/{len(rows)}  "
          f"(excluded: {[r['pitcher_name'] for r in rows if r['avg_velocity_ftps']=='']})")
    if vals:
        print(f"velocity ft/s  mean={st.mean(vals):.2f}  median={st.median(vals):.2f}  "
              f"min={min(vals):.2f}  max={max(vals):.2f}  "
              f"std={st.pstdev(vals):.2f}")

    # SB vs CS split (only 1 CS in sample -> illustrative)
    sb = [float(r["avg_velocity_ftps"]) for r in usable_rows if r["result"] == "SB"]
    cs = [float(r["avg_velocity_ftps"]) for r in usable_rows if r["result"] == "CS"]
    print()
    if sb:
        print(f"  SB (n={len(sb)})  mean vel = {st.mean(sb):.2f} ft/s")
    if cs:
        print(f"  CS (n={len(cs)})  mean vel = {st.mean(cs):.2f} ft/s   "
              f"<- single sample (Joe Ryan); illustrative only")

    # correlation of velocity with delivery_s and with lead_at_firstmove
    def corr(xs, ys):
        if len(xs) < 3:
            return None
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        dx = sum((a - mx) ** 2 for a in xs) ** 0.5
        dy = sum((b - my) ** 2 for b in ys) ** 0.5
        return num / (dx * dy) if dx and dy else None

    dlv = [float(r["delivery_s"]) for r in usable_rows]
    gns = [float(r["gain_to_release_ft"]) for r in usable_rows]
    lfm = [float(r["lead_at_firstmove_ft"]) for r in usable_rows]
    print()
    c1 = corr(vals, dlv)
    c2 = corr(vals, gns)
    c3 = corr(vals, lfm)
    if c1 is not None:
        print(f"  corr(velocity, delivery_s)          = {c1:+.2f}  "
              f"(neg = slower deliveries give LOWER ft/s for same gain)")
    if c2 is not None:
        print(f"  corr(velocity, gain_to_release_ft)  = {c2:+.2f}  "
              f"(pos = more ground gained drives the metric)")
    if c3 is not None:
        print(f"  corr(velocity, lead_at_firstmove)   = {c3:+.2f}")
    print("=" * 74)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
