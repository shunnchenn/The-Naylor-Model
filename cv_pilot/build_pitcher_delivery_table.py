#!/usr/bin/env python3
"""
Consolidate every CV-measured pitcher delivery time into one lookup table:
    data/DF_PitcherDelivery.csv  ->  pitcher_id, n_clips, median_delivery_s

This is the join table v6_explore.py uses to replace the constant LEAGUE_PITCHER_TTP
(1.30 s) with a measured per-pitcher first-move->release time when we have one.

Sources (all under cv_pilot/):
  - pitcher_delivery_sb.csv               20-pitcher steal-condition pool (raw times)
  - Naylor_2025/pitcher_delivery_2025.csv per-attempt (usable, NOT reused)
  - Naylor_2026/pitcher_delivery_2026.csv per-attempt (usable)
  - Soto_2025/pitcher_delivery_2025.csv   per-attempt (usable, NOT reused)
  - Soto_2026/pitcher_delivery_2026.csv   per-attempt (usable)
  - {Vladdy,Yandy,Torres,Bichette}_2025/pitcher_delivery_2025.csv  per-attempt
  - Naylor_2024  leads + /tmp detector rows (3 attempts)

We pool independent measurements per pitcher_id and take the MEDIAN. Rows flagged
analysis_method == 'reused' are skipped (they are copies of another measurement and
would double-count).
"""
import csv, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "DF_PitcherDelivery.csv")

times = {}   # pitcher_id(int) -> list[float]


def add(pid, val):
    try:
        pid = int(float(pid)); val = float(val)
    except (TypeError, ValueError):
        return
    if val > 0:
        times.setdefault(pid, []).append(val)


# 1) 20-pitcher pool (raw per-clip times in 'delivery_times', ';'-joined)
pool = os.path.join(HERE, "pitcher_delivery_sb.csv")
if os.path.exists(pool):
    for r in csv.DictReader(open(pool)):
        pid = r.get("pitcher_id")
        for t in (r.get("delivery_times") or "").split(";"):
            if t.strip():
                add(pid, t)

# 2) Naylor + Soto per-attempt delivery tables (skip reused)
for rel in ("Naylor_2025/pitcher_delivery_2025.csv",
            "Naylor_2026/pitcher_delivery_2026.csv",
            "Soto_2025/pitcher_delivery_2025.csv",
            "Soto_2026/pitcher_delivery_2026.csv",
            "Vladdy_2025/pitcher_delivery_2025.csv",
            "Yandy_2025/pitcher_delivery_2025.csv",
            "Torres_2025/pitcher_delivery_2025.csv",
            "Bichette_2025/pitcher_delivery_2025.csv"):
    p = os.path.join(HERE, rel)
    if not os.path.exists(p):
        continue
    for r in csv.DictReader(open(p)):
        if str(r.get("usable", "")).strip().lower() != "true":
            continue
        if (r.get("analysis_method") or "") == "reused":
            continue
        add(r.get("pitcher_id"), r.get("delivery_s"))

# 3) Naylor 2024 (leads carry pitcher_id; deliveries in /tmp/d2024_*.csv)
leads24 = os.path.join(HERE, "Naylor_2024", "naylor2024_leads.csv")
if os.path.exists(leads24):
    by_play = {r["play_id"]: r for r in csv.DictReader(open(leads24))}
    for c in ("NAYLOR_d21c5fbb_muller_L", "NAYLOR_977c39d5_sale_L",
              "NAYLOR_ea0116d0_hernandez_R"):
        f = f"/tmp/d2024_{c}.csv"
        if os.path.exists(f):
            d = list(csv.DictReader(open(f)))[0]
            L = by_play.get(d.get("play_id"))
            if L and str(d.get("usable", "")).strip().lower() == "true":
                add(L["pitcher_id"], d.get("delivery_s"))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
rows = []
for pid, vals in sorted(times.items()):
    rows.append({"pitcher_id": pid, "n_clips": len(vals),
                 "median_delivery_s": round(st.median(vals), 4)})
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pitcher_id", "n_clips", "median_delivery_s"])
    w.writeheader(); w.writerows(rows)

allv = [v for vs in times.values() for v in vs]
print(f"[write] {OUT}")
print(f"  {len(rows)} pitchers measured  ({len(allv)} total clips)")
print(f"  median delivery across pitchers = "
      f"{st.median([r['median_delivery_s'] for r in rows]):.3f} s  "
      f"(constant fallback was 1.300)")
