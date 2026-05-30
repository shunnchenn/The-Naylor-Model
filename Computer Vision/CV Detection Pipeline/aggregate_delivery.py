#!/usr/bin/env python3
"""
aggregate_delivery.py  —  per-pitcher delivery time under stealing conditions
==============================================================================

Rolls the per-clip detector output (pilot_results.csv) up to a per-pitcher
delivery-time signature.  Every clip here is a real base-stealing pitch (the
runner broke on that delivery), so this is the per-pitcher mean *under stealing
conditions* — exactly what the Naylor model wants in place of the
LEAGUE_PITCHER_TTP = 1.30 constant.

Only `usable=True` clips feed the average (a clip is unusable when YOLO never
tracked the throwing hand, so its release is unreliable — see extract_delivery).

Output: cv_pilot/pitcher_delivery_sb.csv
    pitcher_id, pitcher_name, p_throws, n_clips, n_usable,
    mean_delivery_s, median_delivery_s, std_delivery_s, delivery_times
The keys (pitcher_id) join straight back to the Naylor model's DF_Attempts.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT        = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "pilot_results.csv"
OUT_CSV     = ROOT / "pitcher_delivery_sb.csv"


def main():
    if not RESULTS_CSV.exists():
        raise SystemExit(f"{RESULTS_CSV} not found — run extract_delivery.py first")
    df = pd.read_csv(RESULTS_CSV)

    # need a pitcher key; fall back to a name parsed from clip_id if absent
    if "pitcher_id" not in df.columns:
        df["pitcher_id"] = pd.NA
    df["pitcher_key"] = df["pitcher_id"]
    miss = df["pitcher_key"].isna()
    if "pitcher_name" in df.columns:
        df.loc[miss, "pitcher_key"] = df.loc[miss, "pitcher_name"]

    usable = df[df.get("usable", False) == True].copy()      # noqa: E712

    rows = []
    for key, g in df.groupby("pitcher_key", dropna=False):
        gu = usable[usable["pitcher_key"] == key]
        times = gu["delivery_s"].dropna().tolist()
        name = (g["pitcher_name"].dropna().iloc[0]
                if "pitcher_name" in g and g["pitcher_name"].notna().any() else "")
        arm = (g["p_throws"].dropna().iloc[0]
               if "p_throws" in g and g["p_throws"].notna().any() else "")
        pid = (g["pitcher_id"].dropna().iloc[0]
               if "pitcher_id" in g and g["pitcher_id"].notna().any() else "")
        s = pd.Series(times)
        rows.append({
            "pitcher_id": pid,
            "pitcher_name": name,
            "p_throws": arm,
            "n_clips": int(len(g)),
            "n_usable": int(len(times)),
            "mean_delivery_s": round(s.mean(), 4) if len(times) else None,
            "median_delivery_s": round(s.median(), 4) if len(times) else None,
            "std_delivery_s": round(s.std(ddof=0), 4) if len(times) > 1 else None,
            "delivery_times": ";".join(f"{t:.3f}" for t in sorted(times)),
        })

    out = (pd.DataFrame(rows)
           .sort_values(["n_usable", "pitcher_name"], ascending=[False, True]))
    out.to_csv(OUT_CSV, index=False)

    n_meas = (out["n_usable"] > 0).sum()
    print(f"[write] {OUT_CSV}")
    print(f"   pitchers: {len(out)}  ·  with >=1 usable clip: {n_meas}")
    print(f"   pitchers with >=2 usable (std available): {(out['n_usable'] >= 2).sum()}")
    show = out[out["n_usable"] > 0][
        ["pitcher_name", "p_throws", "n_usable", "median_delivery_s",
         "std_delivery_s", "delivery_times"]]
    if len(show):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
