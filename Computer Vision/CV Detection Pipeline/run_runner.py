#!/usr/bin/env python3
"""
run_runner.py  —  One-command per-runner pipeline (leads -> clips -> delivery)
=============================================================================

Takes a single (runner_id, name_tag, year) — typically a row that
discover_runners.py surfaced — and runs the whole hands-off chain that used to
be manual hovering + clicking + per-folder scripting:

    1. fetch_leads.py      runner_id year  -> <dir>/<tag><year>_leads.csv
                                              <dir>/<tag><year>_targets.csv
    2. fetch_clips.py      --targets ...    -> <dir>/clips/*.mp4 + clips_meta.csv
    3. extract_delivery.py --clips-dir ...  -> <dir>/pilot_results.csv  (heel default)

Then the existing delivery_velocity.py / statcast_ref.py can be pointed at <dir>.

Usage
-----
  python3 cv_pilot/run_runner.py 647304 naylor 2025
  python3 cv_pilot/run_runner.py 665742 soto 2025 --dir cv_pilot/Soto_2025 --no-detect
  python3 cv_pilot/run_runner.py 592518 machado 2025 --dry-run   # print the plan only

Network stages (leads, clips) need dangerouslyDisableSandbox.  The detector stage
is CPU/MPS-heavy (~tens of seconds per clip); pass --no-detect to stop after the
clips download and run extract_delivery.py yourself later.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str], dry: bool) -> int:
    print("   $", " ".join(str(c) for c in cmd))
    if dry:
        return 0
    return subprocess.run(cmd, check=False).returncode


def main():
    ap = argparse.ArgumentParser(description="Per-runner leads->clips->delivery pipeline")
    ap.add_argument("runner_id", type=int)
    ap.add_argument("name_tag", help="lowercase tag for file/clip naming, e.g. 'naylor'")
    ap.add_argument("year", type=int)
    ap.add_argument("--dir", default=None, help="output dir (default cv_pilot/<Tag>_<year>)")
    ap.add_argument("--no-clips", action="store_true", help="stop after leads")
    ap.add_argument("--no-detect", action="store_true", help="stop after clips (skip detector)")
    ap.add_argument("--first-move", choices=["disp", "heel"], default="heel",
                    help="detector first-move cue (default heel)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = ap.parse_args()

    tag, y = args.name_tag, args.year
    d = Path(args.dir) if args.dir else (HERE / f"{tag.capitalize()}_{y}")
    leads = d / f"{tag}{y}_leads.csv"
    targets = d / f"{tag}{y}_targets.csv"
    py = sys.executable

    print(f"[run] {tag} ({args.runner_id}) {y} -> {d}")

    # 1) leads
    rc = run([py, str(HERE / "fetch_leads.py"), str(args.runner_id), str(y),
              "--runner-name", tag, "--out", str(leads), "--targets-out", str(targets)],
             args.dry_run)
    if rc != 0 and not args.dry_run:
        sys.exit(f"fetch_leads failed (rc={rc})")
    if args.no_clips:
        return

    # 2) clips
    rc = run([py, str(HERE / "fetch_clips.py"), "--targets", str(targets),
              "--out-dir", str(d), "--download"], args.dry_run)
    if rc != 0 and not args.dry_run:
        sys.exit(f"fetch_clips failed (rc={rc})")
    if args.no_detect:
        return

    # 3) detector (heel default)
    rc = run([py, str(HERE / "extract_delivery.py"),
              "--clips-dir", str(d / "clips"), "--meta", str(d / "clips_meta.csv"),
              "--out", str(d / "pilot_results.csv"), "--no-qa",
              "--first-move", args.first_move], args.dry_run)
    if rc != 0 and not args.dry_run:
        sys.exit(f"extract_delivery failed (rc={rc})")

    print(f"[done] {tag} {y}: leads + clips + pilot_results.csv under {d}")


if __name__ == "__main__":
    main()
