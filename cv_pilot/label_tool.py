#!/usr/bin/env python3
"""
label_tool.py  —  Manual frame labeler for ground-truth delivery times
======================================================================

Step through a clip frame-by-frame and mark the FIRST-MOVEMENT and
BALL-RELEASE frames by eye.  These labels are the ground truth that
evaluate.py scores the automatic detector against.

Usage:
    python3 cv_pilot/label_tool.py                 # label every clip in clips/
    python3 cv_pilot/label_tool.py --clip 0001.mp4 # label one clip

Keys (focus the video window):
    d / RIGHT   next frame            a / LEFT   previous frame
    e           jump +10 frames       q          jump -10 frames
    f           mark FIRST MOVEMENT (current frame)
    r           mark RELEASE         (current frame)
    s           save this clip's labels and go to next
    x           skip this clip (no label)
    ESC         quit the tool

Writes / updates:
    cv_pilot/labels_manual.csv
        clip_id, fps, manual_first_move_frame, manual_release_frame,
        manual_delivery_s, notes
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import cv2
except ImportError:
    sys.exit("opencv-python not installed.  pip install -r cv_pilot/requirements.txt")

ROOT       = Path(__file__).resolve().parent
CLIPS_DIR  = ROOT / "clips"
LABELS_CSV = ROOT / "labels_manual.csv"


def load_labels() -> dict:
    if LABELS_CSV.exists():
        df = pd.read_csv(LABELS_CSV)
        return {r["clip_id"]: r.to_dict() for _, r in df.iterrows()}
    return {}


def save_labels(labels: dict):
    df = pd.DataFrame(list(labels.values()))
    cols = ["clip_id", "fps", "manual_first_move_frame",
            "manual_release_frame", "manual_delivery_s", "notes"]
    df = df.reindex(columns=cols)
    df.to_csv(LABELS_CSV, index=False)
    print(f"[write] {LABELS_CSV}  ({len(df)} clips labeled)")


def label_clip(path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        print(f"  ! could not read {path.name}")
        return None

    n = len(frames)
    i = 0
    fm = None
    rel = None
    win = f"label: {path.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"\n[{path.name}] {n} frames @ {fps:.2f} fps  "
          f"(f=first-move  r=release  s=save  x=skip  ESC=quit)")

    while True:
        f = frames[i].copy()
        h, w = f.shape[:2]
        cv2.rectangle(f, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(f, f"frame {i}/{n-1}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        txt = f"FM={fm}  REL={rel}"
        cv2.putText(f, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.imshow(win, f)

        k = cv2.waitKey(0) & 0xFF
        if k in (ord('d'), 83):            # next
            i = min(n - 1, i + 1)
        elif k in (ord('a'), 81):          # prev
            i = max(0, i - 1)
        elif k == ord('e'):                # +10
            i = min(n - 1, i + 10)
        elif k == ord('q'):                # -10
            i = max(0, i - 10)
        elif k == ord('f'):
            fm = i; print(f"   first-move = {i}")
        elif k == ord('r'):
            rel = i; print(f"   release    = {i}")
        elif k == ord('s'):
            cv2.destroyWindow(win)
            delivery = (rel - fm) / fps if (fm is not None and rel is not None) else None
            return {"clip_id": path.name, "fps": round(fps, 3),
                    "manual_first_move_frame": fm,
                    "manual_release_frame": rel,
                    "manual_delivery_s": round(delivery, 4) if delivery else None,
                    "notes": ""}
        elif k == ord('x'):
            cv2.destroyWindow(win)
            return None
        elif k == 27:                      # ESC
            cv2.destroyWindow(win)
            raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description="Manual delivery-frame labeler")
    ap.add_argument("--clip", help="label a single clip filename")
    args = ap.parse_args()

    if not CLIPS_DIR.exists() or not any(CLIPS_DIR.glob("*.mp4")):
        sys.exit(f"No clips in {CLIPS_DIR}.")

    labels = load_labels()
    clips = ([CLIPS_DIR / args.clip] if args.clip
             else sorted(CLIPS_DIR.glob("*.mp4")))

    try:
        for path in clips:
            if not path.exists():
                print(f"  ! missing {path.name}")
                continue
            res = label_clip(path)
            if res is not None:
                labels[res["clip_id"]] = res
                save_labels(labels)
    except KeyboardInterrupt:
        print("\n[quit] saving what we have …")
    finally:
        if labels:
            save_labels(labels)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
