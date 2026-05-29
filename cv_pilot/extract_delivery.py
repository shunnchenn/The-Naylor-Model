#!/usr/bin/env python3
"""
extract_delivery.py  —  Pitch-delivery detector for Statcast base-stealing clips
================================================================================

Reads broadcast mp4 clips of stolen-base attempts and measures, per clip:

        delivery_s = ball_release_time  -  first_movement_time

where first movement = the pitcher initiating the delivery (from the stretch /
slide-step) and ball release = the ball leaving the hand.  Outputs a per-clip
results row plus an annotated QA video so each measurement can be eyeballed.

Pipeline (see cv_pilot plan):
  A  read frames + true container fps
  B  scene-cut guard       (analyse only the opening pitching shot)
  C  YOLOv8-pose + pitcher selection/tracking  (crowded MLB frames)
  D  "set" window          (low-motion plateau before delivery)
  E  first movement        (motion-energy onset; hand-break / leg-lift cross-check)
  F  ball release          (peak throwing-hand speed + reach, parabolic sub-frame)
  G  delivery time + confidence flags
  H  QA overlay video

Usage:
    python3 cv_pilot/extract_delivery.py
    python3 cv_pilot/extract_delivery.py --clip 0001_glasnow_R.mp4
    python3 cv_pilot/extract_delivery.py --no-qa     # skip annotated videos (faster)

Input:
    cv_pilot/clips/*.mp4         hand-dropped pilot clips
    cv_pilot/clips_meta.csv      clip_id,pitcher_name,p_throws[,pitcher_id,bbox_hint]
                                 bbox_hint optional: "x0 y0 x1 y1" (early-frame pitcher box)

Output:
    cv_pilot/pilot_results.csv
    cv_pilot/qa/<clip_id>_annotated.mp4
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    sys.exit("opencv-python not installed.  pip install -r cv_pilot/requirements.txt")

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
CLIPS_DIR   = ROOT / "clips"
QA_DIR      = ROOT / "qa"
META_PATH   = ROOT / "clips_meta.csv"
RESULTS_CSV = ROOT / "pilot_results.csv"
YOLO_WEIGHTS = "yolov8m-pose.pt"        # downloaded on first run

# COCO-17 keypoint indices (YOLOv8-pose)
NOSE = 0
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

KPT_CONF_MIN   = 0.30      # ignore keypoints below this confidence
SCENE_CUT_CORR = 0.55      # histogram-correlation below this = scene cut
PLAUSIBLE_BAND = (0.80, 1.80)   # delivery_s sanity band (first-move -> release)

# Pose-skeleton edges for the QA overlay
SKELETON = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def throwing_wrist(p_throws: str) -> int:
    return R_WRI if str(p_throws).upper().startswith("R") else L_WRI

def throwing_elbow(p_throws: str) -> int:
    return R_ELB if str(p_throws).upper().startswith("R") else L_ELB

def lead_ankle(p_throws: str) -> int:
    # RHP strides with the LEFT leg (lead), LHP with the RIGHT leg.
    return L_ANK if str(p_throws).upper().startswith("R") else R_ANK


def iou(b1, b2) -> float:
    ax0, ay0, ax1, ay1 = b1
    bx0, by0, bx1, by1 = b2
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (ax1 - ax0) * (ay1 - ay0)
    a2 = (bx1 - bx0) * (by1 - by0)
    return inter / (a1 + a2 - inter + 1e-9)


def parabolic_peak(y_prev: float, y0: float, y_next: float) -> float:
    """Sub-frame offset (in [-0.5, 0.5]) of a peak given 3 samples around it."""
    denom = (y_prev - 2.0 * y0 + y_next)
    if abs(denom) < 1e-9:
        return 0.0
    off = 0.5 * (y_prev - y_next) / denom
    return float(np.clip(off, -0.5, 0.5))


def interp_crossing(idx_below: int, v_below: float, v_above: float, thresh: float) -> float:
    """Linear sub-frame frame index where a rising signal crosses `thresh`."""
    if v_above == v_below:
        return float(idx_below)
    frac = (thresh - v_below) / (v_above - v_below)
    return idx_below + float(np.clip(frac, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# B — scene-cut guard
# ─────────────────────────────────────────────────────────────────────────────
def first_scene_cut(frames: list[np.ndarray]) -> int:
    """Return the frame index of the first scene cut, or len(frames) if none."""
    prev_hist = None
    for i, f in enumerate(frames):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if corr < SCENE_CUT_CORR and i > 3:   # ignore the very first frames
                return i
        prev_hist = hist
    return len(frames)


# ─────────────────────────────────────────────────────────────────────────────
# C — YOLOv8-pose detection + pitcher selection / tracking
# ─────────────────────────────────────────────────────────────────────────────
def run_pose(frames: list[np.ndarray], model):
    """Per-frame multi-person pose with ByteTrack IDs.

    Returns dict: track_id -> list of (frame_idx, bbox, kpts[17,3]).
    kpts columns = (x, y, conf).
    """
    tracks: dict[int, list] = {}
    for fi, frame in enumerate(frames):
        res = model.track(frame, persist=True, verbose=False,
                          classes=[0], tracker="bytetrack.yaml")[0]
        if res.boxes is None or res.boxes.id is None or res.keypoints is None:
            continue
        ids   = res.boxes.id.cpu().numpy().astype(int)
        boxes = res.boxes.xyxy.cpu().numpy()
        kxy   = res.keypoints.xy.cpu().numpy()              # (n,17,2)
        kcf   = res.keypoints.conf
        kcf   = (kcf.cpu().numpy() if kcf is not None
                 else np.ones(kxy.shape[:2], dtype=float))
        for tid, box, xy, cf in zip(ids, boxes, kxy, kcf):
            kpts = np.concatenate([xy, cf[:, None]], axis=1)  # (17,3)
            tracks.setdefault(int(tid), []).append((fi, box, kpts))
    return tracks


def track_motion_energy(seq, diag_ref: float) -> float:
    """Total scale-normalised keypoint motion across a track's frames."""
    total = 0.0
    for (_, _, k0), (_, _, k1) in zip(seq[:-1], seq[1:]):
        good = (k0[:, 2] > KPT_CONF_MIN) & (k1[:, 2] > KPT_CONF_MIN)
        if not good.any():
            continue
        d = np.linalg.norm(k1[good, :2] - k0[good, :2], axis=1)
        total += float(d.mean())
    return total / max(diag_ref, 1.0)


def select_pitcher(tracks, frame_w, frame_h, bbox_hint=None):
    """Pick the pitcher track by center-x proximity + delivery motion energy.

    If bbox_hint is given, pick the track whose early bbox best matches it.
    Returns (track_id, seq) or (None, None).
    """
    if not tracks:
        return None, None
    diag = float(np.hypot(frame_w, frame_h))

    if bbox_hint is not None:
        best, best_iou = None, 0.0
        for tid, seq in tracks.items():
            early = [b for (_, b, _) in seq[:8]]
            if not early:
                continue
            m = max(iou(b, bbox_hint) for b in early)
            if m > best_iou:
                best, best_iou = tid, m
        if best is not None and best_iou > 0.1:
            return best, tracks[best]

    cx = frame_w / 2.0
    best, best_score = None, -1e18
    for tid, seq in tracks.items():
        if len(seq) < 4:
            continue
        boxes = np.array([b for (_, b, _) in seq])
        bcx = ((boxes[:, 0] + boxes[:, 2]) / 2.0).mean()
        center_prox = 1.0 - abs(bcx - cx) / (frame_w / 2.0)   # 1=center, 0=edge
        energy = track_motion_energy(seq, diag)
        score = 1.0 * center_prox + 2.5 * energy              # energy dominates
        if score > best_score:
            best, best_score = tid, score
    return best, (tracks[best] if best is not None else None)


# ─────────────────────────────────────────────────────────────────────────────
# Build dense per-frame keypoint arrays for the chosen pitcher
# ─────────────────────────────────────────────────────────────────────────────
def densify(seq, n_frames):
    """seq -> (kpts[n_frames,17,3], boxes[n_frames,4]) with NaN gaps filled."""
    kpts  = np.full((n_frames, 17, 3), np.nan)
    boxes = np.full((n_frames, 4), np.nan)
    for (fi, box, k) in seq:
        kpts[fi]  = k
        boxes[fi] = box
    # forward/back-fill bbox for overlay continuity
    for c in range(4):
        s = pd.Series(boxes[:, c]).ffill().bfill()
        boxes[:, c] = s.values
    return kpts, boxes


def smooth(x, win=3):
    """Light centered moving average ignoring NaN."""
    s = pd.Series(x)
    return s.rolling(win, center=True, min_periods=1).mean().values


# ─────────────────────────────────────────────────────────────────────────────
# D/E/F — set window, first movement, release
# ─────────────────────────────────────────────────────────────────────────────
def motion_energy_series(kpts, diag_ref):
    """Per-frame scale-normalised mean keypoint displacement."""
    n = len(kpts)
    me = np.zeros(n)
    for i in range(1, n):
        k0, k1 = kpts[i - 1], kpts[i]
        good = (k0[:, 2] > KPT_CONF_MIN) & (k1[:, 2] > KPT_CONF_MIN)
        if not good.any():
            me[i] = np.nan
            continue
        d = np.linalg.norm(k1[good, :2] - k0[good, :2], axis=1)
        me[i] = d.mean() / max(diag_ref, 1.0)
    me = pd.Series(me).interpolate().bfill().ffill().values
    return smooth(me, 3)


def find_set_and_first_move(me):
    """Return (set_end_idx, first_move_subframe, floor, std)."""
    n = len(me)
    # Peak motion = somewhere in the delivery; search the set BEFORE it.
    peak = int(np.nanargmax(me))
    search_hi = max(peak, 5)
    # Rolling std to find the quietest plateau before the peak
    win = 5
    best_i, best_var = 0, 1e18
    for i in range(0, max(1, search_hi - win)):
        seg = me[i:i + win]
        v = float(np.nanstd(seg))
        if v < best_var:
            best_var, best_i = v, i
    set_lo = best_i
    set_hi = min(best_i + win, search_hi)
    floor = float(np.nanmean(me[set_lo:set_hi]))
    std   = float(np.nanstd(me[set_lo:set_hi])) + 1e-6
    thresh = floor + 4.0 * std

    # First sustained crossing after the set plateau
    fm = None
    for i in range(set_hi, peak + 1):
        if me[i] > thresh and np.all(me[i:min(i + 3, n)] > floor + 2.0 * std):
            fm = interp_crossing(i - 1, me[i - 1], me[i], thresh)
            break
    if fm is None:
        fm = float(set_hi)
    return set_hi, fm, floor, std


def leg_lift_frame(kpts, p_throws, set_end, peak):
    """Cross-check: frame where lead-ankle starts rising (image y decreases)."""
    a = lead_ankle(p_throws)
    y = kpts[:, a, 1].copy()
    c = kpts[:, a, 2]
    y[c < KPT_CONF_MIN] = np.nan
    y = pd.Series(y).interpolate().bfill().ffill().values
    base = np.nanmean(y[max(0, set_end - 4):set_end + 1])
    diag_lo, diag_hi = set_end, max(set_end + 1, peak)
    for i in range(diag_lo, diag_hi):
        if base - y[i] > 0.03 * abs(base):   # ankle risen ~3% of its height
            return float(i)
    return None


def hand_break_frame(kpts, set_end, peak):
    """Cross-check: frame where the two wrists start separating."""
    lw, rw = kpts[:, L_WRI, :2], kpts[:, R_WRI, :2]
    cf = np.minimum(kpts[:, L_WRI, 2], kpts[:, R_WRI, 2])
    sep = np.linalg.norm(lw - rw, axis=1)
    sep[cf < KPT_CONF_MIN] = np.nan
    sep = pd.Series(sep).interpolate().bfill().ffill().values
    base = np.nanmean(sep[max(0, set_end - 4):set_end + 1]) + 1e-6
    for i in range(set_end, max(set_end + 1, peak)):
        if sep[i] > 1.5 * base:
            return float(i)
    return None


def find_release(kpts, p_throws, first_move_idx, diag_ref):
    """Release ≈ peak throwing-hand speed × reach, parabolic sub-frame refined."""
    w = throwing_wrist(p_throws)
    sho = R_SHO if w == R_WRI else L_SHO
    hip = R_HIP if w == R_WRI else L_HIP

    wx = pd.Series(np.where(kpts[:, w, 2] > KPT_CONF_MIN, kpts[:, w, 0], np.nan))
    wy = pd.Series(np.where(kpts[:, w, 2] > KPT_CONF_MIN, kpts[:, w, 1], np.nan))
    wx = wx.interpolate().bfill().ffill().values
    wy = wy.interpolate().bfill().ffill().values

    # hand speed (per-frame displacement), scale-normalised
    speed = np.zeros(len(wx))
    speed[1:] = np.hypot(np.diff(wx), np.diff(wy)) / max(diag_ref, 1.0)
    speed = smooth(speed, 3)

    # reach: wrist distance from shoulder-hip body centroid
    bx = np.nanmean([kpts[:, sho, 0], kpts[:, hip, 0]], axis=0)
    by = np.nanmean([kpts[:, sho, 1], kpts[:, hip, 1]], axis=0)
    bx = pd.Series(bx).interpolate().bfill().ffill().values
    by = pd.Series(by).interpolate().bfill().ffill().values
    reach = np.hypot(wx - bx, wy - by) / max(diag_ref, 1.0)
    reach = smooth(reach, 3)

    lo = int(np.ceil(first_move_idx))
    n = len(speed)
    if lo >= n - 1:
        return float(n - 1), 0.0
    # combine speed (dominant) with reach; release is near max hand speed
    combo = speed.copy()
    combo[:lo] = -1e9
    pk = int(np.nanargmax(combo))
    conf = float(kpts[pk, w, 2]) if not np.isnan(kpts[pk, w, 2]) else 0.0
    # sub-frame parabolic refine on the speed curve
    if 0 < pk < n - 1:
        off = parabolic_peak(speed[pk - 1], speed[pk], speed[pk + 1])
    else:
        off = 0.0
    return pk + off, conf


# ─────────────────────────────────────────────────────────────────────────────
# H — QA overlay video
# ─────────────────────────────────────────────────────────────────────────────
def write_qa(frames, fps, kpts, boxes, set_i, fm, rel, delivery_s, out_path):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    fm_i, rel_i = int(round(fm)), int(round(rel))
    for i, frame in enumerate(frames):
        f = frame.copy()
        k = kpts[i]
        for a, b in SKELETON:
            if k[a, 2] > KPT_CONF_MIN and k[b, 2] > KPT_CONF_MIN:
                pa = (int(k[a, 0]), int(k[a, 1]))
                pb = (int(k[b, 0]), int(k[b, 1]))
                cv2.line(f, pa, pb, (0, 255, 0), 2)
        for j in range(17):
            if k[j, 2] > KPT_CONF_MIN:
                cv2.circle(f, (int(k[j, 0]), int(k[j, 1])), 3, (0, 200, 255), -1)
        if not np.isnan(boxes[i]).any():
            x0, y0, x1, y1 = boxes[i].astype(int)
            cv2.rectangle(f, (x0, y0), (x1, y1), (255, 180, 0), 1)
        label = ""
        color = (255, 255, 255)
        if i == set_i:   label, color = "SET",         (200, 200, 200)
        if i == fm_i:    label, color = "FIRST MOVE",  (0, 255, 255)
        if i == rel_i:   label, color = "RELEASE",     (0, 0, 255)
        if label:
            cv2.rectangle(f, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(f, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)
        cv2.putText(f, f"f{i}  delivery={delivery_s:.3f}s",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        vw.write(f)
    vw.release()


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip driver
# ─────────────────────────────────────────────────────────────────────────────
def parse_hint(s):
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        vals = [float(v) for v in s.replace(",", " ").split()]
        return vals if len(vals) == 4 else None
    except ValueError:
        return None


def process_clip(path: Path, p_throws: str, model, bbox_hint=None, make_qa=True):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    row = {"clip_id": path.name, "p_throws": p_throws, "fps": round(fps, 3),
           "n_frames": len(frames), "status": "ok"}
    if len(frames) < 8:
        row["status"] = "too_short"
        return row

    cut = first_scene_cut(frames)
    seg = frames[:cut]
    row["analysis_frames"] = len(seg)
    row["scene_cut_at"] = cut if cut < len(frames) else -1

    h, w = seg[0].shape[:2]
    diag = float(np.hypot(w, h))

    tracks = run_pose(seg, model)
    tid, pseq = select_pitcher(tracks, w, h, bbox_hint)
    if pseq is None:
        row["status"] = "no_pitcher"
        return row
    row["pitcher_track_id"] = tid

    kpts, boxes = densify(pseq, len(seg))
    me = motion_energy_series(kpts, diag)
    peak = int(np.nanargmax(me))

    set_i, fm, floor, std = find_set_and_first_move(me)
    # cross-checks — take the earliest consistent first-move cue
    cands = [fm]
    ll = leg_lift_frame(kpts, p_throws, set_i, peak)
    hb = hand_break_frame(kpts, set_i, peak)
    for c in (ll, hb):
        if c is not None and c >= set_i:
            cands.append(c)
    fm = float(min(cands))

    rel, rel_conf = find_release(kpts, p_throws, fm, diag)

    delivery_s = (rel - fm) / fps
    row.update({
        "set_frame": set_i,
        "first_move_frame": round(fm, 2),
        "release_frame": round(rel, 2),
        "delivery_s": round(delivery_s, 4),
        "release_kpt_conf": round(rel_conf, 3),
        "first_move_cue_n": len(cands),
    })

    # confidence flags
    in_band = PLAUSIBLE_BAND[0] <= delivery_s <= PLAUSIBLE_BAND[1]
    pre_cut = (cut == len(frames)) or (int(round(rel)) < cut)
    good_conf = rel_conf >= KPT_CONF_MIN
    row["in_band"] = bool(in_band)
    row["release_pre_cut"] = bool(pre_cut)
    row["usable"] = bool(in_band and pre_cut and good_conf)
    row["confidence"] = round(
        (0.5 * good_conf + 0.3 * in_band + 0.2 * pre_cut), 3)

    if make_qa:
        QA_DIR.mkdir(parents=True, exist_ok=True)
        out = QA_DIR / f"{path.stem}_annotated.mp4"
        try:
            write_qa(seg, fps, kpts, boxes, set_i, fm, rel, delivery_s, out)
            row["qa_video"] = out.name
        except Exception as e:               # QA is non-critical
            row["qa_video"] = f"ERR:{e}"
    return row


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def load_meta():
    if META_PATH.exists():
        return pd.read_csv(META_PATH)
    return pd.DataFrame(columns=["clip_id", "pitcher_name", "p_throws",
                                 "pitcher_id", "bbox_hint"])


def main():
    ap = argparse.ArgumentParser(description="Pitch-delivery CV detector")
    ap.add_argument("--clip", help="process a single clip filename in clips/")
    ap.add_argument("--no-qa", action="store_true", help="skip QA videos")
    ap.add_argument("--weights", default=YOLO_WEIGHTS)
    args = ap.parse_args()

    if not CLIPS_DIR.exists() or not any(CLIPS_DIR.glob("*.mp4")):
        sys.exit(f"No clips found in {CLIPS_DIR}.  Drop pilot mp4s there first.")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed.  pip install -r cv_pilot/requirements.txt")

    print(f"[load] YOLOv8-pose weights: {args.weights}")
    model = YOLO(args.weights)

    meta = load_meta()
    meta_map = {r["clip_id"]: r for _, r in meta.iterrows()}

    clips = ([CLIPS_DIR / args.clip] if args.clip
             else sorted(CLIPS_DIR.glob("*.mp4")))

    rows = []
    for path in clips:
        if not path.exists():
            print(f"  ! missing {path.name}")
            continue
        m = meta_map.get(path.name, {})
        # infer p_throws from filename suffix (_R / _L) if meta missing
        p_throws = m.get("p_throws")
        if not isinstance(p_throws, str) or not p_throws:
            stem = path.stem.upper()
            p_throws = "L" if stem.endswith("_L") else "R"
        hint = parse_hint(m.get("bbox_hint")) if m is not None else None
        print(f"[clip] {path.name}  (p_throws={p_throws})")
        try:
            row = process_clip(path, p_throws, model, hint, make_qa=not args.no_qa)
        except Exception as e:
            row = {"clip_id": path.name, "status": f"ERR:{e}"}
        if "pitcher_name" in m:
            row["pitcher_name"] = m["pitcher_name"]
        if "pitcher_id" in m:
            row["pitcher_id"] = m["pitcher_id"]
        print(f"        -> {row.get('status')}  "
              f"delivery_s={row.get('delivery_s')}  usable={row.get('usable')}")
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["clip_id", "pitcher_name", "pitcher_id", "p_throws", "fps", "n_frames",
            "analysis_frames", "scene_cut_at", "set_frame", "first_move_frame",
            "release_frame", "delivery_s", "release_kpt_conf", "first_move_cue_n",
            "in_band", "release_pre_cut", "usable", "confidence", "qa_video",
            "pitcher_track_id", "status"]
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[write] {RESULTS_CSV}  ({len(df)} clips)")
    if "usable" in df.columns:
        print(f"        usable: {int(df['usable'].sum())}/{len(df)}")


if __name__ == "__main__":
    main()
