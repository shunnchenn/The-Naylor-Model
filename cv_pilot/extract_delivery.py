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
  B  scene-cut split       (enumerate shots; pick the one with the full delivery,
                           since a steal clip may open on a runner-on-first angle)
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

# Inference knobs (set from CLI in main()).  DEVICE=None lets ultralytics
# auto-select; on Apple-Silicon we resolve it to "mps" so the M2 GPU is used
# instead of CPU (the single biggest speedup).  IMGSZ trades detail for speed.
DEVICE = None
IMGSZ  = 640
HALF   = False
# Optional cap (seconds) on how much of each clip to analyse.  Steal broadcasts
# put the delivery near the start, then pan to follow the runner; that late
# runner-slide motion can outweigh the pitch and pull the detector off-target.
# Capping trims it — faster AND more robust on continuous follow-shots.
# None = analyse the whole clip (preserves the validated pool behaviour).
MAX_ANALYSIS_S = None

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
MIN_SEG_FRAMES = 40        # a contiguous shot shorter than this can't hold a full delivery

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


def scene_segments(frames: list[np.ndarray]) -> list[tuple[int, int]]:
    """Split the clip into contiguous shots at EVERY scene cut.

    Broadcast steal clips sometimes OPEN on a runner-on-first / different
    camera angle and only show the pitcher's full throw in a later shot.  So we
    can't assume the delivery is in the opening segment — we enumerate all shots
    and let the caller pick the one where the pitcher's throw is fully visible.

    Returns a list of (start, end) frame-index pairs (end exclusive).
    """
    prev_hist = None
    cuts = [0]
    for i, f in enumerate(frames):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            # require a gap from the previous cut so a multi-frame dissolve
            # doesn't register as several cuts in a row
            if corr < SCENE_CUT_CORR and (i - cuts[-1]) > 3:
                cuts.append(i)
        prev_hist = hist
    cuts.append(len(frames))
    return [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]


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
                          classes=[0], tracker="bytetrack.yaml",
                          device=DEVICE, imgsz=IMGSZ, half=HALF)[0]
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


def leg_lift_frame(kpts, p_throws, set_end, peak, diag_ref):
    """PRIMARY first-move signal: the lead leg leaves its planted stance.

    Per the project (and user) definition, "first movement" = the lead leg
    starting to move — *upwards OR horizontally* (slide-steps barely lift the
    foot but always shift it).  The old version watched only the lead ankle's
    vertical rise against a hard pixel floor, which missed subtle / horizontal
    loading and fired tens of frames late.

    Fix: track the **total displacement** (√Δx²+Δy²) of BOTH the lead ankle and
    lead knee from a *rolling* baseline (median of the recent planted position,
    so a slow camera pan doesn't count as motion).  First move = the first frame
    after the set where that displacement clears a noise-relative threshold and
    stays elevated.  Scale-normalised by the frame diagonal so the threshold is
    framing-independent.  Returns a sub-frame float index, or None if both
    lead-leg keypoints are too occluded.
    """
    la = lead_ankle(p_throws)
    lk = L_KNE if str(p_throws).upper().startswith("R") else R_KNE
    if (np.isfinite(np.where(kpts[:, la, 2] > KPT_CONF_MIN, kpts[:, la, 1], np.nan)).sum() < 5
            and np.isfinite(np.where(kpts[:, lk, 2] > KPT_CONF_MIN, kpts[:, lk, 1], np.nan)).sum() < 5):
        return None

    def pos(idx):
        x = np.where(kpts[:, idx, 2] > KPT_CONF_MIN, kpts[:, idx, 0], np.nan)
        y = np.where(kpts[:, idx, 2] > KPT_CONF_MIN, kpts[:, idx, 1], np.nan)
        x = pd.Series(x).interpolate().bfill().ffill().values
        y = pd.Series(y).interpolate().bfill().ffill().values
        return x, y

    ax, ay = pos(la)
    kx, ky = pos(lk)
    n = len(ax)

    # FIXED baseline = the planted lead-leg position during the set.  We anchor
    # it on the *quietest* sub-window of the pre-set frames, NOT the raw window
    # edge — otherwise frames where the pitcher is still settling INTO the set
    # contaminate both the baseline position and the noise estimate (observed:
    # a 16 px threshold that hid the real first move).  (A rolling baseline is
    # also wrong here: it tracks an oscillating keypoint and hides the move.)
    w0 = max(0, set_end - 18)
    w1 = max(w0 + 7, set_end + 2)

    def disp_from(px, py, lo, hi):
        bx = float(np.nanmedian(px[lo:hi]))
        by = float(np.nanmedian(py[lo:hi]))
        return np.hypot(px - bx, py - by)

    # pass 1: rough displacement to locate the quietest 7-frame plateau
    d0 = np.maximum(disp_from(ax, ay, w0, w1), disp_from(kx, ky, w0, w1))
    qs, qbest = w0, 1e18
    for j in range(w0, max(w0 + 1, w1 - 7)):
        v = float(np.nanstd(d0[j:j + 7]))
        if v < qbest:
            qbest, qs = v, j
    qe = qs + 7

    # pass 2: planted baseline + noise from that quiet plateau
    m = np.maximum(disp_from(ax, ay, qs, qe), disp_from(kx, ky, qs, qe)) / max(diag_ref, 1.0)
    m = smooth(m, 3)
    noise = float(np.nanstd(m[qs:qe]))
    # floor ≈ 7 px on a 1280×720 clip: clears pre-pitch micro-drift / camera sway
    # so we fire on the genuine delivery initiation, not the settle.
    thresh = max(4.0 * noise, 0.0048)

    start = max(1, set_end - 3)
    hi = max(start + 1, peak + 1)
    for i in range(start, min(hi, n)):
        # commit on first crossing whose *median* over the next 5 frames stays
        # elevated — median ignores 1–2 flicker dropouts (robust to noisy joints)
        if m[i] > thresh and float(np.nanmedian(m[i:min(i + 5, n)])) > thresh:
            return interp_crossing(i - 1, m[i - 1], m[i], thresh)
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
    # (frames where both shoulder & hip are missing yield NaN; interpolate fills them —
    #  suppress the benign "Mean of empty slice" warning for those all-NaN columns)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bx = np.nanmean([kpts[:, sho, 0], kpts[:, hip, 0]], axis=0)
            by = np.nanmean([kpts[:, sho, 1], kpts[:, hip, 1]], axis=0)
    bx = pd.Series(bx).interpolate().bfill().ffill().values
    by = pd.Series(by).interpolate().bfill().ffill().values
    reach = np.hypot(wx - bx, wy - by) / max(diag_ref, 1.0)
    reach = smooth(reach, 3)

    n = len(speed)
    lo = int(np.ceil(first_move_idx)) + 3
    if lo >= n - 2:
        return float(n - 1), 0.0
    hi = min(n - 1, lo + 95)                 # cap the delivery window (~1.6 s @60fps);
                                             # excludes the far follow-through whip that
                                             # otherwise wins a naive global speed-argmax.

    # Is the throwing wrist actually tracked well? (reach must have real dynamic
    # range — a flat reach means the keypoint is barely moving / poorly detected.)
    rw = reach[lo:hi]
    spread = float(np.nanpercentile(rw, 90) - np.nanpercentile(rw, 10))

    if spread >= 0.010:
        method = "extension_collapse"
        # ── good tracking: extension-then-collapse model ──────────────────────
        # At release the throwing hand is at max forward extension while HIGH
        # (out front), then `reach` collapses as the arm whips down across the
        # body.  Earlier reach humps (arm swinging *down* during the leg lift)
        # are rejected by requiring the wrist to be in the upper part of its
        # vertical travel.  Release = peak hand speed between that high-hand
        # extension peak and the reach-collapse onset.
        wyw = wy[lo:hi]
        wy_gate = float(np.nanmin(wyw) + 0.55 * (np.nanmax(wyw) - np.nanmin(wyw)))
        rg = reach.copy()
        rg[wy > wy_gate] = -1e9              # drop low-hand (arm-swing-down) frames
        rg[:lo] = -1e9
        rg[hi:] = -1e9
        r_ext = int(np.nanargmax(rg))
        if rg[r_ext] <= -1e8:                # all frames gated out — fall back
            r_ext = lo + int(np.nanargmax(reach[lo:hi]))
        peakv = reach[r_ext]

        coll = None
        for j in range(r_ext + 1, min(r_ext + 12, hi)):
            if reach[j] < 0.5 * peakv:
                coll = j
                break
        end_ref = coll if coll is not None else min(r_ext + 8, hi)

        a = max(lo, r_ext)
        b = max(a + 1, int(np.ceil(end_ref)))
        seg = speed[a:b]
        pk = a + int(np.nanargmax(seg)) if seg.size and not np.all(np.isnan(seg)) else r_ext
    else:
        # ── poor wrist tracking (flat reach): reach is uninformative.  Use the
        #    PEAK WRIST HEIGHT (top of the throwing arc) as the release proxy,
        #    but take the FIRST turning point — not the global min — because a
        #    poorly-tracked wrist jumps around in the follow-through and would
        #    otherwise win the global extreme.  Heavy smoothing kills jitter so
        #    the descent-to-top-then-drop reads as one clean local minimum.
        method = "flat_reach_height"
        hh = min(n - 1, lo + 85)
        wysm = smooth(wy, 7)
        rise_margin = 0.004 * diag_ref               # hand must come back down ≥~6 px
        pk = None
        for i in range(lo + 3, hh - 5):
            # local min that the hand clearly comes back DOWN from (wy rises)
            if (wysm[i] <= wysm[i - 1]
                    and float(np.nanmedian(wysm[i + 1:i + 6])) > wysm[i] + rise_margin
                    and wysm[i] < float(np.nanmedian(wysm[lo:lo + 4]))):
                pk = i
                break
        if pk is None:
            seg = wy[lo:hh]
            pk = lo + int(np.nanargmin(seg)) if seg.size and not np.all(np.isnan(seg)) else lo

    conf = float(kpts[pk, w, 2]) if not np.isnan(kpts[pk, w, 2]) else 0.0
    if method == "flat_reach_height":
        conf *= 0.7                          # weaker estimator → flag lower trust
    # sub-frame parabolic refine on the speed curve
    if 0 < pk < n - 1:
        off = parabolic_peak(speed[pk - 1], speed[pk], speed[pk + 1])
    else:
        off = 0.0
    return pk + off, conf, method


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


def _detect_in_segment(seg, p_throws, model, diag, bbox_hint=None):
    """Run pose + event detection on one contiguous shot.

    Returns a dict of segment-relative results (or None if no pitcher track).
    All frame indices are relative to the START of this segment.
    """
    tracks = run_pose(seg, model)
    tid, pseq = select_pitcher(tracks, seg[0].shape[1], seg[0].shape[0], bbox_hint)
    if pseq is None:
        return None

    kpts, boxes = densify(pseq, len(seg))
    me = motion_energy_series(kpts, diag)
    peak = int(np.nanargmax(me))

    set_i, motion_onset, floor, std = find_set_and_first_move(me)
    # FIRST MOVEMENT = lead foot leaves the ground (project definition).
    # Priority: leg-lift (primary) -> hand-break -> generic motion onset.
    ll = leg_lift_frame(kpts, p_throws, set_i, peak, diag)
    hb = hand_break_frame(kpts, set_i, peak)
    if ll is not None:
        fm, fm_cue = float(ll), "leg_lift"
    elif hb is not None:
        fm, fm_cue = float(hb), "hand_break"
    else:
        fm, fm_cue = float(motion_onset), "motion_onset"

    rel, rel_conf, rel_method = find_release(kpts, p_throws, fm, diag)
    return {
        "tid": tid, "kpts": kpts, "boxes": boxes,
        "set_i": set_i, "fm": fm, "rel": rel, "fm_cue": fm_cue,
        "rel_conf": rel_conf, "rel_method": rel_method,
        "ll": ll, "hb": hb, "me_peak": float(me[peak]),
    }


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

    n_full = len(frames)
    if MAX_ANALYSIS_S is not None and fps > 0:
        cap_n = int(round(MAX_ANALYSIS_S * fps))
        if 0 < cap_n < len(frames):
            frames = frames[:cap_n]

    row = {"clip_id": path.name, "p_throws": p_throws, "fps": round(fps, 3),
           "n_frames": n_full, "analysis_frames": len(frames), "status": "ok"}
    if len(frames) < 8:
        row["status"] = "too_short"
        return row

    h, w = frames[0].shape[:2]
    diag = float(np.hypot(w, h))

    # Enumerate every contiguous shot.  The pitcher's full delivery may NOT be in
    # the opening shot (broadcast can open on a runner-on-first angle), so we
    # scan shots in order and keep the first one that yields a clean, fully
    # visible delivery (trackable throwing hand + in-band timing).  Shorter
    # shots that can't physically contain a delivery are skipped.
    segments = scene_segments(frames)
    candidates = [(s, e) for (s, e) in segments if (e - s) >= MIN_SEG_FRAMES]
    if not candidates:                       # nothing long enough — use longest shot
        candidates = [max(segments, key=lambda se: se[1] - se[0])]
    row["n_segments"] = len(segments)
    row["scene_cut_at"] = segments[0][1] if len(segments) > 1 else -1

    best = None                              # (quality_key, seg, det)
    chosen = None
    for (s, e) in candidates:
        det = _detect_in_segment(frames[s:e], p_throws, model, diag, bbox_hint)
        if det is None:
            continue
        delivery_s = (det["rel"] - det["fm"]) / fps
        in_band = PLAUSIBLE_BAND[0] <= delivery_s <= PLAUSIBLE_BAND[1]
        good_track = (det["rel_method"] != "flat_reach_height")
        good_conf = det["rel_conf"] >= KPT_CONF_MIN
        # rank: a fully-visible throw (good_track) that is in-band and well
        # tracked wins; ties broken by detection confidence then motion energy.
        qkey = (int(good_track), int(in_band), int(good_conf),
                det["rel_conf"], det["me_peak"])
        if best is None or qkey > best[0]:
            best = (qkey, (s, e), det)
        # short-circuit: a clean in-band, well-tracked delivery is good enough
        if good_track and in_band and good_conf:
            chosen = (s, e, det)
            break

    if best is None:
        row["status"] = "no_pitcher"
        return row
    if chosen is None:
        chosen = (best[1][0], best[1][1], best[2])
    s, e, det = chosen

    kpts, boxes = det["kpts"], det["boxes"]
    set_i, fm, rel = det["set_i"], det["fm"], det["rel"]
    rel_conf, rel_method, fm_cue = det["rel_conf"], det["rel_method"], det["fm_cue"]
    ll, hb = det["ll"], det["hb"]
    delivery_s = (rel - fm) / fps

    row["pitcher_track_id"] = det["tid"]
    row["analysis_frames"] = e - s
    row["chosen_segment"] = f"[{s}:{e}]"
    row["segment_index"] = segments.index((s, e)) if (s, e) in segments else -1
    # report event frames in FULL-CLIP coordinates (so they line up with manual
    # labels, which are numbered against the whole clip)
    off = s
    row.update({
        "set_frame": set_i + off,
        "first_move_frame": round(fm + off, 2),
        "release_frame": round(rel + off, 2),
        "delivery_s": round(delivery_s, 4),
        "release_kpt_conf": round(rel_conf, 3),
        "release_method": rel_method,
        "first_move_cue": fm_cue,
        "leg_lift_frame": round(ll + off, 2) if ll is not None else None,
        "hand_break_frame": round(hb + off, 2) if hb is not None else None,
    })

    # confidence flags
    in_band = PLAUSIBLE_BAND[0] <= delivery_s <= PLAUSIBLE_BAND[1]
    good_conf = rel_conf >= KPT_CONF_MIN
    # a delivery detected within a contiguous shot is, by construction, not cut
    # off mid-throw; pre_cut stays True for any successful in-shot detection.
    pre_cut = True
    # `flat_reach_height` means YOLO never tracked the throwing hand (reach was
    # flat) — the release estimate is unreliable, so the clip is NOT usable for
    # accuracy.  It still counts toward coverage (flag what we can't measure
    # rather than emit a silently-wrong release).
    good_track = (rel_method != "flat_reach_height")
    row["good_track"] = bool(good_track)
    row["in_band"] = bool(in_band)
    row["release_pre_cut"] = bool(pre_cut)
    row["usable"] = bool(in_band and pre_cut and good_conf and good_track)
    row["confidence"] = round(
        (0.4 * good_conf + 0.25 * in_band + 0.15 * pre_cut + 0.2 * good_track), 3)

    if make_qa:
        QA_DIR.mkdir(parents=True, exist_ok=True)
        out = QA_DIR / f"{path.stem}_annotated.mp4"
        try:
            # QA overlay uses segment-relative indices on the chosen shot
            write_qa(frames[s:e], fps, kpts, boxes, set_i, fm, rel, delivery_s, out)
            row["qa_video"] = out.name
        except Exception as e2:               # QA is non-critical
            row["qa_video"] = f"ERR:{e2}"
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
    ap.add_argument("--weights", default=YOLO_WEIGHTS,
                    help="pose weights; yolov8n-pose.pt is ~5-8x faster than -m")
    ap.add_argument("--device", default="auto",
                    help="auto|mps|cpu|cuda|0  (auto picks MPS on Apple Silicon)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference image size (lower = faster, less detail)")
    ap.add_argument("--half", action="store_true", help="FP16 inference where supported")
    ap.add_argument("--max-analysis-s", type=float, default=None,
                    help="only analyse the first N seconds of each clip "
                         "(faster; trims runner-follow tail on continuous shots)")
    ap.add_argument("--clips-dir", help="override clips/ directory")
    ap.add_argument("--meta", help="override clips_meta.csv path")
    ap.add_argument("--out", help="override pilot_results.csv output path")
    ap.add_argument("--qa-dir", help="override qa/ directory")
    args = ap.parse_args()

    global CLIPS_DIR, QA_DIR, META_PATH, RESULTS_CSV, DEVICE, IMGSZ, HALF, MAX_ANALYSIS_S
    MAX_ANALYSIS_S = args.max_analysis_s
    # Resolve compute device.  On a fanless M2 the GPU (MPS) is far faster than
    # CPU and avoids pegging all cores; fall back to CPU if MPS is unavailable.
    dev = args.device
    if dev == "auto":
        try:
            import torch
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    DEVICE, IMGSZ, HALF = dev, args.imgsz, args.half
    print(f"[device] {DEVICE}  imgsz={IMGSZ}  half={HALF}")
    if args.clips_dir:
        CLIPS_DIR = Path(args.clips_dir)
    if args.qa_dir:
        QA_DIR = Path(args.qa_dir)
    if args.meta:
        META_PATH = Path(args.meta)
    if args.out:
        RESULTS_CSV = Path(args.out)

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
        # carry relational metadata (join keys to the Naylor model) from meta
        for col in ("pitcher_name", "pitcher_id", "catcher_name", "catcher_id",
                    "batter_name", "play_id", "game_pk", "at_bat_number",
                    "pitch_number", "is_naylor"):
            if col in m and pd.notna(m[col]):
                row[col] = m[col]
        print(f"        -> {row.get('status')}  "
              f"delivery_s={row.get('delivery_s')}  usable={row.get('usable')}")
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["clip_id", "pitcher_name", "pitcher_id", "catcher_name", "catcher_id",
            "play_id", "game_pk", "at_bat_number", "pitch_number",
            "p_throws", "fps", "n_frames", "analysis_frames", "scene_cut_at",
            "n_segments", "segment_index", "chosen_segment",
            "set_frame", "first_move_frame", "release_frame", "delivery_s",
            "first_move_cue", "leg_lift_frame", "hand_break_frame",
            "release_method", "release_kpt_conf", "good_track",
            "in_band", "release_pre_cut", "usable",
            "confidence", "qa_video", "pitcher_track_id", "status"]
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[write] {RESULTS_CSV}  ({len(df)} clips)")
    if "usable" in df.columns:
        print(f"        usable: {int(df['usable'].sum())}/{len(df)}")


if __name__ == "__main__":
    main()
