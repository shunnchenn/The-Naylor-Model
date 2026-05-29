# CV Pilot — Pitch Delivery Detection

A feasibility pilot for measuring **pitcher delivery time** — *first movement
(delivery initiation from the stretch/slide-step) → ball release* — directly from
Statcast base-stealing broadcast clips, using pose estimation.

The goal: prove the extraction is accurate and repeatable enough to later replace the
Naylor Model's `LEAGUE_PITCHER_TTP = 1.30` constant with a **real per-pitcher average**.
Nothing here touches the model pipeline until the pilot passes its go/no-go gate.

## How it works

| Step | What it does |
|---|---|
| **Scene-cut guard** | Only the opening pitching shot is analyzed (clips cut to follow the runner). |
| **YOLOv8-pose** | Multi-person keypoints; the pitcher is picked by frame-center + motion-energy and tracked. |
| **Set window** | Lowest-motion plateau before the delivery anchors the baseline. |
| **First movement** | Motion-energy onset after the set, cross-checked with hand-break and lead-leg lift. |
| **Ball release** | Peak throwing-hand speed + reach, parabolic sub-frame refinement (beats the 33 ms/frame quantum). |

## Setup

```bash
pip install -r cv_pilot/requirements.txt   # ultralytics, opencv, numpy, pandas
```

YOLOv8-pose weights download automatically on first run.

## Workflow

1. **Drop clips** (~4–6 pitchers × 3–4 clips; mix LHP/RHP and windup/slide-step) into
   `cv_pilot/clips/` and list them in `cv_pilot/clips_meta.csv`.
2. **Run the detector** → `pilot_results.csv` + annotated QA videos in `qa/`:
   ```bash
   python3 cv_pilot/extract_delivery.py
   ```
3. **Eyeball QA** — open a few `qa/*_annotated.mp4`; check the RELEASE / FIRST-MOVE
   markers land on the right frames.
4. **Label ground truth** — step through each clip and mark the true frames:
   ```bash
   python3 cv_pilot/label_tool.py
   ```
5. **Evaluate + verdict** → writes `PILOT_FINDINGS.md`:
   ```bash
   python3 cv_pilot/evaluate.py
   ```

## Go/no-go gate

- Release error ≤ **2 frames** (≈ ±66 ms @ 30 fps)
- Within-pitcher delivery std ≤ **0.10 s**
- Coverage ≥ **70 %** of clips usable

A NO-GO with quantified error is a valid result — the model simply keeps the constant.

## Files

```
extract_delivery.py   detector (core)
label_tool.py         manual frame labeler (ground truth)
evaluate.py           accuracy / repeatability / coverage + verdict
clips_meta.csv        per-clip pitcher + handedness metadata
clips/   qa/          video in/out  (gitignored — never committed)
labels_manual.csv     committed ground-truth labels
pilot_results.csv     committed auto results
PILOT_FINDINGS.md     committed verdict
```

## Note on tempo (60 fps clips)

Higher-fps clips give finer temporal resolution. The detector reads the container's
true fps per clip, so 60 fps clips automatically yield ~16 ms resolution vs. ~33 ms at
30 fps — prefer them where available.
