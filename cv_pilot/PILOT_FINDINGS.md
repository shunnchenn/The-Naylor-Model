# Pitch Delivery CV Pilot — Findings

_Generated 2026-05-29_

## Coverage

- Clips processed: **3**
- Usable auto measurements (events found pre-cut, in-band, confident): **2**
- **Coverage = 66.7%**  (gate ≥ 70%)

## Accuracy (auto vs. manual ground truth)

- n labeled clips: **3**  (mean 59.7 fps)
- Release frame error — MAE **11.17 frames (≈ 187 ms)**, bias +11.17 frames
- First-move frame error — MAE **1.85 frames (≈ 31 ms)**
- Fitted constant release offset (subtract from auto): **+11.17 frames** — applying it would reduce systematic bias.
- Release MAE after offset correction: **14.22 frames**

| clip | auto rel | manual rel | err (frames) | auto delivery_s | manual delivery_s |
|---|---|---|---|---|---|
| 8c62e264_pagan_R.mp4 | 205.5 | 173.0 | 32.5 | 1.432 | 0.855 |
| d7686082_waldron_R.mp4 | 179.5 | 179.0 | 0.5 | 1.106 | 1.055 |
| d9c74db4_barnes_R.mp4 | 176.5 | 176.0 | 0.5 | 1.263 | 1.272 |

## Repeatability (within-pitcher delivery std)

| pitcher | n | mean delivery_s | std (s) |
|---|---|---|---|
| Matt Barnes | 1 | 1.263 | n/a |
| Matt Waldron | 1 | 1.106 | n/a |

## External sanity check

- Usable deliveries within plausible MLB band 0.85–1.55s: **100%**
- Median usable delivery_s: **1.185** (scout TTP ≈ delivery + ~0.40s ball-flight)

## Verdict

- ❌ coverage ≥ 70%: 66.7%
- ❌ release MAE ≤ 2 frames: 11.17 frames
- ❌ within-pitcher std ≤ 0.10s: insufficient n

### → **INCOMPLETE (need manual labels / more clips)**

Run `label_tool.py` on all clips and/or add more clips per pitcher, then re-run `evaluate.py` for a final verdict.
