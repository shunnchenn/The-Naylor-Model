# Pitch Delivery CV Pilot — Findings

_Generated 2026-05-29_

## Coverage

- Clips processed: **60**
- Usable auto measurements (events found pre-cut, in-band, confident): **45**
- **Coverage = 75.0%**  (gate ≥ 70%)

## Accuracy (auto vs. manual ground truth)

- n labeled clips: **44**  (mean 59.7 fps)
- Release frame error — MAE **132.75 frames (≈ 2223 ms)**, bias +127.10 frames
- First-move frame error — MAE **135.63 frames (≈ 2272 ms)**
- Fitted constant release offset (subtract from auto): **+127.10 frames** — applying it would reduce systematic bias.
- Release MAE after offset correction: **147.41 frames**

| clip | auto rel | manual rel | err (frames) | auto delivery_s | manual delivery_s |
|---|---|---|---|---|---|
| 0028354f_bassitt_R.mp4 | 211.1 | 212.0 | -0.9 | 1.110 | 1.183 |
| 0670c556_urena_R.mp4 | 198.2 | 197.0 | 1.2 | 0.909 | 0.900 |
| 0b6ea438_gausman_R.mp4 | 457.5 | 209.0 | 248.5 | 1.000 | 1.083 |
| 13727210_garcia_R.mp4 | 178.5 | 178.0 | 0.5 | 1.021 | 1.050 |
| 17477d36_castillo_R.mp4 | 827.5 | 173.0 | 654.5 | 1.056 | 1.083 |
| 1f5c9475_castillo_R.mp4 | 331.5 | 180.0 | 151.5 | 0.242 | 0.967 |
| 21a4d22a_king_R.mp4 | 483.8 | 180.0 | 303.8 | 0.064 | 0.867 |
| 26b50bd5_castillo_R.mp4 | 359.1 | 176.0 | 183.1 | 1.232 | 1.283 |
| 2762fd25_littell_R.mp4 | 645.0 | 182.0 | 463.0 | 1.336 | 1.117 |
| 28a9b0ce_eflin_R.mp4 | 179.5 | 180.0 | -0.5 | 0.895 | 0.883 |
| 2e082507_brown_R.mp4 | 180.5 | 181.0 | -0.5 | 1.050 | 1.050 |
| 34c44990_vasquez_R.mp4 | 683.5 | 176.0 | 507.5 | 0.142 | 1.033 |
| 42e35d54_gallen_R.mp4 | 181.5 | 182.0 | -0.5 | 1.293 | 1.300 |
| 583db962_wilson_R.mp4 | 377.9 | 185.0 | 192.9 | 1.438 | 0.967 |
| 5c19401f_wilson_R.mp4 | 180.5 | 180.0 | 0.5 | 0.935 | 0.917 |
| 5c298e7e_crawford_R.mp4 | 425.0 | 179.0 | 246.0 | 0.611 | 1.017 |
| 650ed646_houck_R.mp4 | 181.5 | 180.0 | 1.5 | 1.085 | 1.083 |
| 7a8fbf89_bassitt_R.mp4 | 236.3 | 182.0 | 54.3 | 1.595 | 1.033 |
| 7c15c67b_eflin_R.mp4 | 184.5 | 184.0 | 0.5 | 0.949 | 0.967 |
| 836d520f_urena_R.mp4 | 177.5 | 181.0 | -3.5 | 0.875 | 0.933 |
| 8c148f19_maeda_R.mp4 | 235.5 | 185.0 | 50.5 | 0.943 | 0.900 |
| 8c62e264_pagan_R.mp4 | 505.1 | 173.0 | 332.1 | 0.069 | 0.855 |
| 96786d3f_flaherty_R.mp4 | 177.5 | 179.0 | -1.5 | 0.965 | 0.967 |
| 9ddf4f0e_keller_R.mp4 | 409.7 | 179.0 | 230.7 | 0.086 | 0.783 |
| a32b0537_brown_R.mp4 | 110.0 | 185.0 | -75.0 | 0.863 | 0.867 |
| a3cf2cda_littell_R.mp4 | 413.6 | 179.0 | 234.6 | 0.941 | n/a |
| a93076b6_urena_R.mp4 | 180.5 | 182.0 | -1.5 | 0.961 | 1.017 |
| b39330a9_keller_R.mp4 | 412.9 | 181.0 | 231.9 | 1.074 | 0.667 |
| b3cf833b_eflin_R.mp4 | 226.2 | 182.0 | 44.2 | 0.090 | 0.850 |
| bdb833e1_flaherty_R.mp4 | 180.5 | 184.0 | -3.5 | 0.961 | 1.000 |
| bf24608f_keller_R.mp4 | 652.1 | 176.0 | 476.1 | 0.270 | 0.617 |
| c1cf66e8_lively_R.mp4 | 168.5 | 175.0 | -6.5 | 0.961 | 1.083 |
| cab1943e_morton_R.mp4 | 178.5 | 181.0 | -2.5 | 0.983 | 1.033 |
| d0ee545f_vasquez_R.mp4 | 369.5 | 180.0 | 189.5 | 1.138 | 1.050 |
| d69c672a_king_R.mp4 | 154.5 | 181.0 | -26.5 | 0.962 | 0.883 |
| d7686082_waldron_R.mp4 | 179.5 | 179.0 | 0.5 | 1.017 | 1.055 |
| d9c74db4_barnes_R.mp4 | 176.5 | 176.0 | 0.5 | 1.157 | 1.272 |
| da344eb5_bassitt_R.mp4 | 179.5 | 179.0 | 0.5 | 1.023 | 1.050 |
| db567a85_feltner_R.mp4 | 587.6 | 176.0 | 411.6 | 0.710 | 1.167 |
| e6fa2a28_wilson_R.mp4 | 265.3 | 181.0 | 84.3 | 1.218 | 0.867 |
| f45957d5_brown_R.mp4 | 182.4 | 182.0 | 0.4 | 0.942 | 0.933 |
| f8c40b5b_maeda_R.mp4 | 249.5 | 176.0 | 73.5 | 1.270 | 0.917 |
| f9188098_gausman_R.mp4 | 179.5 | 181.0 | -1.5 | 0.968 | 1.050 |
| f9c3cc8b_maeda_R.mp4 | 526.4 | 180.0 | 346.4 | 1.297 | 1.267 |

## Delivery-time accuracy (delivery_s, the production metric)

- ALL labelled+detected: n=43, **MAE 0.219s** (median 0.073s), bias -0.095s, |err|>0.30s on 14
- usable only: n=32, **MAE 0.108s** (median 0.047s), bias +0.053s, |err|>0.30s on 5

## Repeatability (within-pitcher delivery std)

| pitcher | n | mean delivery_s | std (s) |
|---|---|---|---|
| Ben Lively | 3 | 0.962 | 0.051 |
| Bryse Wilson | 3 | 1.197 | 0.252 |
| Charlie Morton | 3 | 0.967 | 0.023 |
| Chris Bassitt | 3 | 1.243 | 0.308 |
| Hunter Brown | 2 | 0.996 | 0.077 |
| Jack Flaherty | 2 | 0.963 | 0.003 |
| José Ureña | 3 | 0.915 | 0.043 |
| Kenta Maeda | 3 | 1.170 | 0.197 |
| Kevin Gausman | 3 | 1.058 | 0.129 |
| Kutter Crawford | 1 | 1.060 | n/a |
| Luis Castillo | 2 | 1.144 | 0.125 |
| Luis García | 2 | 1.288 | 0.378 |
| Michael King | 1 | 1.012 | n/a |
| Mitch Keller | 1 | 1.074 | n/a |
| Randy Vásquez | 1 | 1.138 | n/a |
| Ryan Feltner | 1 | 0.880 | n/a |
| Tanner Houck | 2 | 1.113 | 0.040 |
| Zac Gallen | 2 | 1.466 | 0.245 |
| Zach Eflin | 2 | 0.922 | 0.039 |
| Zack Littell | 3 | 1.306 | 0.350 |

- Worst within-pitcher std (n≥2): **0.378 s**  (gate ≤ 0.10 s)

## External sanity check

- Usable deliveries within plausible MLB band 0.85–1.55s: **91%**
- Median usable delivery_s: **1.021** (scout TTP ≈ delivery + ~0.40s ball-flight)

## Verdict

- ✅ coverage ≥ 70%: 75.0%
- ❌ release MAE ≤ 2 frames: 132.75 frames
- ❌ within-pitcher std ≤ 0.10s: 0.378 s

### → **NO-GO**

Extraction is **not** reliable enough at this resolution. The model keeps the LEAGUE_PITCHER_TTP=1.30 constant unchanged. This is a clean negative result, not a bug — see the error numbers above. Options: higher-fps clips, better release model, or accept per-pitcher means with wider uncertainty.
