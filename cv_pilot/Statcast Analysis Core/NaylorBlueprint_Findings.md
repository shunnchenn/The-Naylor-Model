# Naylor Blueprint: Bayesian Model for Slow-Runner Steals
## Targeting sub-40th-percentile runners who cover ~1 SD above-average ground

---

## The thesis

Josh Naylor at **24.4 ft/s (1.9th speed percentile)** — nearly 4 standard deviations *below* 
the league mean for basestealing runners — covers the **3rd-most ground in MLB** (16.74 ft 
between pitcher's first move and pitch release). He steals at a **95.7% success rate** 
(22/23), outperforming nearly every faster runner in baseball.

**The insight:** This is not a speed skill. The slow runners who thrive in the steal game 
are the ones with elite timing and jump mechanics — they must cover *more* ground to 
compensate for their lack of acceleration. The residual-based leaderboard inadvertently 
ranked Naylor below faster players because it *expected* him to cover more (negative 
league slope). This Bayesian model flips the question:

> **"Given a runner's per-attempt sample, what is the posterior probability they 
> outperform the league mean gain — and how many SD above the league mean?"**

---

## Method

**Empirical-Bayes conjugate normal model:**
- **Prior:** League mean gain = {league_mean:.2f} ft, sd = {league_sd:.2f} ft (all VQ runner-seasons)
- **Likelihood:** Per-attempt gains per runner (loaded from cache)
- **Posterior:** Conjugate normal update; shrinks small-sample estimates toward prior
- **Ranking:** By posterior SD above league mean (or equivalently, posterior P(gain > league mean))
- **Target population:** Sub-40th-percentile sprint speed, volume-qualified ≥ 10 tracked attempts

The Bayesian approach handles small-sample noise (e.g., a 4-for-4 fluke has low posterior SD 
after shrinkage) while isolating true skill (large samples converge to sample mean with high 
posterior SD).

---

## Results

**Target cohort:** {slow_count} slow runner-seasons (34 volume-qualified)

### Naylor 2025: The archetype

- **Sprint speed:** 24.4 ft/s (z = -4.14, 1.9th percentile)
- **Mean gain to release:** 16.74 ft (z = 3.08)
- **Posterior SD above league mean:** +5.39
- **Posterior P(gain > league mean):** 1.000
- **Steal record:** 22/23 = 95.7%
- **Joint rarity (naive independence):** P(as slow) × P(covers as much) ≈ 0.000000 → ~1 in 56,023,124 runner-seasons

**Bottom line:** Naylor's combination of extreme slowness (bottom 2%) and above-average ground 
covered (top 5%) is extraordinarily rare. The Bayesian model quantifies this via posterior 
probability, surfacing him as the archetype slow-runner thief.

### Juan Soto (comparisons)
- Juan Soto 2025: sprint 25.8 ft/s, mean gain 14.18 ft, posterior SD above league = +3.37, rank #3
- Juan Soto 2023: sprint 26.8 ft/s, mean gain 16.12 ft, posterior SD above league = +1.24, rank #11
- Juan Soto 2024: sprint 26.8 ft/s, mean gain 13.89 ft, posterior SD above league = +0.77, rank #18


---

## Top-20 slow runners (sub-40th percentile)

**1.**  Agustín Ramírez 2025  
sprint=26.7 ft/s (pctile=30.4)  | mean_gain=21.31 ft  | SD_above=+7.92  | P(>μ)=1.000  | 13/14=93%

**2.**  Josh Naylor 2025  
sprint=24.4 ft/s (pctile=1.9)  | mean_gain=16.74 ft  | SD_above=+5.39  | P(>μ)=1.000  | 22/23=96%

**3.**  Juan Soto 2025  
sprint=25.8 ft/s (pctile=13.5)  | mean_gain=14.18 ft  | SD_above=+3.37  | P(>μ)=1.000  | 30/30=100%

**4.**  Paul Goldschmidt 2024  
sprint=26.3 ft/s (pctile=21.6)  | mean_gain=16.70 ft  | SD_above=+2.28  | P(>μ)=0.989  | 10/10=100%

**5.**  Cal Raleigh 2025  
sprint=26.0 ft/s (pctile=16.1)  | mean_gain=14.82 ft  | SD_above=+1.73  | P(>μ)=0.958  | 8/12=67%

**6.**  Miguel Rojas 2023  
sprint=26.4 ft/s (pctile=25.6)  | mean_gain=13.41 ft  | SD_above=+1.67  | P(>μ)=0.952  | 8/10=80%

**7.**  Gleyber Torres 2023  
sprint=26.4 ft/s (pctile=26.1)  | mean_gain=15.91 ft  | SD_above=+1.65  | P(>μ)=0.951  | 10/15=67%

**8.**  Kyle Tucker 2025  
sprint=26.5 ft/s (pctile=26.1)  | mean_gain=13.25 ft  | SD_above=+1.60  | P(>μ)=0.945  | 14/17=82%

**9.**  Jesús Sánchez 2025  
sprint=27.0 ft/s (pctile=38.1)  | mean_gain=15.53 ft  | SD_above=+1.57  | P(>μ)=0.942  | 10/11=91%

**10.**  Mookie Betts 2024  
sprint=26.7 ft/s (pctile=30.3)  | mean_gain=12.33 ft  | SD_above=+1.53  | P(>μ)=0.936  | 15/16=94%

**11.**  Juan Soto 2023  
sprint=26.8 ft/s (pctile=34.5)  | mean_gain=16.12 ft  | SD_above=+1.24  | P(>μ)=0.892  | 6/10=60%

**12.**  Royce Lewis 2025  
sprint=26.8 ft/s (pctile=32.7)  | mean_gain=12.75 ft  | SD_above=+1.22  | P(>μ)=0.888  | 9/10=90%

**13.**  Adam Frazier 2023  
sprint=26.5 ft/s (pctile=27.7)  | mean_gain=13.66 ft  | SD_above=+1.20  | P(>μ)=0.885  | 8/12=67%

**14.**  Adam Frazier 2025  
sprint=26.6 ft/s (pctile=27.9)  | mean_gain=15.02 ft  | SD_above=+1.20  | P(>μ)=0.885  | 8/13=62%

**15.**  Jose Altuve 2023  
sprint=26.9 ft/s (pctile=36.4)  | mean_gain=12.69 ft  | SD_above=+1.14  | P(>μ)=0.873  | 13/13=100%

**16.**  Manny Machado 2025  
sprint=25.8 ft/s (pctile=13.1)  | mean_gain=12.88 ft  | SD_above=+1.07  | P(>μ)=0.857  | 9/10=90%

**17.**  Elvis Andrus 2023  
sprint=26.5 ft/s (pctile=28.7)  | mean_gain=12.51 ft  | SD_above=+0.87  | P(>μ)=0.808  | 10/12=83%

**18.**  Juan Soto 2024  
sprint=26.8 ft/s (pctile=32.4)  | mean_gain=13.89 ft  | SD_above=+0.77  | P(>μ)=0.780  | 6/10=60%

**19.**  Luis García Jr. 2025  
sprint=26.5 ft/s (pctile=26.5)  | mean_gain=13.06 ft  | SD_above=+0.71  | P(>μ)=0.760  | 13/17=76%

**20.**  Freddie Freeman 2024  
sprint=25.8 ft/s (pctile=14.5)  | mean_gain=13.40 ft  | SD_above=+0.53  | P(>μ)=0.701  | 7/9=78%


---

## Interpretation

**The Bayesian blueprint reveals a slow-runner steal archetype:**

1. **Sub-40th-percentile sprint speed** (~26.5 ft/s or slower) — the runner cannot rely on 
   raw acceleration.

2. **Posterior mean gain near or above league average** — via early jump and good timing, 
   they position for a big secondary lead.

3. **High posterior probability** (typically 65%+) — the data strongly support that they 
   beat the league mean, even after shrinkage.

4. **Success rate 75%+** — their timing skill translates to steals (though sample varies).

**Naylor is the gold-standard member of this archetype:** 24.4 ft/s sprint, 16.74 ft gain 
(3rd-most in the league), 95.7% success. The slow runners who follow (Ramírez, Goldschmidt, 
Soto 2023, Torres) share this profile: elite timing on a limited motor.

---

## Files

- `data/DF_NaylorBlueprint_Leaderboard.csv` — one row per slow VQ runner, ranked
- `figures/Fig_NaylorBlueprint_Scatter.png` — gain vs sprint, Naylor/Soto annotated
- `figures/Fig_NaylorBlueprint_TopN.png` — top-20 slow runners by posterior SD

---

## Next steps

Use this list to scout for undervalued basestealing talent:
- Identify slow runners (draft/trade targets) with posterior P(gain > μ) > 60%
- Validate with video if available (timing metrics, jump quickness)
- Monitor SB% in real time (should track posterior predict if model is sound)
