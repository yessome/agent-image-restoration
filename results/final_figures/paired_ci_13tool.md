**Paired contrasts, 13-tool configuration.**

Each source image is one cluster: its degraded versions and three classifier seeds enter and leave the resampling together. Intervals are percentile bootstrap 95% CIs over 10,000 resamples of the 430 sources.

**Six-scene matrix (430 sources x 6 scenes x 3 seeds, 7,740 records)**

| Contrast | Delta (pp) | 95% CI (pp) |
|---|---:|---|
| learned - raw | +4.20 | [+3.28, +5.17] |
| learned - fixed denoise | +4.11 | [+2.80, +5.44] |
| learned - fixed lowlight | +7.09 | [+5.36, +8.84] |

**Two-family matrix (430 sources x 2 scenes x 3 seeds, 2,580 records)**

| Contrast | Delta (pp) | 95% CI (pp) |
|---|---:|---|
| learned - raw | +8.99 | [+6.82, +11.20] |
| learned - fixed denoise | +7.56 | [+5.50, +9.65] |
| learned - fixed lowlight | +8.64 | [+6.55, +10.70] |

| Scope | Metric | Raw | Learned iterative agent |
|---|---|---:|---:|
| Two-family matrix (860 instances) | Accuracy | 0.6899 | 0.7798 |
| Two-family matrix (860 instances) | Macro-F1 | 0.6626 | 0.7577 |
| Six-scene matrix (2,580 instances) | Accuracy | 0.7093 | 0.7513 |
| Six-scene matrix (2,580 instances) | Macro-F1 | 0.6823 | 0.7259 |
