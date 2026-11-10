**Table 7. Mechanism ablation on the six-scene test matrix (13 tools, 430 sources x 6 scenes x 3 seeds).**

Full rows reuse the frozen main-run records; the two ablations change one factor each on the same checkpoint. The one-call condition keeps the original budget encoding and only truncates execution after the first call, so it is a mechanism ablation rather than an optimised single-step policy.

### Per-seed

| Seed | Condition | Accuracy | Fixes | Harms | Mean calls |
|---:|---|---:|---:|---:|---:|
| 1 | Full agent | 0.7450 | 165 | 62 | 0.468 |
| 1 | At most one call | 0.7411 | 155 | 62 | 0.426 |
| 1 | No branch feedback | 0.7488 | 165 | 52 | 0.562 |
| 2 | Full agent | 0.7469 | 181 | 69 | 0.405 |
| 2 | At most one call | 0.7434 | 173 | 70 | 0.359 |
| 2 | No branch feedback | 0.7477 | 182 | 68 | 0.482 |
| 3 | Full agent | 0.7620 | 169 | 59 | 0.461 |
| 3 | At most one call | 0.7616 | 170 | 61 | 0.423 |
| 3 | No branch feedback | 0.7578 | 167 | 68 | 0.623 |

### Three-seed summary (mean ± population SD)

| Condition | Accuracy | Delta vs full | Fixes | Harms | Mean calls |
|---|---:|---:|---:|---:|---:|
| Full agent | 0.7513 ± 0.0076 | — | 171.7 ± 6.8 | 63.3 ± 4.2 | 0.445 ± 0.028 |
| At most one call | 0.7487 ± 0.0092 | -0.26 pp | 166.0 ± 7.9 | 64.3 ± 4.0 | 0.403 ± 0.031 |
| No branch feedback | 0.7514 ± 0.0045 | +0.01 pp | 171.3 ± 7.6 | 62.7 ± 7.5 | 0.556 ± 0.058 |

### Paired contrasts against Full (clustered by source image)

Each source carries its six scenes and three seeds into every resample; 10,000 percentile bootstrap draws over the 430 sources.

| Contrast | Delta (pp) | 95% CI (pp) | Sources | Records |
|---|---:|---|---:|---:|
| At most one call − Full | -0.26 | [-0.45, -0.08] | 430 | 7740 |
| No branch feedback − Full | +0.01 | [-0.23, +0.26] | 430 | 7740 |

### Stop reasons (rerun subset only)

Reused records come from the frozen main run, which does not log a stop reason; they are excluded from this table rather than imputed. Denominators are the number of rerun records, not the full 2,580.

| Seed | Condition | n rerun / 2580 | Distribution |
|---:|---|---:|---|
| 1 | At most one call | 103 / 2580 | call_limit_ablation: 103 |
| 1 | No branch feedback | 1100 / 2580 | learned_stop: 1094, budget_exhausted: 6 |
| 2 | At most one call | 109 / 2580 | call_limit_ablation: 109 |
| 2 | No branch feedback | 926 / 2580 | learned_stop: 923, budget_exhausted: 3 |
| 3 | At most one call | 82 / 2580 | call_limit_ablation: 82 |
| 3 | No branch feedback | 1092 / 2580 | learned_stop: 1086, budget_exhausted: 6 |

### Exploratory: images on which the full agent called at least twice

Restricted to the 294 source-scene records (3.8%) where Full executed 2–4 tools. This is a post-hoc subgroup defined by the agent's own behaviour, not a pre-registered condition.

| Condition | Seed | n | Full acc | Ablated acc | Delta | Extra calls removed |
|---|---:|---:|---:|---:|---:|---:|
| At most one call | 1 | 103 | 0.6019 | 0.5049 | -9.71 pp | 1.04 |
| At most one call | 2 | 109 | 0.6606 | 0.5780 | -8.26 pp | 1.08 |
| At most one call | 3 | 82 | 0.5854 | 0.5732 | -1.22 pp | 1.20 |
| No branch feedback | 1 | 103 | 0.6019 | 0.6408 | +3.88 pp | 0.03 |
| No branch feedback | 2 | 109 | 0.6606 | 0.6606 | +0.00 pp | 0.27 |
| No branch feedback | 3 | 82 | 0.5854 | 0.5366 | -4.88 pp | -0.21 |

| Condition | Scene | n | Full acc | Ablated acc | Delta |
|---|---|---:|---:|---:|---:|
| At most one call | noise | 155 | 0.6194 | 0.5806 | -3.87 pp |
| At most one call | lowlight | 58 | 0.6552 | 0.5172 | -13.79 pp |
| At most one call | jpeg | 3 | 0.6667 | 0.6667 | +0.00 pp |
| At most one call | mixed | 78 | 0.5897 | 0.5128 | -7.69 pp |
| No branch feedback | noise | 155 | 0.6194 | 0.6323 | +1.29 pp |
| No branch feedback | lowlight | 58 | 0.6552 | 0.6552 | +0.00 pp |
| No branch feedback | jpeg | 3 | 0.6667 | 0.6667 | +0.00 pp |
| No branch feedback | mixed | 78 | 0.5897 | 0.5641 | -2.56 pp |
