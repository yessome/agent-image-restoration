**Table 6. Component ablation of the iterative scheduler (development split valB, 328 images, three paired seeds).**
Each row changes exactly one factor; router weights, tool weights, classifier, and thresholds are held fixed.
The configuration is the 13-tool subset (raw + 8 denoisers + 4 low-light tools).

| Variant | Accuracy | SD | Δ vs full | Fixes | Harms | Mean calls |
|---|---:|---:|---:|---:|---:|---:|
| Full iterative agent | 0.7917 | 0.0141 | +0.0000 | 37.7 | 6.7 | 0.76 |
| No branch feedback | 0.7917 | 0.0047 | +0.0000 | 38.0 | 7.0 | 1.01 |
| At most one call | 0.7866 | 0.0152 | -0.0051 | 36.7 | 7.3 | 0.69 |
| Diagnostic rule (single tool) | 0.7398 | 0.0168 | -0.0518 | 25.3 | 11.3 | 0.62 |

Removing the branch feedback that the network observes leaves accuracy unchanged (0.7917 versus 0.7917,
delta +0.0000) while raising the mean number of restoration calls from 0.76 to 1.01. The feedback therefore
governs when to stop rather than which branch to return. Capping execution at a single call lowers accuracy by
0.51 percentage points (0.7866), so the current gains come from choosing the first tool and stopping, not from
multi-round composition. A single-tool diagnostic rule reaches 0.7398 with 0.62 calls, 5.18 points below the
full agent at a comparable call budget.
