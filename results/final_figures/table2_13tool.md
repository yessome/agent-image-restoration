**Table 2. Mean accuracy over three paired classifier seeds (test split, 13-tool configuration).**
Values are fractions; parenthesized values are population SDs across the three seeds.
All columns come from the same 13-tool configuration and the same frozen classifier seeds.

Scale: the test split holds 430 unique source images. Each of the six scenes degrades those same 430 sources, giving 6 x 430 = 2,580 scene instances per seed; with seven methods per instance this yields 18,060 method records per seed (54,180 over three seeds).

| Scene | Raw | Fixed denoise | Fixed low light | Diagnostic rule | Legacy rule agent | Learned iterative agent | Agent − raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.8217 (0.0071) | 0.7744 (0.0190) | 0.7357 (0.0251) | 0.8209 (0.0062) | 0.8209 (0.0070) | 0.8225 (0.0082) | +0.0008 |
| Noise, sigma 25 | 0.6620 (0.0256) | 0.7457 (0.0181) | 0.6062 (0.0401) | 0.7457 (0.0181) | 0.7349 (0.0326) | 0.7837 (0.0207) | +0.1217 |
| Low light, gamma 3.0 | 0.7178 (0.0329) | 0.6628 (0.0163) | 0.7806 (0.0097) | 0.7403 (0.0259) | 0.7256 (0.0304) | 0.7760 (0.0249) | +0.0581 |
| Motion blur | 0.6760 (0.0233) | 0.6667 (0.0210) | 0.6295 (0.0132) | 0.6760 (0.0233) | 0.6760 (0.0233) | 0.6760 (0.0233) | +0.0000 |
| JPEG, quality 30 | 0.7992 (0.0082) | 0.7752 (0.0249) | 0.7209 (0.0203) | 0.8000 (0.0070) | 0.7984 (0.0082) | 0.8023 (0.0084) | +0.0031 |
| Mixed, gamma 3.0 then sigma 25 | 0.5791 (0.0304) | 0.6364 (0.0312) | 0.6093 (0.0203) | 0.6364 (0.0312) | 0.6295 (0.0223) | 0.6473 (0.0132) | +0.0682 |

Agent − raw: Clean +0.0008; Noise, sigma 25 +0.1217; Low light, gamma 3.0 +0.0581; Motion blur +0.0000; JPEG, quality 30 +0.0031; Mixed, gamma 3.0 then sigma 25 +0.0682.

Six-scene mean: raw 0.7093, learned iterative agent 0.7513 (+0.0420).
