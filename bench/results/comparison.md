| measurement | ? (5.0.0) | B (4.12.0) | C (5.0.0) | D (4.12.0) |
|---|---|---|---|---|
| resize | 0.016 ±0.002 | 0.015 ±0.001 | 0.011 ±0.000 | 0.013 ±0.001 |
| adaptiveThreshold | 0.674 ±0.013 | 0.935 ±0.001 | 0.976 ±0.001 | 0.875 ±0.001 |
| findContours | 0.891 ±0.294 | 1.035 ±0.003 | 1.074 ±0.001 | 1.018 ±0.001 |
| proposal_pass | 3.404 ±0.066 | 3.751 ±0.006 | 3.600 ±0.004 | 3.673 ±0.007 |
| full_analysis | 45.980 ±2.799 | 58.513 ±0.120 | 49.239 ±0.049 | 59.098 ±0.214 |

- **D over B** — COOL against the vendor's own stated baseline (stock 4.12.0, no KleidiCV): **1.02x** on the core pass.
- **D over C** — COOL against the free pip wheel, which already bundles KleidiCV 26.03: **0.98x** on the core pass.
- **C over A**: not measured.
- **C over B** — what the wheel's bundled KleidiCV alone is worth on the same silicon: **1.04x** on the core pass.
