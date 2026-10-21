# Measurement accuracy

Measured on simulated drive-pasts over a persistent ground plane, where each
defect is a true circle of known radius, so its area is exactly pi*r^2. 15 runs:
five defect sizes (r = 0.20 to 0.45 m) across three surface seeds, each a
14-frame approach from 11 m at 0.6 m per frame.

## Headline

| Metric | Value |
|---|---|
| Detected | 15/15 |
| Measured (survived the resolution and vantage gates) | 14/15 |
| Area bias | -14.4% |
| Area MAE | 15.8% |
| Median absolute error | 14.3% |
| Max absolute error | 35.3% |

## The result that matters: the confidence signal is calibrated

The system reports an area only with a spread (`iqr_ratio`, the interquartile
range of independent per-frame measurements over their median) and escalates to
human review when that spread, the class confidence or the measurement count
falls short. Splitting the same 14 measurements by what the system decided
about itself:

| | n | MAE | Max abs error |
|---|---|---|---|
| Passed without escalation | 5 | **11.1%** | **14.4%** |
| Escalated for human review | 9 | 18.4% | 35.3% |

The cases it flags really are the less accurate ones. This is what makes the
agentic loop more than decoration: the spread is a measured quantity that
correctly predicts the system's own error, so acting on it - gathering more
views, or escalating - is a decision grounded in evidence rather than in a
confidence number a language model invented.

It is also deliberately conservative. Only 5 of 14 measurements passed
unescalated, and priority-1 defects always require approval regardless. For
public spending on road repair that is the correct direction to err.

## Known limitations

* **A residual -14% bias**, varying with defect size (-7% at r=0.45 to -16.6%
  at r=0.32) rather than constant, so it is not corrected by a single factor. A
  perimeter-proportional correction was tried and rejected on held-out data
  (see negative-results.md). It must be re-characterised on real imagery.
* **Small defects are over-measured and correctly distrusted.** At r=0.15 the
  error reached +48%, with too few comparable views to establish a spread, so
  the defect escalated rather than being reported.
* **Below a 300 px source footprint nothing is measured at all.** Calibrated
  against ground truth: at 231 px the error was 39.5%, at 572 px it was 12.4%.
  Defects too distant or too small are refused rather than given a confident
  wrong number.
* **All of these numbers come from synthetic imagery.** The renderer is
  physically constructed - a real pinhole camera over a persistent ground
  plane, shading before sensor noise, penumbra on cast shadows - and its own
  ground-truth error was measured at +2.5%, but it is not road footage. Every
  figure here must be re-established on RDD2022 before it is claimed of real
  roads. The classifier in particular is fitted on synthetic cues and its
  0.98 accuracy should be read as "the cues separate these classes", not as a
  claim about Indian or Japanese road surfaces.
