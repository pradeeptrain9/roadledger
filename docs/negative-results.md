# Hypotheses tested and rejected

Kept deliberately, because a report that lists only what worked is not
evaluation evidence. Each of these cost real time and changed the design.

## 1. Perimeter-proportional edge-offset correction — REJECTED

**Hypothesis.** The refined area was biased consistently low (about -7%) across
every defect size and distance. A boundary placed d pixels inside the true edge
produces a shortfall of d x perimeter, so one least-squares fit over
(shortfall, perimeter) should recover d and remove the bias.

**Test.** Fitted d on 16 training scenes, applied to 16 held-out scenes.

**Result.** Fitted d came out at -0.238 px, and applying it made the holdout
*worse*: bias -5.2% to -8.5%, MAE 6.5% to 9.1%. An earlier fit on noisier
ground truth gave -0.084 px, also worsening the holdout. The bias is not
perimeter-proportional, so the model is wrong and the correction is not shipped.

**What it did reveal.** The bias was flat with distance (-12%, -9%, -11% across
3-5 m, 5-7 m, 7-9 m), which ruled out a resolution effect and pointed at the
renderer instead. See below.

## 2. Ground truth from directly rendered ellipses — REPLACED

**Found while chasing the bias above.** `cv2.ellipse` was being given
`int()`-truncated semi-axes at output resolution, which under-draws every
defect. Measured against the intended pi*r^2, the *renderer* was off by +13.5%
on average, and by +29.6% on one defect — larger than the pipeline error being
investigated. The ground truth was the problem, not the measurement.

**Fix.** Render at 4x resolution with rounded semi-axes, anti-aliased, then
area-average down. Renderer error fell to +2.5%, most of which is the
`g < 50` threshold used to count drawn pixels rather than render inaccuracy.

**Consequence.** Every accuracy number predating this fix was measured against
bad truth and was discarded. Post-fix holdout: precision 1.000, recall 0.962,
refined bias -5.2%, MAE 6.5%. The residual bias is of the same order as the
renderer's own measurement artifact, so the true pipeline bias is smaller than
-5% and is not separable from it with this harness.

## 3. CLAHE plus adaptive threshold inside the refinement ROI — REPLACED

**Symptom.** Refinement reported a 0.07 m2 defect as 0.23 m2 (+229%), with a
source footprint larger than the biggest defect in the scene.

**Cause.** Two compounding errors. The adaptive window (block_size 31) was
comparable to the ROI itself, so the operator degenerated and flooded the
region; and CLAHE on a small, low-contrast ROI amplified asphalt noise into
false structure. A watershed fallback that took "the largest non-background
region" when the seed missed then locked the error in.

**Fix.** Otsu inside the ROI, since an ROI is by construction one defect plus
its surround and therefore genuinely bimodal. The watershed now requires the
region to contain the seed rather than guessing. Adaptive thresholding stays in
the proposal pass, where it belongs and where the COOL claim lives. A growth
guard rejects any refinement exceeding 2.5x the proposal area.

**Result.** Max |err| fell from 105.8% to 22.2% on holdout. The growth guard
has not had to fire since the root cause was fixed, and is retained as a bound.

## 4. Hand-tuned classifier bands — REPLACED by a fitted model

Trapezoidal membership bands over the physical cues were hand-tuned to separate
pothole from shadow, wet stain and patch. On held-out confuser scenes they got
**3 of 12** right, and every adjustment was being fitted by hand to one
renderer's noise model.

Replaced with multinomial logistic regression over the same eight cues:
**0.980 five-fold cross-validated accuracy**, and **12/12** on seeds outside the
fitting range. The cue extraction was kept - it encodes physics and transfers.
Only the decision boundary was fitted.

The fitted weights recover the physics without being told it: `edge_width_px`
carries the largest positive weight for shadow (+3.31), `texture_ratio` the
largest negative for wet stain (-1.5), `darkness_ratio` strongly negative for
pothole and positive for patch.

### 4a. "Shadows preserve surface texture" — WRONG AS STATED

The first texture cue assumed a shadow darkens a surface without changing its
roughness, so a contrast-normalised texture ratio near 1 would identify one.
Measured, a rendered shadow came back at 1.42, not 1.0.

The reason is physical and was missed: sensor read noise is **additive and
applied after shading**, so a darker region has a *higher* relative roughness.
Normalising by the local mean removes the illumination factor but not this. The
cue cannot separate a smooth wet patch from a shadow, and saying it could was
wrong. What does separate them is `edge_width_px`: the sun is not a point
source, so a cast shadow has a penumbra and ramps over several pixels, where a
pothole rim is a depth discontinuity and steps in one or two.

The synthetic renderer was also applying noise *before* shading, which hid this.
Fixed to render in float and add sensor noise last.

## 5. Ego-motion from the whole bird's-eye view — WRONG TWICE

Needed to follow one defect across frames. Three attempts:

1. **Lucas-Kanade on the defect's own centroid.** Failed: a pothole interior is
   a smooth dark region with no gradient structure. Followed the defect into
   **1** frame of 8, leaving the spread undefined and the entire multi-frame
   argument inert.
2. **Global phase correlation over the bird's-eye views.** Correct on 6 of 13
   frame pairs, collapsed on the rest (response 0.003-0.035, forward motion
   essentially zero against a 0.60 m truth).
3. **Lucas-Kanade on road-texture features with a median flow.** *Worse*: 3 of
   13. The hypothesis that many features beat one global peak was wrong.

All three failed for one reason, found only by brute-force correlation scanning
and then by projecting a fixed ground point through both homographies to
establish that the true shift was 28.9 px while the estimators reported 16:
**lane markings dominate the view and do not move along the direction of
travel.** Driving along a solid line looks exactly like standing still, and the
markings are the brightest, highest-contrast, most trackable thing in frame. A
median over inliers does not save you when the confuser is most of the signal.

**Fix:** exclude lane paint and its halo from the ego-motion estimate and use
only asphalt aggregate - lower contrast, but the only thing in the frame that
actually moves with the road. Result: **12 of 13** frame pairs within 0.1 m of
truth.

Two wrong diagnoses were discarded along the way, both checked and refuted
rather than assumed: that per-frame homography jitter was breaking the
translation assumption (lane-width std was 0.65 px and the scale constant), and
that the bird's-eye depth scale was wrong (verified correct to 3.7%, and linear
to within 1.7 px, by round-tripping known ground points).

## 6. Unconditional watershed in refinement — now conditional

A pothole interior is dark *and rough*, so Otsu leaves speckle holes that
fragment it. On one near-field 0.50 m2 defect the mask broke into **seven**
components and the largest held 64% of the true area.

Worse, watershed ran unconditionally on that fragmented mask: the distance
transform seeded every speckle, watershed split the defect into scraps, and the
seed pixel landed in a five-pixel one. The reported area was **0.003 m2 for a
0.50 m2 pothole - a 99.3% error.**

**Fix:** morphological closing at 9 px after opening, to reconnect the interior
(largest component 64% to 95.5% of truth), and watershed only when the seeded
region exceeds 1.6x the proposal, which is the only situation it was ever for -
two defects that touch.

**End-to-end effect** across a 15-run sweep: bias -37.9% to **-14.4%**,
measured 3/6 to **14/15**, and multi-frame agreement became meaningful
(`iqr_ratio` 0.014-0.216 where it had been undefined).

## 7. Retuning the closing kernel for small defects — rejected on held-out data

The live path over-measured small, distant defects badly: a 0.21 m-radius
pothole (0.139 m2) came back at 0.251 m2, **+81%**, while a 0.34 m one in the
same drive came back 12% *low*. The closing kernel was the obvious suspect,
since 9 px is a large fraction of a 19 px-radius blob and closing can bridge it
outward into surrounding dark texture.

A development grid of five radii at one lateral offset and start distance
looked conclusive:

| kernel | 0.15 m | 0.21 m | 0.28 m | 0.34 m | 0.42 m | MAE |
|---|---|---|---|---|---|---|
| 3 | -38.8% | -32.9% | -8.0% | -5.2% | -1.9% | 17.4% |
| 5 | -6.2% | +15.5% | -6.1% | -4.6% | -1.6% | 6.8% |
| **7** | -11.4% | +9.8% | -4.4% | -4.0% | -1.5% | **6.2%** |
| 9 (current) | +42.0% | +58.7% | +1.9% | -4.0% | +4.8% | 22.3% |
| 13 | -1.7% | +63.8% | +13.1% | -3.7% | +11.5% | 18.8% |

A 3.5x improvement in MAE. It did not replicate. On a held-out grid of 15
drives over five different radii, three lateral offsets, three start distances
and three unused noise seeds:

| kernel | bias | MAE | worst | confirmed |
|---|---|---|---|---|
| 5 | -13.9% | 14.7% | 55.2% | 9/12 |
| 7 | -4.7% | 18.1% | 67.1% | 11/12 |
| **9 (kept)** | **-0.7%** | **18.5%** | **56.3%** | **11/12** |
| 11 | +18.3% | 40.8% | 93.5% | 11/12 |

7 and 9 are 0.4 points apart on 12 samples - inside the noise - and 9 has both
the better bias and the better worst case. **The kernel was left at 9.**

Two things worth taking from this beyond the kernel itself.

First, this is the third tuning change in this project to look strong on a
development grid and fail on held-out data, after the perimeter-proportional
bias correction (twice). The pattern is consistent enough to be the finding:
per-configuration variance on the synthetic rig is large compared with the
effects being chased, so a sweep that varies only one factor will keep
producing false positives.

Second, the held-out numbers are the honest state of the measurement on hard
cases - **bias -0.7%, MAE 18.5%, worst case 56%** on small and distant defects,
against 6.5% MAE on the easier holdout in `accuracy.md`. No further synthetic
tuning is planned. The binding requirement now is real footage with
hand-measured areas, because the remaining question - whether the error is the
algorithm or the rig's own projection chain - cannot be answered by the rig
that is under suspicion.
