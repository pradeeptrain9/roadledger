# RDD2022: what happens on real roads

Every figure in [accuracy.md](accuracy.md) comes from the synthetic rig. This
file is what happened when the pipeline met 300 real Indian road images from
RDD2022, and it is the most useful evidence in the project — because most of it
is bad.

Dataset: RDD2022 (CC BY 4.0), India subset, 300 images sampled as 12 contiguous
blocks spread through the capture order. Already privacy-scrubbed by its
authors. Cite Arya et al., *Geoscience Data Journal* 11(4), 2024.

---

## 1. Rectification: 2.3% → 100%, by abandoning lane markings

The original rectification found the vanishing point from converging lane
markings. On real Indian roads it produced a usable bird's-eye view on **4 of
175 frames — 2.3%**.

| Failure | Frames |
|---|---|
| no vanishing point: too few lane-like segments | 113 |
| no lane pair at frame base | 48 |
| implausible vanishing point | 10 |

Unmarked and worn roads have no converging lane lines to find. Since every
later stage measures in the rectified frame, this failure took the whole
measurement chain with it. A 6.5% area error on synthetic data is worth nothing
if 97.7% of real frames never reach the measurement.

**The fix was structural.** Lane markings were only ever a way to *estimate*
the road-plane geometry from an uncalibrated camera. For a camera of known
height, field of view and tilt, the homography follows directly: the horizon is
fixed by the pitch, and a ground point at lateral offset `u`, distance `v`
images at `x = cx + f·u/v`, `y = y_horizon + f·h/v`, which is a homography.

Adding `rectify_calibrated()` took applicability from **2.3% to 100%** (195 of
200 frames by calibration, 5 still by lane markings where they existed).

This is a better architecture, not a workaround. On any camera we control — a
mounted dashcam, a phone on a bracket — height and field of view are known
exactly, so calibration is both available and more accurate than inferring
geometry from paint. Lane markings are retained as an independent cross-check
when present, which is the only way the scale gets corroborated at all.

**Caveat:** the calibration used here is *assumed* (1.25 m, 55°, 5° down), not
measured, because RDD2022 is someone else's dashcam. Applicability and
detection are meaningful under an assumed calibration; **areas are not**, and
none are reported from these frames.

## 2. Coverage is not the problem

Before blaming detection, the geometry was checked: of 172 annotated defects
placed by calibration, **167 (97.1%) fall inside the bird's-eye band**, which
covers roughly 2.1–9.0 m ahead and 7.0 m across. Median defect distance is
3.1 m. Only 5 were beyond the far edge, none too close, none outside the
lateral width.

So the misses that follow are detection failures, not framing failures.

## 3. Detection: 42.9% recall, and it depends entirely on damage type

After tuning the proposal pass on real frames (`block_size` 51, `C` 3.0,
`min_area` 180 — recall 28.3% → 42.9%):

| Damage type | Found | Total | Recall |
|---|---|---|---|
| Alligator cracking (D20) | 62 | 99 | **0.626** |
| Pothole (D40) | 45 | 131 | 0.344 |
| Longitudinal crack (D00) | 9 | 43 | 0.209 |
| Transverse crack (D10) | 2 | 2 | 1.000 |
| **Overall** | **118** | **275** | **0.429** |

The pattern makes sense. The proposal pass finds regions that are *darker than
their surroundings over an area*, so it does best on broad alligator cracking,
worse on potholes (which on dusty Indian roads are often no darker than the
surface), and worst on thin longitudinal cracks, which have almost no area.

## 3a. A second country, and the hypothesis that turned out wrong

The expectation was that this would work better on marked European roads. It
does not — it is worse, and the reason is more useful than the guess was.

| | India (300 images) | Czech (175 images) |
|---|---|---|
| Lane-marking rectification | 2.5% | **1.7%** |
| Detection recall (calibrated) | 0.429 | **0.092** |

Lane-marking rectification fails on both, so that failure is not about road
markings at all — these are 600–720 px square smartphone crops in which the
near-field road band and its markings do not present the way the Hough
parameters expect.

And the recall gap is not about country, it is about **damage type**:

| Damage type | India | Czech | Instances (both) |
|---|---|---|---|
| Alligator cracking (area-like) | 0.626 | 0.000 | 105 |
| Pothole (area-like) | 0.344 | 0.133 | 146 |
| Longitudinal crack (linear) | 0.209 | 0.113 | 123 |
| Transverse crack (linear) | 1.000 (n=2) | 0.000 | 39 |

Czech damage is 85% linear cracking; India's is far more area-like. The
proposal pass finds regions that are *darker than their surroundings over an
area*, so it performs on the second kind and fails on the first.

This was checked rather than assumed. Relaxing the geometry filters that might
have excluded long thin shapes — `max_aspect_ratio` 8 → 40, `min_solidity`
0.35 → 0.12 — moved Czech recall only from 0.080 to 0.101. The filters were not
the cause: a 1–3 px crack simply does not carry enough contrast at bird's-eye
resolution for an area threshold to find.

**This is a coherent scope, not just a shortfall.** The cost model prices
repair at a rate per square metre, which is how patching is quoted — and that
only makes sense for area-type damage. Linear cracks are sealed and priced per
metre, which would need a length measurement and a different rate. The
detection gap and the cost model are consistent with each other: this is a
system for potholes and surface deterioration, not a crack detector.

## 4. Classification: the synthetic-fitted boundary does not transfer

The classifier fitted on synthetic cues scored 0.937 there. On real detected
defects it labelled **35 of 45 potholes as "shadow"**, and 2 as pothole.

Refitting on real cues, with labels taken from the annotations by weak
supervision (inside a D40 box → pothole, inside D00/D10/D20 → crack, far from
every annotation → not damage), five-fold cross-validated **grouped by source
image** so candidates from one frame cannot leak across folds:

| | |
|---|---|
| Damage recall | 0.811 |
| **Damage precision** | **0.314** |
| F1 | 0.453 |
| Accuracy | 0.588 |
| Majority-class baseline accuracy | **0.790** |

**This is a failure and should be read as one.** Accuracy is below the baseline
of predicting "not damage" for everything, and a precision of 0.31 means two in
three dispatches would be to nothing — the exact error the whole system was
designed to avoid.

Two honest caveats, neither of which rescues it:

* The negative label is weak. RDD2022 annotates prominent damage, so an
  unannotated region is *unlabelled*, not verified sound. Some "not damage"
  samples are probably real damage, which depresses precision artificially. A
  60 px ambiguity margin around every annotation reduces this but cannot remove
  it.
* The fit is class-weighted against a 79% majority, which trades precision for
  recall by construction.

Even allowing for both, the eight hand-crafted cues do not separate real road
damage from real road clutter — tar seams, stains, tyre marks, vehicle shadows,
patched repairs and wet patches — anywhere near well enough to spend money on.

## 5. What this means for the system

The honest conclusion: **detection and classification are the weak link, and
they are the replaceable part.**

What survived contact with real data:

* **Rectification**, once it stopped depending on lane markings — 100%
  applicability, and better posed than before.
* **Coverage**, verified rather than assumed — 97.1% of real defects fall in
  the measured band.
* **The measurement, aggregation and decision layers**, which are independent
  of how a candidate was proposed. The spread across views, the vantage filter,
  the resolution floor, the deterministic costing and the approval gate do not
  care whether a candidate came from an adaptive threshold or a neural network.

What did not:

* The classical proposal pass as a *detector* for real roads: 0.429 recall in
  India, 0.092 in Czechia, and essentially zero on fine linear cracking.
* The cue classifier as a *discriminator* for real roads: 0.314 precision,
  below the majority-class baseline.

The architectural implication is that the proposal stage should be pluggable,
with the classical COOL-accelerated pass as one implementation and a trained
detector as another. That is also a tension worth naming: the COOL claim rests
on the classical pass being the core per-frame workload, and swapping in a CNN
detector would move that workload onto the model. Both can be reported — the
classical pass is what COOL accelerates and what is benchmarked; a detector
would be what makes the product deployable on Indian roads.

## 6. What was done about it

The proposal stage is now an interface (`pipeline/proposers.py`) with two
implementations:

* `ClassicalProposer` — resize, adaptive threshold, contours. The default, and
  the workload the COOL benchmark measures, because those three operations are
  exactly what COOL accelerates on Graviton.
* `DetectorProposer` — a trained detector supplies regions; the same Otsu
  segmentation turns each into a measurable contour. The model runs through
  OpenCV's own DNN module, so it adds no runtime dependency and on the Graviton
  worker runs under the COOL build.

A detection is treated as a **region of interest, never as the defect's
extent**. A bounding box around an irregular pothole overstates its area badly,
and area is the number this system exists to get right, so the box only selects
where to segment.

The path is validated end to end with a stubbed detector (see
`test_detector_proposed_candidates_flow_through_unchanged`): a box becomes a
candidate measured at −6.7% of truth, classified, tracked across three frames
to an `iqr_ratio` of 0.019, and scored — with nothing downstream changed.

**The real model is not yet integrated.** No ONNX road-damage detector is
published; the available RDD2022 weights are PyTorch, and exporting them needs
torch, which does not fit in the free disk on this machine. The export is a
single command and is scheduled for the Graviton worker, which has room:

```bash
pip install ultralytics && yolo export model=YOLOv8_Small_RDD.pt format=onnx imgsz=640
```

**The tension this creates is real and is not papered over.** The Best Use of
COOL claim rests on COOL accelerating the *core* image workload, which the
classical pass is. Moving proposals onto a neural network moves a large part of
that workload onto the model. Both are therefore reported: the classical pass
is what COOL accelerates and what the benchmark measures, and the detector is
what would make the product deployable on real roads. They are different
claims about different things.

Nothing in this document should be read as a claim that this system is ready to
direct real road spending.


---

# Real phone footage: a 69-second clip of an Indian street

RDD2022 is stills. This is the first real *video* the pipeline has seen:
848x478, 28 fps, 69 s, a concrete residential street with parked vehicles,
an autorickshaw, a manhole cover, surface cracking and heavy dappled tree
shadow. Captured 2026-10-22 on a phone.

**No area is reported from it, and none can be.** Two independent reasons, both
fatal to metric scale and neither fixable in post:

1. **The camera was hand-held.** Every area depends on a camera whose height
   and pitch are constant. Hand-held, both change from frame to frame, so no
   calibration can be valid for more than an instant.
2. **It is uncalibrated.** No measured lens height, no markers at known
   distances. Area scales as the square of height, so a guessed height is a
   guessed area.

The clip was still worth running, because the half of the system that does not
depend on calibration is exactly the half that failed on RDD2022.

## What it establishes

| | |
|---|---|
| Frames sampled | 60 of 60 usable (sharpness gate) |
| **Rectification applicability** | **100%**, quality 0.80, calibrated route |
| Candidates proposed | 368, **6.1 per frame** |
| Measurable | **84 (23%)** |
| Median source footprint | 496 px |

**Rectification is solved on real roads.** The lane-marking route managed 2.3%
on RDD2022 and took the whole measurement chain down with it. The calibrated
route recovers the road plane on **every frame** of real footage, on a concrete
street with no lane markings at all. That failure is closed.

**Three quarters of candidates are refused, every one for a stated reason:**

| count | reason |
|---|---|
| 193 | too distant: source footprint under the 300 px measurable floor |
| 52 | refinement did not retain the candidate |
| 39 | clipped: boundary cut off by the edge of the rectified view |

This is the result, not a disappointment attached to it. A measurement system
that sizes everything it sees is not trustworthy; one that declines 77% of the
time with a reason per refusal is. The clipping guard in particular was added
only a week earlier, after the live path exposed a view-edge truncation
reporting a 0.363 m2 defect as 0.087 m2 - and on real footage it fires 39 times
in 60 frames.

## What it exposes

**The proposal stage outlines tree shadows.** Visible directly in
`data/artifacts/real_frame_015.png`: the largest contour in the frame is
dappled shade, not damage. This is the same weakness RDD2022 measured - the
classifier called 35 of 45 real potholes "shadow" - seen from the other
direction. The proposal pass is a brightness-and-geometry filter and shade on
concrete satisfies it. Nothing downstream rescues a candidate that was never a
defect; it is the detector backend in `pipeline/proposers.py` that has to.

**The plate scrubber fires far too often.** 463 "plates" across 60 frames,
about eight per frame, on a street with perhaps three vehicles in it. Dappled
shadow on concrete produces bright high-contrast horizontal rectangles, which
is precisely what the heuristic looks for.

It was then checked rather than assumed, because an over-firing privacy filter
once destroyed this pipeline's own input (negative-results.md #12, rectification
quality 0.94 to 0.00). Same 30 frames, scrubber on and off:

| | plates blurred | candidates | measurable |
|---|---|---|---|
| privacy ON | 214 | 158 | 31 (20%) |
| privacy OFF | 0 | 164 | 30 (18%) |

A 4% difference in candidates and one measurement. **The over-firing is
cosmetic, not corrupting** - the `plate_max_area_frac` and
`plate_max_width_frac` bounds added after that earlier failure are holding each
false positive small enough not to matter. Over-inclusive is the right
direction for a privacy filter; it is bounded over-inclusion that makes it
safe.

## What would make this clip measure

Nothing in software. A mount, a tape measure, and two markers:

1. Clamp the phone so it cannot move, and leave it clamped.
2. Measure the lens height above the road with a tape.
3. Place objects at 3.00 m and 6.00 m ahead, film ten seconds, remove them.
4. `python -m scripts.calibrate --video calibration.mp4 --grid`
5. `python -m eval.real_video --video drive.mp4 --camera-height H --vfov V --pitch P`

Supplying all three promotes every area in the output from arithmetic to
measurement. Until then `eval/real_video.py` prints them under an explicit
warning and the report claims nothing from them.
