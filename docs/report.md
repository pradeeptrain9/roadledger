# RoadLedger — technical report

OpenCV AI Competition 2026, powered by AWS.

---

## 1. The problem, and who has it

A municipal road authority does not need to be told a pothole exists. Its
inspectors already know. What it cannot produce cheaply is the thing that
actually governs spending: **a ranked list of defects with quantities** — how
large each one is, how urgent, what it will cost to repair, and how confident
anyone should be in those figures.

Pothole *detection* is a crowded space and stops exactly where the useful work
begins. A bounding box does not tell a crew how much asphalt to load, does not
let a district compare two wards, and does not survive the question "how do you
know?". Meanwhile the single most expensive error is the one detection-only
systems make most often: dispatching a crew to a **shadow**.

RoadLedger takes dashcam video and produces a costed, prioritised work order
where every area is a measurement in square metres carrying its own
uncertainty, and anything urgent is held for a human.

**Users.** Municipal and district road maintenance teams; the engineer who
signs off the repair schedule; the auditor who asks later why a particular
stretch was prioritised.

---

## 2. Architecture

![architecture](architecture.svg)

Video lands in **S3**; **EventBridge** triggers a run; a **Lambda** (arm64)
orchestrates it and starts an **ephemeral Graviton worker** running the COOL
AMI. The worker serves the OpenCV pipeline as **MCP tools**, which the agent
calls. Results — defects, work orders and the full decision trace — go to
**DynamoDB**, and the reviewer-facing endpoint is served from **CloudFront**.
**Bedrock** provides the optional language-model driver. **CloudWatch** carries
tool calls, timings and rule firings.

The worker is started per run and stopped afterwards, because leaving it up
would spend the budget on nothing. The endpoint therefore answers from cached
results: a demo that only works while an instance is burning is not a demo.

---

## 3. The OpenCV 5 implementation

Real algorithmic work at every stage — not a thin wrapper around a neural net.
The only learned component is an eight-feature logistic regression over
hand-derived physical cues.

### 3.1 Rectification, and two routes to metric scale

Measuring in image pixels is meaningless: the same pothole covers four times
the pixels at half the distance. So the road plane is rectified to a metric
bird's-eye view first.

Lane markings are found with `HoughLinesP` over a Canny edge map restricted to
the near-field road band, filtered to a plausible slope range, and their
vanishing point recovered by RANSAC followed by a weighted least-squares fit.
The lane trapezoid then maps to an upright rectangle through
`getPerspectiveTransform` and `warpPerspective`.

Scale arrives by two independent routes, which is the point:

1. **A lane-width prior.** The dominant markings' separation at the frame base
   corresponds to a known lane width. Needs nothing but the image, and fixes
   the *lateral* scale only.
2. **Camera geometry.** Given camera height and vertical field of view,
   flat-road pinhole geometry fixes *both* scales exactly.

When both are available they are **cross-checked and the disagreement
reported** rather than averaged away. On a consistent synthetic scene they
agree within 1%; given a deliberately wrong camera height they disagree by 94%
and say so. With only the prior, the depth scale is an explicit assumption, the
quality score is capped at 0.75, and the agent can see it is on softer ground.

Measured against known geometry: vanishing point within **0.4 px**, lane width
within **0.6%**, and the bird's-eye depth axis linear in ground distance to
within 1.7 px.

### 3.2 The core pass — and why it is built this way

Candidate proposal runs downscaled on every frame and is composed of exactly
the three operations COOL accelerates on Graviton: **`resize`**,
**`adaptiveThreshold`** (Gaussian), **`findContours`**. That is deliberate. The
COOL award asks for evidence that COOL executes the claimed core workload, and
the only way to make that true rather than arguable is to build the core
workload out of those operations.

Its job is recall, not precision: anything it discards is never measured, while
a false positive costs only one refinement.

One trap had to be handled explicitly. `adaptiveThreshold` compares each pixel
to a Gaussian mean of its neighbourhood, so beside a bright lane marking the
local mean is pulled up and *ordinary asphalt* within a window radius reads as
dark. Masking the paint does not help, because the artifact is outside it. The
paint is therefore dilated by the adaptive window radius before exclusion.
Without this the pass returned two 7 m² "defects" hugging the lane markings,
each larger than any real pothole in the scene.

### 3.3 Refinement

Nothing is measured from the cheap pass, which under-measures by about 7%.
Each surviving candidate is re-examined at full bird's-eye resolution inside
its own ROI: CLAHE for local contrast, **Otsu** thresholding (an ROI is by
construction one defect plus its surround, which is genuinely bimodal),
morphological opening then **closing**, the connected component under the seed,
and contour moments for area, perimeter and convexity.

Two guards earn their place. Watershed runs **only** when the seeded region
exceeds 1.6× the proposal, which is the sole situation it was ever for — two
defects that touch. And a refinement that grows beyond 2.5× the proposal is
rejected in favour of the proposal, which is biased but bounded. Both exist
because of a measured failure, described in §7.

### 3.4 Classification: separating damage from what looks like it

The discriminating physics:

* A **shadow** darkens a surface without changing it. Its edge is a *penumbra*,
  because the sun is not a point source, and ramps over several pixels.
* A **pothole** is a cavity. Its rim is a depth discontinuity and images as a
  step one to three pixels wide; its interior is rougher than the road.
* **Standing water** is darker *and smoother* — specular, filling surface
  texture rather than merely dimming it.

The decisive cue is therefore **edge transition width**, recovered as contrast
divided by boundary gradient without sampling intensity profiles along normals.
Eight cues in total feed a multinomial logistic regression (`eval/fit_classifier.py`).

The fitted weights recover the physics unprompted: `edge_width_px` carries the
largest positive weight for shadow, `texture_ratio` the largest negative for wet
stain, `darkness_ratio` strongly negative for pothole.

| | |
|---|---|
| Five-fold cross-validated accuracy | 0.937 |
| **Binary damage-vs-not accuracy** | **0.9873** |
| False "damage" (crew dispatched to nothing) | 2 in 314 |
| Missed damage | 2 in 314 |

The binary figure is the operational one. Confusing a shadow for a wet stain
changes nothing a crew does; confusing either for a pothole spends public money
on nothing.

### 3.5 Multi-frame measurement — the quantity the agent acts on

One measurement is a number. Several independent measurements of the same
defect, from different distances, are a number **and a spread** — and the
spread is what tells the system whether to trust itself.

A defect is followed across frames by estimating ego-motion in the bird's-eye
view, where forward driving is a near-pure translation. Measurements are then
restricted to **comparable vantage points**: footprint falls as the square of
distance, so observations from the far field rest on a few dozen source pixels
while the closest rest on thousands. Pooling them is not averaging, it is
diluting.

`iqr_ratio` — interquartile range over median — is the most important output of
the pipeline. It is a measured quantity that correctly predicts the system's own
error (§5), which is what makes acting on it legitimate.

### 3.6 Deterministic scoring

Severity, priority, cost and the approval gate come from a published rate table
in `pipeline/score.py`. No model influences any of them, so two runs on the same
measurements always produce the same work order and an engineer can audit the
arithmetic. The gate **fails closed**: a defect with no trustworthy measurement
does not become a cheap work order, it becomes a request for attention.

---

## 4. The agent loop

![agent workflow](agent-workflow.svg)

Each decision thresholds a quantity the vision stages produced, and changes
which tool is called next:

| Observed | Rule | Next action |
|---|---|---|
| rectification quality < 0.35 | `rectification_unusable` | re-rectify over a wider band; still low → skip the frame |
| > 25 candidates in a frame | `too_many_candidates` | raise `block_size`/`C`, re-propose rather than refine noise |
| class confidence < 0.55 | `class_ambiguous` | re-examine with more contrast; still low → escalate |
| source footprint < 300 px | `below_resolution_floor` | defer: report no area at all |
| `iqr_ratio` > 0.20 | `measurement_spread_too_wide` | gather more views (k: 4 → 10) |
| `iqr_ratio` > 0.35 after that | `spread_unresolved_after_more_views` | escalate to human review |
| priority 1 | `work_order_requires_approval` | hold for approval, however confident |
| same ground position as a recorded defect | `duplicate_of_recorded_defect` | merge: one pothole is one work order |

A representative run over four features (two potholes, a shadow, a wet stain)
fired **26 rules across 40 tool calls**, producing 3 defects from 18 candidate
sightings — 15 merged as duplicates — with 6 escalations.

**Vision output changing a later action** is visible directly in the trace: a
candidate whose four views disagree by 24% triggers `track_across_frames` with
k=10; one whose footprint is 23 px is deferred with no area reported; one whose
class confidence is 0.541 is re-examined at higher contrast and, still
ambiguous, escalated.

**Division of labour.** The model (or the policy) decides *which measurement to
take next*. The engine computes every number. A model that hallucinates wastes
tool calls, which the budget bounds; it cannot produce a wrong measurement.

**Human control.** Priority-1 defects always require approval. The web endpoint
records who approved what and when, and a rejection is recorded too — a rejected
priority-1 defect is evidence the system was wrong about something, which is
worth keeping.

**Observability.** Every step records the tool, its arguments, the observation,
the rule that fired, the measured values that fired it, and the elapsed time.
The trace replays: each step carries the parameters used.

---

## 5. Evaluation

Full detail in [accuracy.md](accuracy.md); the design of the ground truth is in
§7.

| Metric | Value |
|---|---|
| Detected | 15/15 |
| Measured (survived the resolution and vantage gates) | 14/15 |
| Area bias | −14.4% |
| Area MAE | 15.8% |
| Max absolute error | 35.3% |

**The result that matters.** Splitting those measurements by what the system
decided about itself:

| | n | MAE | Max abs error |
|---|---|---|---|
| Passed without escalation | 5 | **11.1%** | **14.4%** |
| Escalated for human review | 9 | 18.4% | 35.3% |

The cases it flags really are the less accurate ones. The spread is not a
decoration: it predicts the error. It is also deliberately conservative — only
5 of 14 passed unescalated, which for public spending is the correct direction
to err.

Supporting measurements: ego-motion within 0.1 m on **12 of 13** frame pairs;
defect ground position within **0.15 m**; the same defect assigned a consistent
station across six frames to **±3 cm**.

Correctness is held by **21 invariant tests** (`tests/test_pipeline.py`), most
of which encode a bug that was actually found: that privacy scrubbing must not
degrade rectification, that an unmeasured defect reads as maximum distrust and
never as agreement, that priority-1 always requires approval, that a confuser is
never classified as costed damage.

---

## 6. COOL and AWS

Design and findings in [cool-benchmark.md](cool-benchmark.md). Two things
changed how this is reported.

**The pip wheel already bundles KleidiCV.** `opencv-python-headless==5.0.0.93`
on arm64 reports `Custom HAL: YES (carotene, KleidiCV (ver 26.03))`. COOL's
advertised 20–70% was measured against **OpenCV 4.12.0**, which predates that.
So the benchmark has four arms, not three, and **COOL vs pip-OpenCV-5-on-Arm**
is reported as the honest comparison alongside the vendor's own baseline.

**The COOL-accelerated operations were 6% of the core pass.** Profiling found
the pass dominated not by vision but by `np.median` (38%, replaced with an exact
256-bin histogram), a 51 px ellipse dilation, and a per-candidate full-frame
mask allocation. Removing those made the pass **7.3× faster** — 24.86 → 3.40
ms/frame, 40 → 294 frames/s — and, more importantly, made the COOL-accelerated
operations **46%** of it. Measuring COOL now measures the workload that carries
the product. That evidence did not exist before profiling.

The benchmark harness detects its own environment and **refuses to let a run be
mislabelled**: COOL is recognised only when the *imported* `cv2` lives under
`/opt/cool`, never from a flag, and the result file carries its own objections.

**Status:** the x86, Graviton-OpenCV-4 and COOL arms require the EC2 instances
and are not yet measured; the COOL AMI subscription is timed so its 7-day free
trial spans the benchmark and demo window. The harness, workload and honesty
checks are complete and exercised on Arm locally.

---

## 7. Limitations, and what was got wrong

Six hypotheses were tested and rejected during this build; they are documented
in full in [negative-results.md](negative-results.md) because a report listing
only what worked is not evaluation evidence. The ones that changed the design:

* **A perimeter-proportional bias correction** looked principled and was
  rejected on held-out data — it made the holdout worse (MAE 6.5% → 9.1%).
* **The ground truth itself was wrong.** Chasing that bias revealed the
  renderer was truncating ellipse semi-axes, under-drawing every defect by an
  average of 13.5% — larger than the error being investigated. Every accuracy
  figure predating the fix was discarded.
* **Ego-motion failed three times**, and all three for one reason found only by
  brute-force correlation scanning: **lane markings dominate the view and do
  not move along the direction of travel.** Driving along a solid line looks
  like standing still. Excluding paint and using asphalt aggregate took
  accuracy from 3/13 to 12/13 frame pairs.
* **Unconditional watershed on a fragmented mask** reported a 0.50 m² pothole
  as **0.003 m²**, a 99.3% error, because the distance transform seeded every
  speckle and the seed landed in a five-pixel scrap.
* **The privacy filter destroyed the pipeline's own input.** Unbounded plate
  detection merged the whole road surface into one 966×346 px "plate"; blurring
  it took the lane markings with it and rectification quality fell from 0.94 to
  **0.00**. An over-inclusive privacy filter is not automatically a safe one.
* **Hand-tuned classifier bands** reached 3/12 on held-out confusers and were
  replaced by a fitted boundary at 12/12.

Standing limitations are listed in the [README](../README.md#known-limitations):
synthetic-only results, a residual size-dependent bias, patches undetectable,
shadow/wet confusion, assumed depth scale without calibration, and a hard
resolution floor below which nothing is measured.

---

## 8. Responsible use

**Privacy is not a feature here, it is a precondition.** Dashcam footage of
public roads incidentally records pedestrians and vehicle registration plates. A
system that uploads and retains that has built a surveillance dataset as a side
effect of counting potholes.

Scrubbing therefore runs **on load, before any later stage can see or persist a
frame**, and it **fails closed**: if face blurring is enabled and the detector
model is missing, the pipeline raises rather than passing frames through
unscrubbed. Silently skipping the blur is the one outcome that must not be
possible, because nothing downstream would reveal that it happened. The model
download is checksum-verified — a gate that caught a 131-byte Git-LFS pointer
being installed as a privacy model.

**On public money.** The system estimates repair costs. It is deliberately
conservative: priority-1 work is never dispatched without human approval
however confident the measurement; defects below a billable minimum are
monitored rather than priced; and classes that are not pavement damage are
recorded and never costed. A system that can commit public funds on its own
judgement is not one a road authority should adopt, and this one cannot.

**On the numbers.** Every quantitative claim here is from synthetic imagery and
is labelled as such. The classifier's accuracy should be read as "these cues
separate these classes", not as a claim about real road surfaces in any
particular country. Re-establishing all of it on RDD2022 is the first step
before any deployment.
