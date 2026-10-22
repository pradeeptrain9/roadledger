# RoadLedger

**Road defects measured in square metres, with the evidence for every number.**

OpenCV AI Competition 2026, powered by AWS.

**Live:** <https://d3qxgjjnaz5lwp.cloudfront.net/> — the review UI, which answers
from a deploy-time snapshot while the Graviton worker is stopped (its normal
state). Live phone capture: <https://d3qxgjjnaz5lwp.cloudfront.net/capture.html>,
which needs the worker running.

Detecting a pothole is not the hard part and is not what a road authority can
act on. RoadLedger rectifies the road plane, **measures** each defect's area in
metric units, establishes how much independent views of it agree, separates real
damage from the shadows and standing water that look like it, and emits a
prioritised repair work order with a cost — holding anything urgent for human
approval.

The measurement's own spread is what drives the agent: when views disagree, the
system gathers more evidence, and when they still disagree, it asks a person.

It runs two ways. Point it at a recorded video, or **mount a phone and drive**:
open `/capture`, calibrate in about a minute, and defects are measured, located
and priced as you pass them, with each number visibly firming up as the defect
is seen again from closer in. See
[docs/live-capture.md](docs/live-capture.md).

---

## Quick start

```bash
cd roadledger
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
bash scripts/fetch_models.sh          # YuNet face detector for privacy scrubbing
.venv/bin/python -m pytest tests/ -q  # 41 invariant tests
```

Run the web endpoint:

```bash
.venv/bin/uvicorn web.api:app --host 127.0.0.1 --port 8061
```

Then open <http://localhost:8061>. The page loads a complete cached run
immediately; "Run analysis" executes the pipeline and the agent loop live.

**Live capture** is at <http://localhost:8061/capture>, opened on a phone
mounted in the vehicle. Browsers only give a page the camera and GPS on a
secure origin, so this needs HTTPS unless it is `localhost` on the same device.
To exercise the same API without a vehicle:

```bash
.venv/bin/python -m scripts.simulate_phone --base http://127.0.0.1:8061 --speed-kmh 12
```

Reproduce the live-path accuracy and capture-rate sweep:

```bash
.venv/bin/python -m eval.live_eval
```

Reproduce the accuracy figures:

```bash
.venv/bin/python -m eval.accuracy
```

Refit the surface classifier and print its cross-validated accuracy:

```bash
.venv/bin/python -m eval.fit_classifier --scenes 140
```

Benchmark this machine's arm of the COOL comparison:

```bash
.venv/bin/python -m bench.run_benchmark --repeats 5 --frames 12
```

---

## What it does, stage by stage

| Stage | Module | What it does |
|---|---|---|
| 0. Privacy | `pipeline/privacy.py` | Blurs faces (YuNet) and plates **before anything is persisted**. Fails closed. |
| 1. Frames | `pipeline/frames.py` | Samples video or image sequences; drops motion-blurred frames by Laplacian variance. |
| 2. Rectify | `pipeline/rectify.py` | Vanishing point from lane markings → metric bird's-eye homography. Two independent routes to scale, cross-checked. |
| 3. Propose | `pipeline/propose.py`, `proposers.py` | **The COOL-accelerated core pass**: `resize`, `adaptiveThreshold`, `findContours`. Pluggable — a trained detector can supply regions instead. |
| 4. Refine | `pipeline/refine.py` | Full-resolution measurement inside each ROI: CLAHE, Otsu, conditional watershed, contour moments. |
| 4b. Classify | `pipeline/classify.py`, `cues.py` | Pothole vs shadow vs wet stain, from physical cues with a fitted boundary. |
| 5. Track | `pipeline/track.py` | Follows a defect across frames via ego-motion; yields the median area **and its spread**. |
| 6–7. Score | `pipeline/score.py` | Deterministic severity, priority, cost, and the approval gate. No model touches a number. |

The agent (`agent/loop.py`, `agent/rules.py`) decides *which measurement to take
next*; the pipeline computes every value. Two interchangeable drivers:

* **policy** — the deterministic rules. Reproducible and offline; what the
  evaluation runs against.
* **llm** — a Bedrock model sequencing the same tools (`agent/bedrock_driver.py`).

`mcp/opencv_mcp_server.py` exposes the pipeline as 11 MCP tools over JSON-RPC,
deployed to the Graviton worker so tool calls execute on the Arm path. Every
result is stamped with the OpenCV build that served it.

---

## Results

Measured on simulated drive-pasts over a persistent ground plane, where each
defect is a true circle of known radius so its area is exactly `pi*r^2`.
Full detail in [docs/accuracy.md](docs/accuracy.md).

| | |
|---|---|
| Detected | 15/15 |
| Area MAE | 15.8% |
| **MAE when the system reports confidence** | **11.1%** (max 14.4%) |
| MAE when it escalates instead | 18.4% (max 35.3%) |
| Surface classification, binary damage-vs-not (synthetic) | **0.9873** |
| Rectification applicability on real roads, after calibration | **100%** (was 2.3%) |
| Detection recall on real roads (RDD2022 India) | 0.429 |
| Damage/not-damage precision on real roads | **0.314 — see real-world.md** |
| Ego-motion accuracy | 12/13 frame pairs within 0.1 m |

**The confidence signal is calibrated** — the cases the system flags for review
genuinely are the less accurate ones. That is what makes the agentic loop more
than decoration.

Performance, after profiling (see [docs/cool-benchmark.md](docs/cool-benchmark.md)):

| | before | after |
|---|---|---|
| Core pass | 24.86 ms/frame | **3.40 ms/frame** |
| Throughput | 40.2 frames/s | **293.7 frames/s** |
| COOL-accelerated ops as a share of the core pass | 6% | **46%** |

---

## Documentation

* [docs/real-world.md](docs/real-world.md) — **what happened on real roads (RDD2022), including what failed**
* [docs/report.md](docs/report.md) — the technical report
* [docs/accuracy.md](docs/accuracy.md) — evaluation, with limitations
* [docs/cool-benchmark.md](docs/cool-benchmark.md) — COOL benchmark design and findings
* [docs/live-capture.md](docs/live-capture.md) — **live capture from a phone: calibration, the speed limit, and why it exists**
* [docs/negative-results.md](docs/negative-results.md) — seven hypotheses tested and rejected
* [docs/architecture.svg](docs/architecture.svg) · [docs/agent-workflow.svg](docs/agent-workflow.svg)

---

## Deploying to AWS

See [infra/README.md](infra/README.md). In short:

1. Subscribe to the **Cloud Optimized OpenCV for Graviton3** AMI on AWS
   Marketplace (`prodview-5b2boxpyztidw`). It carries an hourly software charge
   with a 7-day free trial.
2. `aws cloudformation deploy --template-file infra/roadledger.yaml ...`
3. On the Graviton worker, **verify which OpenCV is active before anything
   else**:

   ```bash
   source /opt/cool/venvs/python_3.12/bin/activate
   python3 -c "import cv2; print(cv2.__version__, cv2.__file__)"
   ```

   `bench/environment.py` performs this check automatically and refuses to let
   a run be mislabelled — if `/opt/cool` exists but the imported `cv2` is the
   system wheel, it says so in the results file rather than reporting the run
   as COOL.

The Graviton worker is **ephemeral**: started for a run, stopped afterwards. The
web endpoint answers from cached results so a reviewer never waits on a boot.

---

## Known limitations

Stated plainly, and in more detail in [docs/accuracy.md](docs/accuracy.md):

* **Detection and classification do not yet work on real roads.**
  Measured on RDD2022: detection recall **0.429**, and the damage/not-damage
  classifier reaches precision **0.314** — below the majority-class baseline.
  Two in three dispatches would be to nothing. See
  [docs/real-world.md](docs/real-world.md). The measurement, aggregation and
  decision layers are independent of how a candidate is proposed, so the
  proposal stage is the replaceable part; integrating a detector trained on
  RDD2022 is the first thing to do next.
* **Live capture works at survey speed, not traffic speed.** A defect must be
  seen at least three times from comparable distances, and that window is a
  couple of metres of road, so frames must be about 0.8 m apart. A phone on
  mobile data sustains that to roughly 15 km/h and marginally to 22 km/h. Above
  it, defects are still reported — marked unconfirmed and not costed — and the
  page tells the operator the speed their setup can actually measure from.
* **Measurement error on small, distant defects is much worse than the headline
  figure.** On a held-out grid of 15 drives spanning five radii, three lateral
  offsets and three start distances: bias −0.7%, **MAE 18.5%, worst case 56%**,
  against 6.5% MAE on the easier holdout below. Retuning the closing kernel
  looked like a 3.5× improvement on a development grid and failed to replicate
  on held-out data, so it was not changed
  ([negative-results.md](docs/negative-results.md) §7). No further synthetic
  tuning is planned: real footage with hand-measured areas is the instrument
  that can settle whether this is the algorithm or the rig.
* **COOL is slower than the free pip wheel on this workload.** Measured
  2026-10-22 on one `m7g.xlarge`, three arms in one process style, one
  variable: COOL against `opencv-python-headless==5.0.0.93` is **0.98x on the
  core pass and 0.83x end to end**. Against COOL's own advertised baseline -
  stock 4.12.0, no KleidiCV - the core pass gains **2%**, where the listing
  advertises 20-70%. The cause is not mysterious: OpenCV 5 is 1.19x faster
  than 4.12.0 here, COOL is built on 4.12.0, and COOL ships KleidiCV 0.5.0
  where the wheel ships 26.03. One workload, one instance type; see
  [cool-benchmark.md](docs/cool-benchmark.md) for what it does and does not
  say.
* **COOL is OpenCV 4.12.0, so OpenCV 5 is never claimed for it.** Verified on
  the Graviton3 AMI on 2026-10-21: COOL provides `cv2` 4.12.0 with KleidiCV
  0.5.0, while the pinned `opencv-python-headless==5.0.0.93` arm64 wheel
  carries KleidiCV **26.03**. The OpenCV 5 wheel performs the substantive
  analysis; COOL is the accelerated Arm execution path for the core proposal
  pass. `bench/environment.py` refuses to let a run be mislabelled, and both
  version strings go into every result file.
* **All *accuracy* figures are from synthetic imagery.** The renderer is
  physically constructed — a real pinhole camera over a persistent ground
  plane, shading applied before sensor noise, penumbra on cast shadows — and its
  own ground-truth error is +2.5%. It is still not road footage. Area accuracy
  has not been established on real roads at all, because RDD2022 annotates
  boxes rather than areas.
* **A residual −14% area bias** that varies with defect size, so it is not
  corrected by a single factor. A perimeter-proportional correction was tried
  and rejected on held-out data.
* **Patches are not detected at all.** A completed repair is too close to road
  brightness for the proposal pass to surface it, so the class was dropped from
  the classifier rather than left in untrained. The system cannot inventory
  existing repairs.
* **Shadow and wet stain are confused with each other** (12 cases in 314).
  Neither is costed, so the operational consequence is nil, but the labels are
  not reliable at that granularity.
* **Depth scale is an assumption without camera calibration.** Supplying
  `CameraGeometry` promotes it to a measurement and enables the cross-check.
* **Below a 300 px source footprint nothing is measured.** Defects too distant
  or too small are refused rather than given a confident wrong number.
