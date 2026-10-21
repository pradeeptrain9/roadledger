# COOL benchmark: design, and what was found building it

## Two findings that reshaped the comparison

### 1. The standard pip wheel already bundles KleidiCV

`opencv-python-headless==5.0.0.93` for arm64 reports:

```
3rdparty dependencies: ... tegra_hal kleidicv_hal kleidicv kleidicv_thread
Custom HAL:            YES (carotene (ver 0.0.1) KleidiCV (ver 26.03))
```

COOL is a KleidiCV-optimised build, and its Marketplace listing advertises
20-70% over "standard pip-installed OpenCV" benchmarked against **OpenCV
4.12.0** - a build predating that bundling. So *pip OpenCV 5 on Graviton is
already KleidiCV-accelerated*, and COOL against it is a materially harder
comparison than COOL against pip OpenCV 4.

Reporting only the vendor's baseline would overstate what COOL adds to this
application today. The benchmark therefore has four arms, not three:

| Arm | Build | Architecture | Why |
|---|---|---|---|
| A | pip OpenCV 5 | x86_64 (m6i.xlarge) | architecture baseline |
| B | pip OpenCV 4.12.0 | Graviton3 (m7g.xlarge) | the vendor's own stated baseline |
| C | pip OpenCV 5 | Graviton3 (m7g.xlarge) | already KleidiCV - the honest baseline |
| D | COOL | Graviton3 (m7g.xlarge) | the product under test |

B-vs-D reproduces the advertised claim. **C-vs-D is the one that says what COOL
adds to this application**, and it is the one to read.

### 2. The ops COOL accelerates were 6% of the core pass

First measurement of the core proposal pass, on the Arm development machine:

| | ms/frame |
|---|---|
| `resize` | 0.017 |
| `adaptiveThreshold` | 0.684 |
| `findContours` | 0.780 |
| **sum of COOL-accelerated ops** | **1.48** |
| **whole proposal pass** | **24.86** |

The three operations COOL accelerates were **6%** of the pass it supposedly
carries. A 70% speedup on them would have moved the pass by about 4%, and any
COOL measurement would really have been a measurement of the overhead around
it - which would have made the COOL claim technically true and practically
meaningless.

Profiling found the overhead was not vision work at all:

* `np.median` was **38%** of the pass. The median and the MAD together need
  three full partitions of ~300k pixels per frame. Replaced with an exact
  256-bin histogram computation - same answers, one pass, no sorting.
* Dilating a 51 px ellipse for the paint halo cost ~10 ms/frame. The halo is a
  conservative safety margin, not a precise boundary, so it is computed at
  quarter resolution.
* The per-candidate brightness filter allocated a full 640x640 mask per
  candidate, making the loop O(candidates x 640 x 640). The contour lives in
  its bounding box, so now the mask does too.

| | before | after |
|---|---|---|
| proposal pass | 24.86 ms/frame | **3.40 ms/frame** |
| throughput | 40.2 frames/s | **293.7 frames/s** |
| full analysis | 86.1 ms/frame | 46.0 ms/frame |
| COOL-accelerated ops as a share of the core pass | 6% | **46%** |

7.3x faster, with all 19 invariant tests still passing. The point is not only
the speedup: **the COOL-accelerated operations now genuinely dominate the core
pass**, so measuring COOL measures the workload that carries the product. That
is the evidence the award asks for, and it did not exist before profiling.

## Method

`bench/run_benchmark.py` runs on each arm, detects its own environment, and
writes one JSON file; `bench/report.py` merges them. Nothing about the
environment is typed in by hand.

* **COOL detection is not a flag.** The Marketplace AMI installs to
  `/opt/cool`, so `is_cool_build` is true only when the *imported* `cv2`
  module file lives under that prefix. A run that intended COOL but imported
  the system wheel is detected and flagged rather than reported as COOL.
* **The workload is the application's own data** - real rectified bird's-eye
  views produced by the real renderer and real rectification, so the pixel
  statistics (lane paint, asphalt texture, defects, warp border) are the ones
  the operations meet in production. Benchmarking on synthetic noise would
  measure these operations on data the product never sees.
* **Three levels are timed**: the three COOL operations in isolation, the core
  proposal pass they compose, and the full analysis from rectification to
  scoring - so the core pass's share of real work is visible rather than
  assumed.
* **Everything is repeated** and reported as a median with min, max and IQR.
  Default five repeats after two warmup iterations. A single run of anything is
  not a measurement.
* **Cost** is reported as USD per 1000 frames from recorded on-demand pricing
  plus COOL's hourly software charge where it applies, with the pricing date in
  the output file.

## Honesty checks built into the harness

`bench/environment.py` refuses to let a run be mislabelled, and writes its
objections into the result file:

* `/opt/cool` present but the imported `cv2` is elsewhere -> flags that the run
  is **not** using COOL, with the activation command.
* COOL reporting an OpenCV 4.x version -> flags that OpenCV 5 must not be
  claimed for COOL, and that the report must split the claim: COOL accelerates
  the core ops on Arm while OpenCV 5 performs the substantive analysis.
* COOL detected on a non-Arm machine -> flags that the award's Arm-path
  requirement is not met.

## Status

Arms A, B and D are **not yet measured**: they need the EC2 instances, and the
COOL AMI subscription is timed so its 7-day free trial spans the benchmark and
demo window. The harness, the workload and the honesty checks are complete and
have been exercised on Arm locally (the figures above). Running the other arms
is one command per instance.
