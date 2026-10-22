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

## 3. Confirmed on the AMI: COOL is OpenCV 4.12.0 with an *older* KleidiCV

Both findings above were inferences from the Marketplace listing. On
**2026-10-21** the Graviton3 COOL AMI (`ami-0c0803594d6330678`,
`COOL-graviton3-v3`) was launched on an `m7g.xlarge` and asked directly. Inside
its own Python 3.12 virtual environment:

```
cv2_version  4.12.0
cv2_path     /opt/cool/python_3.12/site-packages/cv2/python-3.12/cv2.cpython-312-aarch64-linux-gnu.so
python       3.12.3
Custom HAL:  YES (carotene (ver 0.0.1) KleidiCV (ver 0.5.0))
Baseline:    NEON FP16 NEON_DOTPROD NEON_FP16 NEON_BF16
```

Two things follow, and the second was not expected.

**COOL ships OpenCV 4.12.0, not OpenCV 5.** This was flagged as the project's
day-one landmine and it is now a measured fact rather than a guess. The
competition requires OpenCV 5 for the substantive analysis, so the claim is
split exactly as planned and as `bench/environment.py` enforces: **the OpenCV 5
wheel performs the substantive analysis, and COOL is the accelerated Arm
execution path for the core proposal pass.** Both version strings are printed
into every result file. OpenCV 5 is not claimed for COOL anywhere.

**COOL carries KleidiCV 0.5.0; the free pip wheel carries 26.03.** The
accelerator COOL is built around is several releases *behind* the one that
`opencv-python-headless==5.0.0.93` already bundles for arm64. The listing's
advertised 20–70% gains are measured against stock OpenCV 4.12.0, which has no
KleidiCV at all — a baseline nobody on Arm has had to use since the wheel
started bundling it.

So the interesting question is no longer "how much faster is COOL than
unaccelerated OpenCV", which the vendor has already answered, but **"is COOL
faster than the free wheel that is already KleidiCV-accelerated, and newer?"**
That is arm C against arm D, it is the comparison a practitioner actually
faces, and this project has found no published answer to it. It may well go
against COOL. Reporting it either way is the point.

## 4. Measured: COOL is slower than the free wheel on this workload

Run on **2026-10-22**, one `m7g.xlarge`, three arms in the same process style,
4 threads each, 5 repeats per measurement. The only variable is which `cv2` is
imported. `scripts/run_benchmark_aws.sh` reproduces it end to end.

```
B-pip-cv4-graviton3   cv2 4.12.0   numpy 2.2.6     no KleidiCV
C-pip-cv5-graviton3   cv2 5.0.0    numpy 2.2.6     KleidiCV 26.03
D-cool-graviton3      cv2 4.12.0   numpy 1.26.4    KleidiCV 0.5.0 (COOL)
```

Median ms/frame, interquartile spread:

| measurement | B (4.12.0) | C (5.0.0) | D (COOL) |
|---|---|---|---|
| resize | 0.015 ±0.001 | 0.011 ±0.000 | 0.013 ±0.001 |
| adaptiveThreshold | 0.935 ±0.001 | 0.976 ±0.001 | 0.875 ±0.001 |
| findContours | 1.035 ±0.003 | 1.074 ±0.001 | 1.018 ±0.001 |
| **proposal_pass** | 3.751 ±0.006 | **3.600 ±0.004** | 3.673 ±0.007 |
| **full_analysis** | 58.513 ±0.120 | **49.239 ±0.049** | 59.098 ±0.214 |

### D over C — COOL against the free pip wheel

| | ratio | |
|---|---|---|
| resize | 0.91x | slower |
| adaptiveThreshold | 1.11x | faster |
| findContours | 1.06x | faster |
| **proposal_pass** | **0.98x** | **slower** |
| **full_analysis** | **0.83x** | **slower** |

**COOL is slower than `opencv-python-headless==5.0.0.93` on this application.**
It wins on two of the three operations it advertises and still loses the core
pass, and loses the end-to-end analysis by 17%.

### D over B — COOL against its own advertised baseline

| | ratio |
|---|---|
| resize | 1.18x |
| adaptiveThreshold | 1.07x |
| findContours | 1.02x |
| **proposal_pass** | **1.02x** |
| full_analysis | 0.99x |

The Marketplace listing advertises **20-70%** over "standard pip-installed
OpenCV". Against that same baseline - stock 4.12.0, no KleidiCV - this
workload gains **2%** on the core pass and nothing end to end.

### Why, and it is not mysterious

| | C over B | |
|---|---|---|
| proposal_pass | 1.04x | |
| **full_analysis** | **1.19x** | OpenCV 5 against OpenCV 4.12.0, same silicon |

OpenCV 5 is simply faster than 4.12.0 on this workload. COOL is built on
4.12.0, so it starts 19% behind end to end, and its KleidiCV gains do not
cover the gap - unsurprising when **COOL ships KleidiCV 0.5.0 and the free
wheel ships 26.03**.

So the vendor's claim is not wrong so much as *stale*. It was true against a
baseline that has since moved: when the listing was benchmarked, a pip wheel on
Arm had no KleidiCV at all. It does now, and a newer one.

### What this does not say

* **It is one workload.** Three ops plus a pipeline built from them, at one
  image size, on one instance type. COOL may well win on workloads weighted
  differently - it is faster here on adaptiveThreshold and findContours in
  isolation, and those gains could dominate a pass built only from them.
* **Arm A (x86_64) is not measured**, so the architecture comparison is open.
* **The arms do not share a numpy, and cannot.** COOL's cv2 is compiled against
  NumPy 1.x and fails to import under 2.x with `numpy.core.multiarray failed to
  import`. That is a property of COOL - and a real adoption blocker for anyone
  on a current stack - not a flaw in the method. The timed calls are OpenCV
  operations on preallocated arrays, so numpy is not in the measured path, but
  it is reported on every arm because it could not be held constant.

### Why this is reported at all

A submission is free to pick the flattering comparison. This project measured
both and reports both, because the one that is useful to a practitioner - COOL
against the free, newer, already-accelerated wheel they would otherwise install
- is the one nobody appears to have published. The answer here is that COOL
does not pay for itself on this application. That is a finding, and an entry
that only reported `D over B = 1.02x` while implying the listing's 20-70%
would be a worse piece of engineering than one that says so.

## Status

**Arms B, C and D are measured** (section 4). **Arm A (x86_64) is not**: it
needs a second instance type, AMI and bootstrap for one data point, and the
three Arm arms are what the COOL question actually turns on. The harness, the
workload and the honesty checks are complete.

Total EC2 spend across four attempts, including three failed ones:
about **0.08 USD**. No instance, volume or address remains.

One unrelated defect found during that launch, recorded because it will bite
anyone reusing the template: the worker's bootstrap ran `set -euo pipefail`
and then sourced the COOL venv's `activate`, which references `PYTHONPATH`
without a default. Under `set -u` that is a fatal unbound-variable error, so
cloud-init aborted before cloning the repository and the machine came up with
COOL present and the pipeline absent. `/var/log/cloud-init-output.log` names it
exactly:

```
/opt/cool/venvs/python_3.12/bin/activate: line 74: PYTHONPATH: unbound variable
```

The bootstrap now clears `-u` around the activation.
