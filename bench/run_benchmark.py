"""Benchmark the core workload, on whatever this machine is.

Run this on each arm of the comparison. It detects its own environment, runs
the same work, and writes one JSON file. `bench/report.py` then merges those
files into the comparison table.

The arms, and why there are four rather than three:

  A  pip OpenCV 5, x86_64        - the architecture baseline
  B  pip OpenCV 4.12.0, Graviton - the vendor's own stated baseline
  C  pip OpenCV 5, Graviton      - already KleidiCV-accelerated, see below
  D  COOL, Graviton              - the product under test

Arm C exists because of something found while building this: the standard
`opencv-python-headless==5.0.0.93` arm64 wheel already bundles KleidiCV. Its
build information reports `Custom HAL: YES (carotene, KleidiCV (ver 26.03))`.
COOL's Marketplace listing advertises 20-70% over "standard pip-installed
OpenCV", benchmarked against **OpenCV 4.12.0** - a build that predates that
bundling. So COOL against pip-OpenCV-5-on-Arm is a materially harder
comparison than COOL against pip-OpenCV-4-on-Arm, and reporting only the
latter would overstate what COOL adds to this application today. Both are
measured.

What is timed:

* **Per-operation microbenchmarks** for the three operations COOL names as
  accelerated - resize, adaptive Gaussian threshold, contour detection - on the
  same bird's-eye images the application produces.
* **The end-to-end proposal pass**, which is the core per-frame workload and is
  built from exactly those three operations.
* **The full analysis**, rectification through scoring, so the proposal pass's
  share of real work is visible rather than assumed.

Every measurement is repeated and reported as a median with spread. A single
run of anything is not a measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from bench.environment import Environment, cool_version, detect
from pipeline.config import ProposeParams
from pipeline.propose import propose

# On-demand USD per hour, us-east-1, for the $/1000-frames figure. Recorded
# here rather than fetched so a benchmark run is reproducible offline; update
# alongside AWS pricing and note the date.
HOURLY_USD = {
    "m6i.xlarge": 0.192,
    "m7g.xlarge": 0.163,
    "c7g.xlarge": 0.145,
    "c7g.2xlarge": 0.290,
    "c6g.2xlarge": 0.272,
    "t4g.small": 0.0168,
}
PRICING_AS_OF = "2026-10-13"
# COOL's Marketplace software charge, USD/hour, on top of the instance.
COOL_SOFTWARE_USD = {"m7g.xlarge": 0.01, "c7g.xlarge": 0.01, "c7g.2xlarge": 0.02}


@dataclass
class Stat:
    """A repeated measurement, as a median and a spread."""

    name: str
    unit: str
    median: float
    minimum: float
    maximum: float
    iqr: float
    n: int
    samples: list[float] = field(default_factory=list)

    @classmethod
    def of(cls, name: str, unit: str, values: list[float]) -> "Stat":
        ordered = sorted(values)
        q1, q3 = (
            np.percentile(ordered, [25, 75]) if len(ordered) > 1 else (ordered[0], ordered[0])
        )
        return cls(
            name=name,
            unit=unit,
            median=round(statistics.median(ordered), 5),
            minimum=round(ordered[0], 5),
            maximum=round(ordered[-1], 5),
            iqr=round(float(q3 - q1), 5),
            n=len(ordered),
            samples=[round(v, 5) for v in ordered],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _time_repeated(
    operation: Callable[[], Any], repeats: int, warmup: int = 2
) -> list[float]:
    """Milliseconds per call, warmed up first so caches are not being measured."""
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def build_workload(n_frames: int, seed: int = 3) -> list[np.ndarray]:
    """The actual images the application processes: rectified bird's-eye views.

    Benchmarking on synthetic noise would measure the operations on data the
    product never sees. These are produced by the real renderer and the real
    rectification, so the pixel statistics - lane paint, asphalt texture,
    defects, warp border - are the ones the operations meet in production.
    """
    from eval.accuracy import default_config
    from pipeline.rectify import rectify
    from pipeline.types import Frame
    from tests.ground import build_ground, render_drive
    from tests.synthetic import SceneCamera

    camera = SceneCamera()
    config = default_config(camera)
    ground = build_ground(
        defects=[(-0.5, 9.0, 0.34), (0.8, 6.5, 0.22)],
        confusers=[(0.3, 11.0, 0.30, "shadow")],
        seed=seed,
    )
    images, _ = render_drive(ground, camera, n_frames=n_frames, step_m=0.6)

    bevs = []
    for index, image in enumerate(images):
        frame = Frame(f"b{index}", index, index * 0.5, 999.0, image, "bench")
        _, bev = rectify(frame, config.rectify)
        if bev is not None:
            bevs.append(bev)
    if not bevs:
        raise RuntimeError("no frame could be rectified; cannot build a workload")
    return bevs


def microbenchmarks(bevs: list[np.ndarray], repeats: int) -> list[Stat]:
    """The three operations COOL names, timed in isolation."""
    params = ProposeParams().normalised()
    grays = [cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) for bev in bevs]
    scale = params.scale
    small_size = (
        max(8, round(grays[0].shape[1] * scale)),
        max(8, round(grays[0].shape[0] * scale)),
    )
    smalls = [cv2.resize(g, small_size, interpolation=cv2.INTER_AREA) for g in grays]
    binaries = [
        cv2.adaptiveThreshold(
            s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            params.block_size, params.c_constant,
        )
        for s in smalls
    ]

    def run_resize() -> None:
        for gray in grays:
            cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)

    def run_threshold() -> None:
        for small in smalls:
            cv2.adaptiveThreshold(
                small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                params.block_size, params.c_constant,
            )

    def run_contours() -> None:
        for binary in binaries:
            cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n = len(bevs)
    return [
        Stat.of("resize", "ms/frame", [t / n for t in _time_repeated(run_resize, repeats)]),
        Stat.of(
            "adaptiveThreshold",
            "ms/frame",
            [t / n for t in _time_repeated(run_threshold, repeats)],
        ),
        Stat.of(
            "findContours", "ms/frame", [t / n for t in _time_repeated(run_contours, repeats)]
        ),
    ]


def proposal_pass(bevs: list[np.ndarray], repeats: int) -> Stat:
    """The core per-frame workload, end to end."""
    params = ProposeParams()

    def run() -> None:
        for index, bev in enumerate(bevs):
            propose(bev, f"b{index}", params)

    n = len(bevs)
    return Stat.of(
        "proposal_pass", "ms/frame", [t / n for t in _time_repeated(run, repeats)]
    )


def full_analysis(n_frames: int, repeats: int) -> Stat:
    """Rectification through scoring, so the core pass's share is visible."""
    from agent.loop import AgentLoop
    from eval.accuracy import default_config
    from pipeline.runner import Session
    from tests.ground import build_ground, render_drive
    from tests.synthetic import SceneCamera

    camera = SceneCamera()
    config = default_config(camera)
    ground = build_ground(
        defects=[(-0.5, 9.0, 0.34), (0.8, 6.5, 0.22)],
        confusers=[(0.3, 11.0, 0.30, "shadow")],
        seed=3,
    )
    images, _ = render_drive(ground, camera, n_frames=n_frames, step_m=0.6)

    def run() -> None:
        session = Session(config=config)
        session.load_images(images)
        AgentLoop(session, config).run(max_frames=4)

    return Stat.of(
        "full_analysis",
        "ms/frame",
        [t / len(images) for t in _time_repeated(run, repeats, warmup=1)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="repeats per measurement")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--threads", type=int, default=0, help="0 leaves OpenCV's default")
    parser.add_argument("--out", default="")
    parser.add_argument("--label", default="", help="override the arm label")
    parser.add_argument(
        "--skip-full", action="store_true", help="microbenchmarks and core pass only"
    )
    args = parser.parse_args()

    if args.threads:
        cv2.setNumThreads(args.threads)

    environment = detect()
    print(f"environment: {environment.label()}")
    print(f"  cv2 {environment.cv2_version} from {environment.cv2_path}")
    print(f"  COOL build: {environment.is_cool_build}   KleidiCV in build: {environment.kleidicv_in_build}")
    for note in environment.notes:
        print(f"  NOTE: {note}")

    print(f"\nbuilding workload ({args.frames} frames)...")
    bevs = build_workload(args.frames)
    print(f"  {len(bevs)} rectified bird's-eye views, {bevs[0].shape}")

    print(f"running microbenchmarks ({args.repeats} repeats)...")
    stats = microbenchmarks(bevs, args.repeats)
    print("running the core proposal pass...")
    stats.append(proposal_pass(bevs, args.repeats))
    if not args.skip_full:
        print("running the full analysis...")
        stats.append(full_analysis(args.frames, max(3, args.repeats // 2)))

    core = next(s for s in stats if s.name == "proposal_pass")
    throughput = 1000.0 / core.median if core.median > 0 else 0.0
    hourly = HOURLY_USD.get(environment.instance_type, 0.0)
    software = COOL_SOFTWARE_USD.get(environment.instance_type, 0.0) if environment.is_cool_build else 0.0
    total_hourly = hourly + software
    usd_per_1000 = (
        (1000.0 / throughput) * total_hourly / 3600.0 if throughput > 0 and total_hourly else 0.0
    )

    payload = {
        "label": args.label or environment.label(),
        "environment": environment.to_dict(),
        "cool_version": cool_version(),
        "workload": {
            "frames": len(bevs),
            "bev_shape": list(bevs[0].shape),
            "repeats": args.repeats,
            "cv2_threads": cv2.getNumThreads(),
        },
        "stats": [stat.to_dict() for stat in stats],
        "derived": {
            "core_pass_ms_per_frame": core.median,
            "throughput_frames_per_s": round(throughput, 2),
            "instance_hourly_usd": hourly,
            "cool_software_hourly_usd": software,
            "usd_per_1000_frames": round(usd_per_1000, 6),
            "pricing_as_of": PRICING_AS_OF,
            "pricing_known": bool(total_hourly),
        },
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    print("\nresults")
    for stat in stats:
        print(
            f"  {stat.name:20s} {stat.median:8.3f} {stat.unit}   "
            f"[{stat.minimum:.3f}-{stat.maximum:.3f}] iqr={stat.iqr:.3f} n={stat.n}"
        )
    print(f"\n  throughput: {throughput:.1f} frames/s on the core pass")
    if usd_per_1000:
        print(f"  cost: ${usd_per_1000:.6f} per 1000 frames (as of {PRICING_AS_OF})")
    else:
        print("  cost: instance type unknown off EC2, so no $/frame figure")

    out = Path(
        args.out
        or f"bench/results/{platform.machine()}-{'cool' if environment.is_cool_build else 'pip'}-cv{environment.cv2_version}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
