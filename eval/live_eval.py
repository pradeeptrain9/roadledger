"""Accuracy of the *live* path, which is not the accuracy of the batch path.

The batch evaluation tracks a defect for a few frames forward from where it
was first detected. A live stream cannot do that and does not want to: it
follows every defect from first sighting until it passes under the vehicle. The
consequences are measurable and went in both directions when this was first
run, which is why this file exists separately rather than reusing
`eval/accuracy.py`:

* The live path gathers **more** views, because it uses the whole approach
  rather than a fixed window. On a 1.2 m/frame drive-past the batch harness
  could not establish a spread at all on three of four defect sizes; the live
  path got five views of each.
* It also reaches places the batch path never does - in particular the moment
  the defect crosses the near edge of the rectified view, where the boundary is
  cut off by the frame and the area collapses. That produced a 76%-low reading
  which, being the lowest of three, became the reported median. It is now
  refused outright by the border guard in `refine.py`.
* And it is governed by a different limit: `VANTAGE_FOOTPRINT_FRACTION`. Only
  views whose source footprint is within that fraction of the best one
  contribute, and because footprint falls as roughly the cube of distance,
  0.25 corresponds to a distance band of only 1.59x. At the near edge of the
  bird's-eye view that is about one and a half metres of travel, so at 25 km/h
  a phone would have to send 14 frames a second to put three usable views
  inside it. That is the binding constraint on live capture, and it is a
  geometric fact rather than a compute one.

So this module sweeps the two things that trade off against each other - the
vantage fraction and the frame spacing - and reports confirmation rate
alongside bias and error. A setting that measures beautifully but confirms
nothing is not better.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

import pipeline.track as track_module
from eval.accuracy import default_config
from pipeline.geo import GeoFix, metres_per_degree
from pipeline.live import LiveSession
from tests.ground import build_ground, render_drive
from tests.synthetic import SceneCamera

# The approach band a defect is usable over, derived rather than guessed.
# Projected ground area falls as 1/d^3, so a vantage fraction v admits
# distances up to v^(-1/3) times the closest approach.
def usable_distance_ratio(vantage_fraction: float) -> float:
    return vantage_fraction ** (-1.0 / 3.0)


@dataclass
class LiveRow:
    truth_area_m2: float
    step_m: float
    vantage: float
    detected: bool
    confirmed: bool
    views_raw: int
    views_used: int
    measured_area_m2: float
    iqr_ratio: float
    rejected_clipped: int
    seed: int

    @property
    def error_frac(self) -> float | None:
        if self.measured_area_m2 <= 0:
            return None
        return (self.measured_area_m2 - self.truth_area_m2) / self.truth_area_m2

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "error_frac": None if self.error_frac is None else round(self.error_frac, 4),
        }


def drive_live(
    truth_r_m: float,
    step_m: float,
    fps: float,
    start_distance_m: float = 12.0,
    lateral_m: float = -0.4,
    n_frames: int = 30,
    seed: int = 5,
    scrub: bool = False,
) -> tuple[LiveSession, float]:
    """One synthetic drive-past, streamed frame by frame as a phone would."""
    camera = SceneCamera()
    config = default_config(camera)
    ground = build_ground(
        defects=[(lateral_m, start_distance_m, truth_r_m)], seed=seed, length_m=90.0
    )
    images, _ = render_drive(ground, camera, n_frames=n_frames, step_m=step_m, seed=seed + 16)

    lat0, lon0 = 17.4401, 78.3489
    per_lat, _ = metres_per_degree(lat0)
    session = LiveSession(config=config, run_id=f"live_{truth_r_m}_{step_m}_{seed}")
    for seq, image in enumerate(images):
        fix = GeoFix(
            lat=lat0 + (seq * step_m) / per_lat,
            lon=lon0,
            accuracy_m=6.0,
            heading_deg=0.0,
            speed_mps=step_m * fps,
            timestamp_s=seq / fps,
        )
        session.ingest(image, fix, seq, scrub_frame=scrub)
    session.finish()
    return session, math.pi * truth_r_m**2


def _best_track(session: LiveSession):
    tracks = [t for t in session.tracks.values() if t.measurements]
    if not tracks:
        return None
    return max(tracks, key=lambda t: t.best_footprint_px)


def row_for(
    truth_r_m: float, step_m: float, fps: float, vantage: float, seed: int = 5
) -> LiveRow:
    """One cell of the sweep, at one vantage fraction."""
    original = track_module.VANTAGE_FOOTPRINT_FRACTION
    track_module.VANTAGE_FOOTPRINT_FRACTION = vantage
    try:
        session, truth = drive_live(truth_r_m, step_m, fps, seed=seed)
    finally:
        track_module.VANTAGE_FOOTPRINT_FRACTION = original

    track = _best_track(session)
    if track is None:
        return LiveRow(truth, step_m, vantage, False, False, 0, 0, 0.0, math.inf, 0, seed)

    record = session.defects.get(track.defect_id, {})
    clipped = sum(
        1 for m in track.measurements if "truncation" in (m.note or "")
    )
    return LiveRow(
        truth_area_m2=round(truth, 6),
        step_m=step_m,
        vantage=vantage,
        detected=True,
        confirmed=bool(record.get("confirmed")),
        views_raw=track.n_views,
        views_used=int(record.get("n_measurements", 0)),
        measured_area_m2=float(record.get("area_m2", 0.0)),
        iqr_ratio=float(record.get("area_iqr_ratio", -1.0)),
        rejected_clipped=clipped,
        seed=seed,
    )


def sweep(
    radii: tuple[float, ...] = (0.22, 0.28, 0.34, 0.40),
    steps: tuple[tuple[float, float], ...] = ((0.45, 6.0), (0.8, 6.0), (1.2, 6.0), (2.0, 4.0)),
    vantages: tuple[float, ...] = (0.10, 0.18, 0.25, 0.40),
    seeds: tuple[int, ...] = (5, 11),
) -> list[LiveRow]:
    rows: list[LiveRow] = []
    for vantage in vantages:
        for step_m, fps in steps:
            for radius in radii:
                for seed in seeds:
                    rows.append(row_for(radius, step_m, fps, vantage, seed))
    return rows


def summarise(rows: list[LiveRow]) -> list[dict[str, Any]]:
    """Group by vantage and frame spacing. Confirmation rate first, then error.

    Reported together on purpose. A tighter vantage fraction gives a better
    measurement of the defects it still manages to confirm, and confirms fewer
    of them, and a summary that showed only the error would recommend the
    wrong setting.
    """
    out: list[dict[str, Any]] = []
    keys = sorted({(r.vantage, r.step_m) for r in rows})
    for vantage, step_m in keys:
        group = [r for r in rows if r.vantage == vantage and r.step_m == step_m]
        confirmed = [r for r in group if r.confirmed and r.error_frac is not None]
        errors = [r.error_frac for r in confirmed]
        out.append(
            {
                "vantage": vantage,
                "distance_band_x": round(usable_distance_ratio(vantage), 2),
                "step_m": step_m,
                "n": len(group),
                "confirmed": len(confirmed),
                "confirm_rate": round(len(confirmed) / len(group), 3) if group else 0.0,
                "mean_views_used": round(
                    statistics.mean([r.views_used for r in group]), 2
                )
                if group
                else 0.0,
                "bias": round(statistics.mean(errors), 4) if errors else None,
                "mae": round(statistics.mean([abs(e) for e in errors]), 4) if errors else None,
                "clipped_views_rejected": sum(r.rejected_clipped for r in group),
            }
        )
    return out


def main() -> None:
    rows = sweep()
    print(
        f"{'vantage':>8} {'band':>6} {'step_m':>7} {'n':>4} {'conf':>5} {'rate':>6} "
        f"{'views':>6} {'bias':>8} {'mae':>7} {'clipped':>8}"
    )
    for item in summarise(rows):
        bias = "-" if item["bias"] is None else f"{item['bias']:+.1%}"
        mae = "-" if item["mae"] is None else f"{item['mae']:.1%}"
        print(
            f"{item['vantage']:>8.2f} {item['distance_band_x']:>5.2f}x {item['step_m']:>7.2f} "
            f"{item['n']:>4} {item['confirmed']:>5} {item['confirm_rate']:>6.2f} "
            f"{item['mean_views_used']:>6.1f} {bias:>8} {mae:>7} "
            f"{item['clipped_views_rejected']:>8}"
        )


if __name__ == "__main__":
    main()
