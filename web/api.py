"""The judge-accessible web endpoint.

Serves the ledger, the per-defect evidence, the agent's decision trace, and the
human approval gate. Three design points matter:

* **Cached demo first.** The Graviton worker is ephemeral - it is started for a
  run and stopped afterwards, because leaving it up would spend the budget on
  nothing. So a reviewer opening the page gets a complete cached run
  immediately, and a live run is a button rather than a precondition. A demo
  that only works when a GPU-hour is burning is not a demo.
* **The approval gate is real.** A priority-1 defect is not dispatched until
  someone approves it here. The endpoint records who approved what and when,
  and refuses to emit a work order for a defect the engine marked as needing
  approval until that happens.
* **The trace is the product, not a debug view.** The reason a defect has the
  area it has - how many views agreed, which rules fired, what changed after
  re-measurement - is what makes the number worth acting on, so it is a
  first-class part of the interface.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.loop import AgentLoop
from eval.accuracy import default_config
from pipeline.runner import Session
from web.live import router as live_router
from tests.ground import build_ground, render_drive
from tests.synthetic import SceneCamera

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="RoadLedger",
    description="Measured road-defect triage. Areas in square metres, with the evidence.",
    version="0.1.0",
)

# Scenarios the endpoint can run. Synthetic by construction, so the ground
# truth is exact and a reviewer can check the measurement rather than take it
# on trust. Real footage is loaded by path through the same code path.
SCENARIOS: dict[str, dict[str, Any]] = {
    "two-potholes": {
        "label": "Two potholes, a shadow and a wet stain",
        "defects": [(-0.5, 9.0, 0.36), (0.9, 6.0, 0.20)],
        "confusers": [(0.2, 11.0, 0.30, "shadow"), (-1.1, 7.5, 0.28, "wet")],
        "frames": 12,
        "seed": 13,
    },
    "large-pothole": {
        "label": "One large pothole, urgent",
        "defects": [(-0.3, 10.0, 0.45)],
        "confusers": [(1.0, 8.0, 0.25, "shadow")],
        "frames": 12,
        "seed": 5,
    },
    "confusers-only": {
        "label": "No damage: a shadow and standing water only",
        "defects": [],
        "confusers": [(0.0, 8.0, 0.32, "shadow"), (-0.9, 6.5, 0.30, "wet")],
        "frames": 10,
        "seed": 21,
    },
}

_runs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


class RunRequest(BaseModel):
    scenario: Literal["two-potholes", "large-pothole", "confusers-only"] = "two-potholes"
    max_frames: int = Field(default=5, ge=1, le=20)


class ApprovalRequest(BaseModel):
    approved: bool
    approver: str = Field(default="", max_length=120)
    note: str = Field(default="", max_length=500)


def _clean(value: Any) -> Any:
    """JSON has no infinity. An unestablished spread becomes null, not a number.

    Serialising `inf` as a large float would make "no agreement could be
    established" look like a very wide but real measurement, which is exactly
    the confusion the spread exists to prevent.
    """
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _ground_truth(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "pothole",
            "lateral_m": lateral,
            "station_m": forward,
            "area_m2": round(math.pi * radius**2, 4),
        }
        for lateral, forward, radius in scenario["defects"]
    ] + [
        {"kind": kind, "lateral_m": lateral, "station_m": forward, "area_m2": None}
        for lateral, forward, _radius, kind in scenario.get("confusers", [])
    ]


def execute(scenario_key: str, max_frames: int) -> dict[str, Any]:
    """Run the full analysis and the agent loop over one scenario."""
    scenario = SCENARIOS[scenario_key]
    camera = SceneCamera()
    config = default_config(camera)

    ground = build_ground(
        defects=scenario["defects"],
        confusers=scenario.get("confusers"),
        seed=scenario["seed"],
    )
    images, _ = render_drive(
        ground, camera, n_frames=scenario["frames"], step_m=0.6
    )

    started = time.perf_counter()
    session = Session(config=config)
    session.load_images(images)
    loop = AgentLoop(session, config)
    outcome = loop.run(max_frames=max_frames)
    elapsed_ms = (time.perf_counter() - started) * 1000

    defects = []
    for defect in outcome.defects:
        record = _clean(defect.to_dict())
        # The per-view measurements behind the reported median: this is the
        # evidence a reviewer needs to judge whether to trust the number.
        candidate_id = session.defect_candidate.get(defect.defect_id, "")
        record["measurements"] = [
            _clean(measurement.to_dict())
            for measurement in session.measurements.get(candidate_id, [])
        ]
        record["candidate_id"] = candidate_id
        record["approval"] = None
        defects.append(record)

    return {
        "run_id": outcome.run_id,
        "scenario": scenario_key,
        "scenario_label": scenario["label"],
        "ground_truth": _ground_truth(scenario),
        "frames": len(session.frames),
        "privacy": session.privacy_counts,
        "defects": defects,
        "work_orders": [d.defect_id for d in outcome.work_orders],
        "escalated": _clean(outcome.escalated),
        "duplicates_merged": outcome.duplicates_merged,
        "frames_skipped": outcome.frames_skipped,
        "trace": _clean(loop.trace.to_dict()),
        "trace_lines": loop.trace.summary_lines(),
        "rules_fired": sorted({step.rule for step in loop.trace.decisions()}),
        "tool_calls": loop.trace.tool_calls,
        "timings": session.timings.to_dict(),
        "elapsed_ms": round(elapsed_ms, 1),
        "total_cost": round(
            sum(d.cost_estimate for d in outcome.defects if d.billable), 2
        ),
        "currency": config.score.currency,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Which OpenCV build is serving this, and where it is running."""
    from bench.environment import cool_version, detect

    environment = detect()
    return {
        "status": "ok",
        "opencv_version": environment.cv2_version,
        "is_cool_build": environment.is_cool_build,
        "kleidicv_in_build": environment.kleidicv_in_build,
        "architecture": environment.architecture,
        "instance_type": environment.instance_type or None,
        "cool_version": cool_version(),
        "notes": environment.notes,
        "scenarios": {key: value["label"] for key, value in SCENARIOS.items()},
    }


@app.get("/api/demo")
def demo() -> dict[str, Any]:
    """A complete cached run, so the page is useful the moment it loads."""
    with _lock:
        cached = _runs.get("demo")
    if cached is None:
        cached = execute("two-potholes", max_frames=5)
        cached["cached"] = True
        with _lock:
            _runs["demo"] = cached
            _runs[cached["run_id"]] = cached
    return cached


@app.post("/api/runs")
def create_run(request: RunRequest) -> dict[str, Any]:
    result = execute(request.scenario, request.max_frames)
    result["cached"] = False
    with _lock:
        _runs[result["run_id"]] = result
    return result


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    with _lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return run


@app.post("/api/runs/{run_id}/defects/{defect_id}/approval")
def set_approval(run_id: str, defect_id: str, request: ApprovalRequest) -> dict[str, Any]:
    """The human gate. A defect needing approval is not dispatched without it."""
    with _lock:
        run = _runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
        defect = next((d for d in run["defects"] if d["defect_id"] == defect_id), None)
        if defect is None:
            raise HTTPException(status_code=404, detail=f"no such defect: {defect_id}")

        defect["approval"] = {
            "approved": request.approved,
            "approver": request.approver or "unattributed",
            "note": request.note,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        # Dispatch only follows an explicit approval. Rejecting is recorded too:
        # a rejected priority-1 defect is evidence the system was wrong about
        # something, which is worth keeping.
        if request.approved and defect_id not in run["work_orders"]:
            run["work_orders"].append(defect_id)
        if not request.approved and defect_id in run["work_orders"]:
            run["work_orders"].remove(defect_id)

    return {"defect_id": defect_id, "approval": defect["approval"], "dispatched": request.approved}


# Live capture from a phone or dashcam. Mounted as its own router because it is
# a different kind of thing from the review endpoints above: it is driven by a
# vehicle in motion rather than by a reviewer, and its state is per-vehicle.
app.include_router(live_router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/capture")
def capture() -> FileResponse:
    """The page the phone opens, mounted in the vehicle."""
    return FileResponse(STATIC_DIR / "capture.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
