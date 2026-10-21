"""MCP server exposing the OpenCV 5 pipeline as agent tools.

Deployed to the Graviton worker, where it runs against the COOL build, so the
agent's tool calls execute the core image workload on the Arm path. This mirrors
the pattern in the official `opencv/COOL-Benchmark` repository, which ships an
`opencv-ami/opencv-mcp-server.py` to each instance - MCP over COOL is the
organisers' own arrangement rather than something invented here.

The protocol is spoken directly over JSON-RPC on stdin/stdout. That is a
deliberate choice: the server has to run inside the COOL AMI's own Python
environment, which provides `cv2` from /opt/cool, and adding an SDK dependency
there risks pulling in a second OpenCV and silently measuring the wrong build.
The surface implemented is `initialize`, `tools/list`, `tools/call` and
`ping` - what an agent needs to discover and invoke these tools.

Each tool maps onto exactly one pipeline operation, so a trace of tool calls is
a trace of vision operations. Every response carries the environment's OpenCV
version and whether it is the COOL build, which means a captured trace proves
on its own which build served it.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable

from pipeline.config import PipelineConfig, ProposeParams, RectifyParams, RefineParams
from pipeline.runner import Session

class UnknownTool(KeyError):
    """The tool name is not in the registry.

    Distinct from any error *inside* a tool. An unknown tool is a protocol
    error the agent cannot act on, while a tool that failed - an unknown frame
    id, an unrectifiable frame - is a result the agent can and should react to,
    so the two must not collapse into the same response.
    """


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "roadledger-opencv"
SERVER_VERSION = "0.1.0"


def _environment() -> dict[str, Any]:
    from bench.environment import cool_version, detect

    detected = detect()
    return {
        "opencv_version": detected.cv2_version,
        "opencv_path": detected.cv2_path,
        "is_cool_build": detected.is_cool_build,
        "kleidicv_in_build": detected.kleidicv_in_build,
        "architecture": detected.architecture,
        "instance_type": detected.instance_type,
        "cool_version": cool_version(),
        "notes": detected.notes,
    }


class ToolServer:
    """Holds one Session and dispatches tool calls against it."""

    def __init__(self) -> None:
        self.config = PipelineConfig()
        self.session = Session(config=self.config)
        self.environment = _environment()

    # -- tools ------------------------------------------------------------

    def load_source(self, source: str) -> dict[str, Any]:
        self.session = Session(config=self.config)
        return self.session.load(source)

    def sample_frames(self) -> dict[str, Any]:
        return {
            "frames": [frame.to_dict() for frame in self.session.frames],
            "count": len(self.session.frames),
        }

    def rectify(self, frame_id: str, road_band_frac: float | None = None) -> dict[str, Any]:
        params = self.config.rectify
        if road_band_frac is not None:
            params = RectifyParams(
                **{**params.__dict__, "road_band_frac": float(road_band_frac)}
            )
        rectification = self.session.rectify_frame(frame_id, params)
        return rectification.to_dict()

    def propose_candidates(
        self,
        frame_id: str,
        scale: float | None = None,
        block_size: int | None = None,
        c_constant: float | None = None,
    ) -> dict[str, Any]:
        """The COOL-accelerated core pass. Parameters are agent-adjustable."""
        params = self.config.propose
        overrides = {}
        if scale is not None:
            overrides["scale"] = float(scale)
        if block_size is not None:
            overrides["block_size"] = int(block_size)
        if c_constant is not None:
            overrides["c_constant"] = float(c_constant)
        if overrides:
            params = ProposeParams(**{**params.__dict__, **overrides})

        candidates = self.session.propose_candidates(frame_id, params)
        return {
            "frame_id": frame_id,
            "n_candidates": len(candidates),
            "parameters": {
                "scale": params.scale,
                "block_size": params.block_size,
                "c_constant": params.c_constant,
            },
            "candidates": [candidate.to_dict() for candidate in candidates],
        }

    def classify_surface(self, candidate_id: str) -> dict[str, Any]:
        return self.session.classify(candidate_id).to_dict()

    def refine_roi(
        self, candidate_id: str, clahe_clip_limit: float | None = None,
        roi_pad_frac: float | None = None,
    ) -> dict[str, Any]:
        params = self.config.refine
        overrides = {}
        if clahe_clip_limit is not None:
            overrides["clahe_clip_limit"] = float(clahe_clip_limit)
        if roi_pad_frac is not None:
            overrides["roi_pad_frac"] = float(roi_pad_frac)
        if overrides:
            params = RefineParams(**{**params.__dict__, **overrides})
        return self.session.refine_candidate(candidate_id, params).to_dict()

    def track_across_frames(self, candidate_id: str, k: int | None = None) -> dict[str, Any]:
        """Gather independent measurements. Returns the spread the agent acts on."""
        aggregate = self.session.track_across_frames(candidate_id, k)
        result = aggregate.to_dict()
        import math

        if not math.isfinite(aggregate.iqr_ratio):
            result["iqr_ratio"] = None
            result["iqr_ratio_note"] = (
                "not establishable from the available views; treat as maximum distrust"
            )
        return result

    def ground_position(self, candidate_id: str) -> dict[str, Any]:
        position = self.session.ground_position(candidate_id)
        if position is None:
            return {"candidate_id": candidate_id, "position": None}
        return {
            "candidate_id": candidate_id,
            "lateral_m": position[0],
            "station_m": position[1],
        }

    def score_defect(self, candidate_id: str) -> dict[str, Any]:
        """Deterministic. Severity, priority and cost come from the rate table."""
        return self.session.score(candidate_id).to_dict()

    def run_summary(self) -> dict[str, Any]:
        return self.session.summary()

    def describe_environment(self) -> dict[str, Any]:
        """Which OpenCV build is serving these tools."""
        return self.environment

    # -- tool registry ----------------------------------------------------

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "load_source",
                "description": (
                    "Load a video file or a directory of images. Frames are sampled, "
                    "sharpness-gated, and privacy-scrubbed (faces and plates blurred) "
                    "before anything else can see them."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                },
            },
            {
                "name": "sample_frames",
                "description": "List the loaded frames with their sharpness.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "rectify",
                "description": (
                    "Rectify a frame's road surface to a metric bird's-eye view. "
                    "Returns a quality score and metres-per-pixel. A quality below "
                    "the configured minimum means nothing should be measured from "
                    "this frame; raise road_band_frac to retry over more of it."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "frame_id": {"type": "string"},
                        "road_band_frac": {
                            "type": "number",
                            "description": "Fraction of frame height treated as road, 0-0.75.",
                        },
                    },
                    "required": ["frame_id"],
                },
            },
            {
                "name": "propose_candidates",
                "description": (
                    "Cheap first pass over a rectified frame: resize, adaptive "
                    "Gaussian threshold, contour detection. This is the core image "
                    "workload. If it returns implausibly many candidates the "
                    "threshold is too loose - raise block_size and c_constant and "
                    "call again rather than refining noise."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "frame_id": {"type": "string"},
                        "scale": {"type": "number", "description": "Downscale factor, 0.05-1."},
                        "block_size": {"type": "integer", "description": "Adaptive window, odd."},
                        "c_constant": {"type": "number"},
                    },
                    "required": ["frame_id"],
                },
            },
            {
                "name": "classify_surface",
                "description": (
                    "Classify a candidate as pothole, shadow or wet_stain from "
                    "physical cues (edge transition width, contrast-normalised "
                    "texture, darkness, geometry). Confidence below the configured "
                    "minimum means the class is not established - re-examine with a "
                    "higher clahe_clip_limit, or escalate."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"candidate_id": {"type": "string"}},
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "refine_roi",
                "description": (
                    "Measure one candidate at full resolution inside its own ROI. "
                    "Returns area in m2 and the source-image footprint the "
                    "measurement rests on. measurable=false means the defect is too "
                    "distant or too small to size, and no area should be reported."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "clahe_clip_limit": {"type": "number"},
                        "roi_pad_frac": {"type": "number"},
                    },
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "track_across_frames",
                "description": (
                    "Follow one defect into k further frames and measure it in each, "
                    "returning the median area and iqr_ratio - the spread across "
                    "independent views. A high iqr_ratio means the measurement is "
                    "not yet trustworthy: call again with a larger k, and if it "
                    "stays high, escalate to a human. A null iqr_ratio means the "
                    "spread could not be established at all."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "k": {"type": "integer", "description": "Further frames to use."},
                    },
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "ground_position",
                "description": (
                    "A defect's position along the route (station) and across it "
                    "(lateral), in metres, from accumulated ego-motion. Two defects "
                    "at the same position are the same defect seen twice."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"candidate_id": {"type": "string"}},
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "score_defect",
                "description": (
                    "Deterministic severity, priority, repair cost and the approval "
                    "gate, from the published rate table. No model influences these "
                    "numbers. Returns needs_human_approval with explicit reasons; "
                    "priority 1 always requires approval."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"candidate_id": {"type": "string"}},
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "run_summary",
                "description": "Everything found in this run, with per-operation timings.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "describe_environment",
                "description": (
                    "Which OpenCV build is serving these tools: version, path, "
                    "whether it is the COOL build, and the instance type."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def handlers(self) -> dict[str, Callable[..., Any]]:
        return {
            "load_source": self.load_source,
            "sample_frames": self.sample_frames,
            "rectify": self.rectify,
            "propose_candidates": self.propose_candidates,
            "classify_surface": self.classify_surface,
            "refine_roi": self.refine_roi,
            "track_across_frames": self.track_across_frames,
            "ground_position": self.ground_position,
            "score_defect": self.score_defect,
            "run_summary": self.run_summary,
            "describe_environment": self.describe_environment,
        }

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self.handlers().get(name)
        if handler is None:
            raise UnknownTool(f"unknown tool: {name}")
        result = handler(**arguments)
        # Stamping the build onto every result means a captured trace proves by
        # itself which OpenCV served it, which is the COOL award's evidence
        # requirement and not something to be asserted separately in a report.
        return {
            "result": result,
            "served_by": {
                "opencv_version": self.environment["opencv_version"],
                "is_cool_build": self.environment["is_cool_build"],
                "architecture": self.environment["architecture"],
                "instance_type": self.environment["instance_type"],
            },
        }


def _response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def handle(server: ToolServer, request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}

    if method == "initialize":
        return _response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    if method in {"notifications/initialized", "initialized"}:
        return None  # a notification: no reply
    if method == "ping":
        return _response(request_id, {})
    if method == "tools/list":
        return _response(request_id, {"tools": server.tools()})
    if method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        try:
            payload = server.call(name, arguments)
        except UnknownTool as exception:
            return _error(request_id, -32601, str(exception))
        except Exception as exception:  # surfaced to the agent, not swallowed
            return _response(
                request_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "error": str(exception),
                                    "type": type(exception).__name__,
                                    "traceback": traceback.format_exc(limit=3),
                                }
                            ),
                        }
                    ],
                    "isError": True,
                },
            )
        return _response(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps(payload, default=str)}],
                "isError": False,
            },
        )

    return _error(request_id, -32601, f"method not found: {method}")


def serve(stdin=None, stdout=None) -> None:
    """Read JSON-RPC requests line by line and write responses."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = ToolServer()
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps(_error(None, -32700, "parse error")) + "\n")
            stdout.flush()
            continue
        response = handle(server, request)
        if response is not None:
            stdout.write(json.dumps(response, default=str) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
