"""Language-model driver: a Bedrock model sequences the same tool calls.

The deterministic policy in agent/rules.py is the reference driver and is what
the evaluation runs against, because a reproducible number needs a reproducible
decision. This driver exists because the loop should also work when the
sequencing is chosen by a model rather than by a rule table - the award permits
any framework or model, and a model can combine evidence in ways a fixed rule
order cannot, such as deciding that a frame is not worth pursuing at all.

The division of labour is the one used throughout this codebase: **the model
chooses which measurement to take next; the engine computes every number.** The
model is never asked for an area, a severity, a priority or a cost. It cannot
emit a work order directly either - `score_defect` applies the published rate
table and the approval gate, and the model only decides when to call it. A
model that hallucinates will waste tool calls, which the budget bounds; it
cannot produce a wrong measurement.

Tool definitions are generated from the MCP server's own registry, so the two
drivers cannot drift apart.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.trace import Trace
from mcp.opencv_mcp_server import ToolServer

# Claude Haiku 4.5 via a cross-region inference profile: this loop makes many
# small tool-selection calls, where latency and cost matter more than depth,
# and the hard reasoning is in the rule table rather than in the model.
DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

SYSTEM_PROMPT = """\
You are sequencing measurements of road-surface defects from dashcam video.

You decide WHICH measurement to take next. You never state a measurement
yourself. Areas, severities, priorities and repair costs come only from the
tools; if you need one, call the tool.

How to work:

1. rectify a frame. If quality is below 0.35, retry once with a larger
   road_band_frac. If it is still low, move to the next frame - nothing may be
   measured from geometry you do not trust.
2. propose_candidates. If it returns more than 25 candidates the threshold is
   too loose: raise block_size and c_constant and call it again rather than
   refining noise.
3. For each candidate worth pursuing: classify_surface, then refine_roi.
   - If measurable is false, stop with that candidate. It is too distant to
     size and no area may be reported for it.
   - If class confidence is below 0.55, call refine_roi again with a higher
     clahe_clip_limit and re-classify. If it is still below 0.55, escalate.
4. track_across_frames to get independent measurements. Read iqr_ratio:
   - at or below 0.20, the measurement is trustworthy;
   - above 0.20, call track_across_frames again with a larger k;
   - above 0.35 after that, or null, escalate to a human.
5. score_defect. It returns needs_human_approval with reasons. Respect it.

Prefer admitting that something cannot be measured over producing a number you
are unsure of. A confident wrong area sends a crew to the wrong place.

When there is nothing further worth measuring, stop and say what you found.
"""


class ConverseClient(Protocol):
    """The slice of bedrock-runtime this driver uses."""

    def converse(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass
class LlmOutcome:
    run_id: str
    trace: Trace
    tool_calls: int = 0
    stop_reason: str = ""
    final_text: str = ""
    model_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


def tool_config(server: ToolServer, exclude: set[str] | None = None) -> dict[str, Any]:
    """Bedrock toolConfig built from the MCP registry, so they cannot diverge."""
    exclude = exclude or set()
    specs = []
    for tool in server.tools():
        if tool["name"] in exclude:
            continue
        specs.append(
            {
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": {"json": tool["inputSchema"]},
                }
            }
        )
    return {"tools": specs}


def make_client(region: str = "us-east-1") -> ConverseClient:
    import boto3

    return boto3.client("bedrock-runtime", region_name=region)


def run(
    source: str,
    client: ConverseClient | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    max_turns: int = 40,
    max_tool_calls: int = 60,
    region: str = "us-east-1",
    server: ToolServer | None = None,
) -> LlmOutcome:
    """Let the model drive the tool surface until it stops or runs out of budget."""
    server = server or ToolServer()
    client = client or make_client(region)

    trace = Trace(run_id=f"llm-{int(time.time())}", source=source)
    outcome = LlmOutcome(run_id=trace.run_id, trace=trace, model_id=model_id)

    # The source is loaded before the model is involved. Letting a model choose
    # what to load would let prompt content pick paths on the worker.
    loaded = server.call("load_source", {"source": source})
    trace.add("perceive", "load_source", {"source": source}, loaded["result"])

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        f"Loaded {loaded['result'].get('frames_sampled')} frames from "
                        f"{source}. Frame ids: "
                        f"{json.dumps(loaded['result'].get('frame_ids', [])[:12])}. "
                        "Measure the road defects in this footage."
                    )
                }
            ],
        }
    ]
    config = tool_config(server, exclude={"load_source"})

    for _ in range(max_turns):
        if outcome.tool_calls >= max_tool_calls:
            trace.add(
                "halt",
                "",
                rule="budget_exhausted",
                because={"tool_calls": outcome.tool_calls, "maximum": max_tool_calls},
                decision="halt: tool-call budget reached",
            )
            trace.halted_reason = "tool-call budget reached"
            break

        response = client.converse(
            modelId=model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=config,
            inferenceConfig={"maxTokens": 2048, "temperature": 0.0},
        )
        outcome.usage = response.get("usage", {}) or outcome.usage
        outcome.stop_reason = response.get("stopReason", "")
        output_message = response["output"]["message"]
        messages.append(output_message)

        tool_uses = [
            block["toolUse"] for block in output_message.get("content", []) if "toolUse" in block
        ]
        texts = [
            block["text"] for block in output_message.get("content", []) if "text" in block
        ]
        if texts:
            outcome.final_text = "\n".join(texts)

        if not tool_uses:
            break

        results = []
        for use in tool_uses:
            name = use["name"]
            arguments = use.get("input") or {}
            started = time.perf_counter()
            try:
                payload = server.call(name, arguments)
                content = payload
                status = "success"
            except Exception as exception:
                # Returned to the model rather than raised, so it can choose a
                # different action - that recovery is part of the behaviour
                # under test.
                content = {"error": str(exception), "type": type(exception).__name__}
                status = "error"
            elapsed = (time.perf_counter() - started) * 1000
            outcome.tool_calls += 1
            trace.add(
                "perceive" if status == "success" else "escalate",
                name,
                arguments,
                content.get("result", content) if isinstance(content, dict) else {},
                elapsed_ms=elapsed,
            )
            results.append(
                {
                    "toolResult": {
                        "toolUseId": use["toolUseId"],
                        "content": [{"json": content}],
                        "status": status,
                    }
                }
            )
        messages.append({"role": "user", "content": results})

    trace.finished_at = time.time()
    return outcome
