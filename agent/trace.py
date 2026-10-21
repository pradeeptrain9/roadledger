"""The decision trace.

The Agentic Vision award asks for evidence that an OpenCV result changed a
later decision, tool call or action. That evidence has to be *recorded as it
happens*, not reconstructed afterwards, so every step appends a record tying
three things together: the measurement that was observed, the rule that fired
on it, and the tool call that followed.

A trace is replayable. Each record carries the parameters used, so any step can
be re-run and checked - which is also what makes the whole run auditable to a
road authority rather than merely plausible.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Step:
    """One perception-decision-action cycle, or part of one."""

    index: int
    kind: str  # "perceive" | "decide" | "act" | "escalate" | "halt"
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    # The rule that fired, and the measured values that made it fire. This
    # pairing is the actual evidence that vision drove the decision.
    rule: str = ""
    because: dict[str, Any] = field(default_factory=dict)
    decision: str = ""
    elapsed_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trace:
    """The full record of one agent run."""

    run_id: str
    source: str = ""
    steps: list[Step] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    halted_reason: str = ""

    def add(
        self,
        kind: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        rule: str = "",
        because: dict[str, Any] | None = None,
        decision: str = "",
        elapsed_ms: float = 0.0,
    ) -> Step:
        step = Step(
            index=len(self.steps),
            kind=kind,
            tool=tool,
            arguments=arguments or {},
            observation=observation or {},
            rule=rule,
            because=because or {},
            decision=decision,
            elapsed_ms=round(elapsed_ms, 3),
        )
        self.steps.append(step)
        return step

    @property
    def tool_calls(self) -> int:
        return sum(1 for step in self.steps if step.kind in {"perceive", "act"})

    def decisions(self) -> list[Step]:
        """Only the steps where a rule fired on a measurement."""
        return [step for step in self.steps if step.rule]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source": self.source,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "halted_reason": self.halted_reason,
            "tool_calls": self.tool_calls,
            "n_steps": len(self.steps),
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary_lines(self) -> list[str]:
        """Human-readable trace, for the demo and the report."""
        lines = []
        for step in self.steps:
            prefix = {
                "perceive": "  see ",
                "decide": "  >>> ",
                "act": "  do  ",
                "escalate": "  !!! ",
                "halt": "  == ",
            }.get(step.kind, "      ")
            if step.rule:
                because = ", ".join(f"{k}={v}" for k, v in step.because.items())
                lines.append(f"{prefix}{step.rule}: {because}")
                if step.decision:
                    lines.append(f"       -> {step.decision}")
            else:
                detail = ", ".join(f"{k}={v}" for k, v in list(step.observation.items())[:4])
                lines.append(f"{prefix}{step.tool}({_short(step.arguments)}) {detail}")
        return lines


def _short(arguments: dict[str, Any]) -> str:
    parts = []
    for key, value in arguments.items():
        text = str(value)
        parts.append(f"{key}={text[:18] + '...' if len(text) > 18 else text}")
    return ", ".join(parts)
