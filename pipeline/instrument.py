"""Per-operation timing, shared by the pipeline and the benchmark harness.

The COOL award asks for evidence that COOL executes the claimed core workload.
The most direct evidence is per-operation timing collected from inside the
running application rather than from a separate microbenchmark, so the same
instrumentation ships in the production path and is simply read out by the
benchmark.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Timings:
    """Accumulated wall-clock time per named operation, in milliseconds."""

    total_ms: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.total_ms[name] = self.total_ms.get(name, 0.0) + elapsed_ms
            self.calls[name] = self.calls.get(name, 0) + 1

    def mean_ms(self, name: str) -> float:
        count = self.calls.get(name, 0)
        return self.total_ms.get(name, 0.0) / count if count else 0.0

    def merge(self, other: "Timings") -> None:
        for name, value in other.total_ms.items():
            self.total_ms[name] = self.total_ms.get(name, 0.0) + value
        for name, count in other.calls.items():
            self.calls[name] = self.calls.get(name, 0) + count

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "total_ms": round(self.total_ms[name], 4),
                "calls": self.calls[name],
                "mean_ms": round(self.mean_ms(name), 4),
            }
            for name in sorted(self.total_ms)
        }
