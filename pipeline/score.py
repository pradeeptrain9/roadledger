"""Stage 6 and 7 - deterministic severity, priority, cost and the approval gate.

No language model touches any number in this file. The agent decides which
measurements to take; this module decides what they mean, from a published
rate table, so two runs on the same measurements always produce the same work
order and a road engineer can audit the arithmetic.

The gate fails closed. A defect with no trustworthy measurement does not become
a cheap work order - it becomes a request for human attention.
"""

from __future__ import annotations

import hashlib
import math

from .config import AgentThresholds, ScoreConfig
from .types import Aggregate, Classification, Defect

# Classes that describe pavement damage. Everything else is recorded and
# excluded from repair costing.
DAMAGE_CLASSES = frozenset({"pothole", "crack"})
# An already-completed repair: worth logging as asset inventory, never billed.
INVENTORY_CLASSES = frozenset({"patch"})


def _defect_id(candidate_id: str, label: str) -> str:
    digest = hashlib.sha1(f"{candidate_id}:{label}".encode()).hexdigest()[:12]
    return f"d_{digest}"


def severity(area_m2: float, label: str, config: ScoreConfig) -> float:
    """Severity in [0, 1], driven by area relative to the urgent threshold.

    Area is the quantity crews mobilise against and the one the pipeline
    actually measures, so it carries the score. A crack of a given area is less
    urgent than a pothole of the same area because it is not an impact hazard,
    which the class weighting reflects.
    """
    if area_m2 <= 0:
        return 0.0
    normalised = min(1.0, area_m2 / max(config.priority1_min_m2, 1e-9))
    class_weight = {"pothole": 1.0, "crack": 0.6}.get(label, 0.3)
    return round(min(1.0, normalised * class_weight), 4)


def priority(area_m2: float, label: str, config: ScoreConfig) -> int:
    """1 is urgent, 2 is scheduled, 3 is monitor. Thresholds are published."""
    if label not in DAMAGE_CLASSES:
        return 3
    if area_m2 >= config.priority1_min_m2:
        return 1 if label == "pothole" else 2
    if area_m2 >= config.priority2_min_m2:
        return 2
    return 3


def cost_estimate(area_m2: float, label: str, config: ScoreConfig) -> tuple[float, bool]:
    """Repair cost and whether the defect is billable at all.

    Returns (cost, billable). Below the minimum billable area a crew would not
    mobilise, so the defect is monitored rather than priced - quoting a
    mobilisation charge to patch a 2 cm blemish is how a system loses the trust
    of the people using it.
    """
    if label not in DAMAGE_CLASSES:
        return 0.0, False
    if area_m2 < config.min_billable_m2:
        return 0.0, False
    rate = config.unit_rate_per_m2.get(label, 0.0)
    if rate <= 0:
        return 0.0, False
    return round(config.mobilisation_cost + rate * area_m2, 2), True


def score_defect(
    aggregate: Aggregate,
    classification: Classification,
    config: ScoreConfig,
    thresholds: AgentThresholds,
) -> Defect:
    """Turn a measurement aggregate plus a class into a ledger-ready defect.

    Every reason the defect needs human attention is recorded explicitly, so
    the person reviewing it sees why it reached them.
    """
    reasons: list[str] = []
    label = classification.label
    area = aggregate.area_m2_median
    ratio = aggregate.iqr_ratio

    needs_approval = False

    if aggregate.n_measurements == 0:
        reasons.append("no trustworthy measurement was obtained")
        needs_approval = True
    elif aggregate.n_measurements < 3:
        reasons.append(
            f"only {aggregate.n_measurements} independent measurement(s); "
            "spread is not yet meaningful"
        )
        needs_approval = True

    if not math.isfinite(ratio):
        reasons.append("measurement spread could not be established")
        needs_approval = True
    elif ratio > thresholds.escalate_to_human_iqr_ratio:
        reasons.append(
            f"measurements disagree by {ratio:.0%} of the median, above the "
            f"{thresholds.escalate_to_human_iqr_ratio:.0%} escalation threshold"
        )
        needs_approval = True
    elif ratio > thresholds.max_area_iqr_ratio:
        reasons.append(
            f"measurements disagree by {ratio:.0%} of the median, above the "
            f"{thresholds.max_area_iqr_ratio:.0%} confidence threshold"
        )

    if classification.confidence < thresholds.min_class_confidence:
        reasons.append(
            f"class '{label}' carries only {classification.confidence:.0%} confidence, "
            f"below the {thresholds.min_class_confidence:.0%} threshold"
        )
        needs_approval = True

    if label == "unknown":
        reasons.append("surface class could not be determined")
        needs_approval = True

    defect_priority = priority(area, label, config)
    cost, billable = cost_estimate(area, label, config)

    # Urgent work is never dispatched unreviewed, however confident the
    # measurement. A system that can commit public money on its own judgement
    # is not one a road authority would adopt.
    if defect_priority <= thresholds.always_approve_priority and label in DAMAGE_CLASSES:
        reasons.append(f"priority {defect_priority} always requires human approval")
        needs_approval = True

    if label in INVENTORY_CLASSES:
        reasons.append("existing repair: recorded as inventory, not costed")
    elif label not in DAMAGE_CLASSES and label != "unknown":
        reasons.append(f"'{label}' is not pavement damage: recorded, not costed")

    if label in DAMAGE_CLASSES and not billable and aggregate.n_measurements > 0:
        reasons.append(
            f"area {area:.3f} m2 is below the {config.min_billable_m2} m2 billable "
            "minimum: monitor rather than repair"
        )

    return Defect(
        defect_id=_defect_id(aggregate.candidate_id, label),
        label=label,
        class_confidence=classification.confidence,
        area_m2=area,
        area_iqr_ratio=ratio if math.isfinite(ratio) else -1.0,
        n_measurements=aggregate.n_measurements,
        severity=severity(area, label, config),
        priority=defect_priority,
        cost_estimate=cost,
        currency=config.currency,
        billable=billable,
        needs_human_approval=needs_approval,
        reasons=reasons,
        frame_ids=list(aggregate.frame_ids),
    )
