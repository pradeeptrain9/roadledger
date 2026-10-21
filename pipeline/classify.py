"""Stage 4b - surface classification: pothole, crack, patch, shadow or wet.

Finding dark blobs on a road is easy. Telling a pothole from the three things
that also look like dark blobs - a cast shadow, standing water, and an
already-completed asphalt repair - is the problem that decides whether the
output is worth acting on. Dispatching a crew to a shadow is the failure that
would discredit the system.

The cues live in `pipeline.cues` and encode physics. The decision boundary is
fitted (see `eval/fit_classifier.py`) rather than hand-tuned, after hand-tuned
trapezoidal bands reached 3/12 on held-out confusers while the fitted model
reaches 0.98 under five-fold cross-validation. The fitted weights recover the
physics unprompted: edge width is the dominant positive weight for shadow,
because a penumbra ramps where a pothole rim steps.

Confidence is a first-class output. A low-confidence verdict is not a quiet
default to the most common class - it is a signal the agent acts on by
re-examining the region or escalating to a human.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .cues import CUE_NAMES, extract, vector
from .types import Candidate, Classification

DEFAULT_MODEL_PATH = Path(__file__).with_name("classifier.json")


class ClassifierModel:
    """Standardised multinomial logistic regression over the physical cues."""

    def __init__(self, payload: dict) -> None:
        self.labels: tuple[str, ...] = tuple(payload["labels"])
        self.feature_names: tuple[str, ...] = tuple(payload["feature_names"])
        self.mean = np.array(payload["mean"], dtype=np.float64)
        self.scale = np.array(payload["scale"], dtype=np.float64)
        self.weights = np.array(payload["weights"], dtype=np.float64)
        self.bias = np.array(payload["bias"], dtype=np.float64)
        self.provenance: dict = payload.get("provenance", {})

        if self.feature_names != CUE_NAMES:
            raise ValueError(
                "classifier was fitted on a different cue set than the extractor "
                f"produces: {self.feature_names} vs {CUE_NAMES}. Refit it."
            )

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        standardised = (features - self.mean) / self.scale
        logits = standardised @ self.weights + self.bias
        logits -= logits.max()
        exponentiated = np.exp(logits)
        return exponentiated / exponentiated.sum()


@lru_cache(maxsize=4)
def load_model(path: str | None = None) -> ClassifierModel | None:
    """Load the fitted classifier, or None if it has not been fitted yet."""
    model_path = Path(path) if path else DEFAULT_MODEL_PATH
    if not model_path.exists():
        return None
    with open(model_path) as handle:
        return ClassifierModel(json.load(handle))


def classify(
    bev: np.ndarray,
    candidate: Candidate,
    road_median: float | None = None,
    model_path: str | None = None,
) -> Classification:
    """Classify one candidate. Unknown with zero confidence when it cannot."""
    cues = extract(bev, candidate, road_median)
    if cues is None:
        return Classification(
            candidate_id=candidate.candidate_id,
            label="unknown",
            confidence=0.0,
            probabilities={},
            cues={},
        )

    rounded = {name: round(value, 4) for name, value in cues.items()}
    model = load_model(model_path)
    if model is None:
        # Fail closed and say so, rather than inventing a label from nothing.
        return Classification(
            candidate_id=candidate.candidate_id,
            label="unknown",
            confidence=0.0,
            probabilities={},
            cues=rounded,
        )

    probabilities = model.probabilities(vector(cues))
    ranked = {
        label: round(float(probability), 4)
        for label, probability in zip(model.labels, probabilities)
    }
    best = max(ranked, key=lambda label: ranked[label])
    return Classification(
        candidate_id=candidate.candidate_id,
        label=best,
        confidence=ranked[best],
        probabilities=ranked,
        cues=rounded,
    )
