"""Fit the surface classifier from labelled samples, with cross-validation.

Multinomial logistic regression on the eight physical cues: few parameters, so
it cannot memorise a small sample, and the weights stay readable - a judge or a
road engineer can see that darkness and edge width carry the pothole decision.

Samples currently come from the synthetic renderer, where the label is known by
construction. That is a stopgap and is stated as one: the decision boundary is
statistics and must be refit on RDD2022 before any accuracy claim is made about
real road imagery. The cue extractor in pipeline/cues.py is what transfers; this
file is what gets refit.

Run:  python -m eval.fit_classifier --out pipeline/classifier.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from eval.sampling import generate_samples
from pipeline.cues import CUE_NAMES

LABELS = ("pothole", "crack", "patch", "shadow", "wet_stain")

# A class with fewer samples than this cannot be fitted and is dropped rather
# than left in to produce a confidently useless column. `patch` hits this: a
# completed asphalt repair is close enough to road brightness that the proposal
# pass does not surface it at all, so the classifier is never shown one. That
# is acceptable - a repaired patch is not a defect - but it does mean the
# system cannot be claimed to inventory existing repairs.
MIN_SAMPLES_PER_CLASS = 10

# The operational decision is binary: does this cost public money to repair, or
# not? Pothole and crack are costed; shadow, wet stain and patch are recorded
# and not costed. Confusing a shadow for a wet stain changes nothing a crew
# does, while confusing either for a pothole dispatches a crew to nothing. So
# this grouping, not the five-way accuracy, is the number that matters.
DAMAGE_CLASSES = frozenset({"pothole", "crack"})


@dataclass
class Model:
    """Standardised multinomial logistic regression."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray  # (n_features, n_classes)
    bias: np.ndarray  # (n_classes,)
    labels: tuple[str, ...]
    feature_names: tuple[str, ...]

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        standardised = (features - self.mean) / self.scale
        logits = standardised @ self.weights + self.bias
        logits -= logits.max(axis=-1, keepdims=True)
        exponentiated = np.exp(logits)
        return exponentiated / exponentiated.sum(axis=-1, keepdims=True)

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "labels": list(self.labels),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Model":
        return cls(
            mean=np.array(payload["mean"], dtype=np.float64),
            scale=np.array(payload["scale"], dtype=np.float64),
            weights=np.array(payload["weights"], dtype=np.float64),
            bias=np.array(payload["bias"], dtype=np.float64),
            labels=tuple(payload["labels"]),
            feature_names=tuple(payload["feature_names"]),
        )


def fit(
    features: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    epochs: int = 4000,
    learning_rate: float = 0.25,
    l2: float = 1e-3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full-batch gradient descent on the cross-entropy loss.

    Returns (mean, scale, weights, bias). Standardisation statistics come only
    from the training fold, so cross-validation does not leak.
    """
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardised = (features - mean) / scale

    rng = np.random.default_rng(seed)
    n_features = standardised.shape[1]
    weights = rng.normal(0.0, 0.01, (n_features, n_classes))
    bias = np.zeros(n_classes)

    onehot = np.zeros((len(targets), n_classes))
    onehot[np.arange(len(targets)), targets] = 1.0

    for _ in range(epochs):
        logits = standardised @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        error = probabilities - onehot
        grad_w = standardised.T @ error / len(targets) + l2 * weights
        grad_b = error.mean(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return mean, scale, weights, bias


def cross_validate(
    features: np.ndarray, targets: np.ndarray, n_classes: int, folds: int = 5, seed: int = 0
) -> tuple[float, np.ndarray]:
    """Stratified k-fold accuracy and the pooled confusion matrix."""
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(targets), dtype=int)
    for label in range(n_classes):
        indices = np.where(targets == label)[0]
        rng.shuffle(indices)
        assignment[indices] = np.arange(len(indices)) % folds

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    correct = 0
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        mean, scale, weights, bias = fit(features[train], targets[train], n_classes, seed=seed)
        model = Model(mean, scale, weights, bias, LABELS[:n_classes], CUE_NAMES)
        predicted = model.probabilities(features[test]).argmax(axis=1)
        for true_label, predicted_label in zip(targets[test], predicted):
            confusion[true_label, predicted_label] += 1
        correct += int((predicted == targets[test]).sum())
    return correct / max(len(targets), 1), confusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="pipeline/classifier.json")
    parser.add_argument("--scenes", type=int, default=90)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    samples = generate_samples(n_scenes=args.scenes)

    raw_counts = {label: sum(1 for s in samples if s.label == label) for label in LABELS}
    present = [
        label
        for label in LABELS
        if raw_counts.get(label, 0) >= MIN_SAMPLES_PER_CLASS
    ]
    dropped = {
        label: count
        for label, count in raw_counts.items()
        if 0 < count < MIN_SAMPLES_PER_CLASS
    }
    if dropped:
        print(
            f"dropping under-represented class(es) {dropped} - fewer than "
            f"{MIN_SAMPLES_PER_CLASS} samples cannot be fitted"
        )
    samples = [s for s in samples if s.label in present]
    index = {label: i for i, label in enumerate(present)}

    features = np.array([s.features for s in samples], dtype=np.float64)
    targets = np.array([index[s.label] for s in samples], dtype=int)

    counts = {label: int((targets == i).sum()) for label, i in index.items()}
    print(f"samples: {len(samples)}  classes: {counts}")

    accuracy, confusion = cross_validate(features, targets, len(present), folds=args.folds)
    print(f"\n{args.folds}-fold cross-validated accuracy: {accuracy:.3f}")
    header = "              " + "".join(f"{label[:9]:>10}" for label in present)
    print("\nconfusion (rows = truth, columns = predicted)")
    print(header)
    for i, label in enumerate(present):
        row = "".join(f"{confusion[i, j]:>10d}" for j in range(len(present)))
        print(f"{label:>13} {row}")

    # The decision that actually matters: costed damage, or not.
    damage_truth = np.array([present[t] in DAMAGE_CLASSES for t in targets])
    binary_correct = 0
    false_damage = 0
    missed_damage = 0
    folds_rng = np.random.default_rng(0)
    assignment = np.empty(len(targets), dtype=int)
    for label in range(len(present)):
        indices = np.where(targets == label)[0]
        folds_rng.shuffle(indices)
        assignment[indices] = np.arange(len(indices)) % args.folds
    for fold in range(args.folds):
        test = assignment == fold
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        m, sc, w, b = fit(features[train], targets[train], len(present), seed=0)
        predicted = Model(m, sc, w, b, tuple(present), CUE_NAMES).probabilities(
            features[test]
        ).argmax(axis=1)
        predicted_damage = np.array([present[p] in DAMAGE_CLASSES for p in predicted])
        binary_correct += int((predicted_damage == damage_truth[test]).sum())
        false_damage += int((predicted_damage & ~damage_truth[test]).sum())
        missed_damage += int((~predicted_damage & damage_truth[test]).sum())

    print(
        f"\nbinary damage-vs-not accuracy: {binary_correct / len(targets):.4f}"
        f"   (false 'damage' {false_damage}, missed damage {missed_damage})"
    )
    print(
        "  <- the operational number: mistaking a shadow for a wet stain costs "
        "nothing,\n     mistaking either for a pothole dispatches a crew to nothing"
    )

    mean, scale, weights, bias = fit(features, targets, len(present))
    model = Model(mean, scale, weights, bias, tuple(present), CUE_NAMES)

    print("\nweights (standardised units; sign shows which way a cue pushes)")
    print("              " + "".join(f"{label[:9]:>10}" for label in present))
    for i, name in enumerate(CUE_NAMES):
        row = "".join(f"{weights[i, j]:>10.2f}" for j in range(len(present)))
        print(f"{name:>19} {row}"[:14 + 10 * len(present) + 5])

    payload = model.to_dict()
    payload["provenance"] = {
        "source": "synthetic renderer (tests/synthetic.py)",
        "n_samples": len(samples),
        "cross_validated_accuracy": round(accuracy, 4),
        "binary_damage_accuracy": round(binary_correct / max(len(targets), 1), 4),
        "false_damage": false_damage,
        "missed_damage": missed_damage,
        "dropped_classes": dropped,
        "folds": args.folds,
        "warning": (
            "Fitted on synthetic data. Refit on RDD2022 before claiming accuracy "
            "on real road imagery; the cue extractor transfers, this boundary may not."
        ),
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
