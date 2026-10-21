"""Fit the surface classifier on RDD2022 — real road imagery.

Replaces a boundary fitted to the synthetic renderer, which did not transfer:
on real frames it labelled 35 of 45 detected potholes "shadow", and 2 as
pothole.

Three classes here, not five. Real annotations distinguish pothole from crack
and say nothing at all about shadows or standing water, so the honest label set
is {pothole, crack, not_damage}. The synthetic shadow/wet distinction is
dropped rather than carried over from data that cannot support it.

Two things the imbalance demands. 79% of samples are not_damage, so plain
accuracy is a trap - predicting "not damage" for everything scores 0.79 and is
worthless. The report therefore leads with **precision and recall on damage**,
which is what decides whether a crew is dispatched. And the fit is
class-weighted, so the majority class cannot simply absorb the minority.

Run:  python -m eval.fit_real_classifier --limit 300
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from eval.fit_classifier import Model
from eval.real_sampling import generate_real_samples
from pipeline.config import ProposeParams
from pipeline.cues import CUE_NAMES

LABELS = ("pothole", "crack", "not_damage")
DAMAGE = {"pothole", "crack"}


def fit_weighted(
    features: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    epochs: int = 6000,
    learning_rate: float = 0.3,
    l2: float = 3e-3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Class-weighted multinomial logistic regression.

    Weights are inverse to class frequency, so the 79% not_damage majority
    cannot win by absorbing the minority classes.
    """
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardised = (features - mean) / scale

    counts = np.bincount(targets, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights_per_class = len(targets) / (n_classes * counts)
    sample_weights = weights_per_class[targets]
    sample_weights = sample_weights / sample_weights.mean()

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, (standardised.shape[1], n_classes))
    bias = np.zeros(n_classes)
    onehot = np.zeros((len(targets), n_classes))
    onehot[np.arange(len(targets)), targets] = 1.0

    for _ in range(epochs):
        logits = standardised @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        error = (probabilities - onehot) * sample_weights[:, None]
        weights -= learning_rate * (standardised.T @ error / len(targets) + l2 * weights)
        bias -= learning_rate * error.mean(axis=0)

    return mean, scale, weights, bias


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/rdd2022/India")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default="pipeline/classifier_real.json")
    parser.add_argument("--block-size", type=int, default=51)
    parser.add_argument("--c-constant", type=float, default=3.0)
    parser.add_argument("--min-area", type=float, default=180.0)
    args = parser.parse_args()

    propose_params = replace(
        ProposeParams(),
        block_size=args.block_size,
        c_constant=args.c_constant,
        min_area_px=args.min_area,
    )
    samples = generate_real_samples(
        root=args.root, limit=args.limit, propose_params=propose_params
    )
    present = [label for label in LABELS if any(s.label == label for s in samples)]
    index = {label: i for i, label in enumerate(present)}
    features = np.array([s.features for s in samples], dtype=np.float64)
    targets = np.array([index[s.label] for s in samples], dtype=int)
    print(f"samples: {len(samples)} from {len(set(s.image_name for s in samples))} real images")
    print(f"  {{{', '.join(f'{l}: {int((targets == i).sum())}' for l, i in index.items())}}}")
    print(f"  majority-class baseline accuracy: {np.bincount(targets).max()/len(targets):.3f}")

    # Grouped by image: candidates from one frame share lighting and surface,
    # so splitting them across folds would leak and flatter the score.
    images = np.array([s.image_name for s in samples])
    unique_images = np.unique(images)
    rng = np.random.default_rng(0)
    rng.shuffle(unique_images)
    fold_of_image = {name: i % args.folds for i, name in enumerate(unique_images)}
    assignment = np.array([fold_of_image[name] for name in images])

    confusion = np.zeros((len(present), len(present)), dtype=int)
    for fold in range(args.folds):
        test = assignment == fold
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        mean, scale, weights, bias = fit_weighted(features[train], targets[train], len(present))
        model = Model(mean, scale, weights, bias, tuple(present), CUE_NAMES)
        predicted = model.probabilities(features[test]).argmax(axis=1)
        for truth, guess in zip(targets[test], predicted):
            confusion[truth, guess] += 1

    print(f"\n{args.folds}-fold grouped-by-image cross-validation")
    print("confusion (rows = truth, columns = predicted)")
    print("              " + "".join(f"{l[:10]:>12}" for l in present))
    for i, label in enumerate(present):
        print(f"{label:>13} " + "".join(f"{confusion[i, j]:>12d}" for j in range(len(present))))

    damage_idx = [index[l] for l in present if l in DAMAGE]
    not_idx = [index[l] for l in present if l not in DAMAGE]
    tp = confusion[np.ix_(damage_idx, damage_idx)].sum()
    fn = confusion[np.ix_(damage_idx, not_idx)].sum()
    fp = confusion[np.ix_(not_idx, damage_idx)].sum()
    tn = confusion[np.ix_(not_idx, not_idx)].sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(confusion.sum(), 1)

    print(f"\nthe decision that matters - is this costed damage?")
    print(f"  precision {precision:.3f}   recall {recall:.3f}   F1 {f1:.3f}   accuracy {accuracy:.3f}")
    print(f"  of {tp + fn} real damage candidates, {tp} were called damage and {fn} missed")
    print(f"  of {fp + tn} non-damage candidates, {fp} were wrongly called damage")
    print("  (precision governs crews dispatched to nothing; recall governs damage left unrepaired)")

    mean, scale, weights, bias = fit_weighted(features, targets, len(present))
    model = Model(mean, scale, weights, bias, tuple(present), CUE_NAMES)
    print("\nweights (standardised; sign shows which way a cue pushes)")
    print("              " + "".join(f"{l[:10]:>12}" for l in present))
    for i, name in enumerate(CUE_NAMES):
        print(f"{name:>19} " + "".join(f"{weights[i, j]:>12.2f}" for j in range(len(present))))

    payload = model.to_dict()
    payload["provenance"] = {
        "source": f"RDD2022 real imagery ({args.root})",
        "n_samples": len(samples),
        "n_images": int(len(unique_images)),
        "damage_precision": round(float(precision), 4),
        "damage_recall": round(float(recall), 4),
        "damage_f1": round(float(f1), 4),
        "majority_baseline": round(float(np.bincount(targets).max() / len(targets)), 4),
        "cross_validation": "grouped by source image",
        "labels_from": "RDD2022 Pascal VOC annotations (weak supervision)",
        "note": (
            "Three classes only. Real annotations distinguish pothole from crack and "
            "say nothing about shadows or standing water, so those classes are not "
            "fitted here rather than being carried over from synthetic data."
        ),
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
