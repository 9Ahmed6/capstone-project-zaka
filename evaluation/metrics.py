from __future__ import annotations

from collections import Counter


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Intersection-over-union for two time ranges."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def match_predictions(predictions: list[dict], ground_truth: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Match each prediction with the best unused ground-truth segment."""
    matches = []
    used_truth = set()

    for pred_index, prediction in enumerate(predictions):
        best_truth_index = None
        best_iou = 0.0

        for truth_index, truth in enumerate(ground_truth):
            if truth_index in used_truth:
                continue

            iou = temporal_iou(
                prediction["start_time"],
                prediction["end_time"],
                truth["start_time"],
                truth["end_time"],
            )
            if iou > best_iou:
                best_iou = iou
                best_truth_index = truth_index

        if best_truth_index is not None and best_iou >= iou_threshold:
            used_truth.add(best_truth_index)
            matches.append(
                {
                    "prediction_index": pred_index,
                    "truth_index": best_truth_index,
                    "iou": best_iou,
                    "pred_label": prediction.get("action_label", "unknown"),
                    "truth_label": ground_truth[best_truth_index].get("action_label", "unknown"),
                }
            )

    return matches


def macro_f1(predictions: list[dict], ground_truth: list[dict], iou_threshold: float = 0.5) -> float:
    """Compute macro-F1 after temporal matching."""
    matches = match_predictions(predictions, ground_truth, iou_threshold)
    labels = sorted(
        {
            item.get("action_label", "unknown")
            for item in predictions + ground_truth
        }
    )

    if not labels:
        return 0.0

    true_positive = Counter()
    false_positive = Counter(item.get("action_label", "unknown") for item in predictions)
    false_negative = Counter(item.get("action_label", "unknown") for item in ground_truth)

    for match in matches:
        if match["pred_label"] == match["truth_label"]:
            label = match["truth_label"]
            true_positive[label] += 1
            false_positive[label] -= 1
            false_negative[label] -= 1

    f1_scores = []
    for label in labels:
        tp = true_positive[label]
        fp = max(0, false_positive[label])
        fn = max(0, false_negative[label])

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def evaluate_predictions(predictions: list[dict], ground_truth: list[dict], iou_threshold: float = 0.5) -> dict:
    """Return the Week 5 baseline metrics."""
    matches = match_predictions(predictions, ground_truth, iou_threshold)
    mean_iou = sum(item["iou"] for item in matches) / len(matches) if matches else 0.0

    return {
        "matched_segments": len(matches),
        "prediction_segments": len(predictions),
        "ground_truth_segments": len(ground_truth),
        "mean_temporal_iou": mean_iou,
        "macro_f1": macro_f1(predictions, ground_truth, iou_threshold),
        "iou_threshold": iou_threshold,
    }

