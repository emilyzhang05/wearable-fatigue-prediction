from __future__ import annotations


def evaluate_hypotheses(
    classification_metrics: dict[str, dict[str, float]],
    regression_metrics: dict[str, dict[str, float]],
    behavior_profiles: list[str],
) -> dict[str, dict[str, str]]:
    baseline_f1 = classification_metrics.get("baseline_most_frequent", {}).get("f1_macro", 0.0)
    best_f1 = max((metrics.get("f1_macro", 0.0) for metrics in classification_metrics.values()), default=0.0)

    baseline_mae = regression_metrics.get("baseline_mean", {}).get("mae", float("inf"))
    best_mae = min((metrics.get("mae", float("inf")) for metrics in regression_metrics.values()), default=float("inf"))

    distinct_profiles = len(set(behavior_profiles))

    return {
        "H1": {
            "statement": "Wearable signals reproduce the fatigue-risk proxy better than baseline.",
            "status": _status(best_f1 >= baseline_f1 + 0.05),
            "evidence": f"Best macro F1 {best_f1:.3f} vs baseline {baseline_f1:.3f}.",
        },
        "H2": {
            "statement": "Recent sleep/activity predicts next-day active minutes better than baseline.",
            "status": _status(best_mae <= baseline_mae * 0.95),
            "evidence": f"Best MAE {best_mae:.1f} vs baseline {baseline_mae:.1f}.",
        },
        "H3": {
            "statement": "Users can be grouped into interpretable behavior profiles.",
            "status": _status(distinct_profiles >= 2),
            "evidence": f"Identified {distinct_profiles} behavior profiles.",
        },
    }


def _status(condition: bool) -> str:
    return "supported" if condition else "not supported"
