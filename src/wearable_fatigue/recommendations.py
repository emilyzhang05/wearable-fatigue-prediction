from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkoutRecommendation:
    action: str
    best_window: str
    reason: str


def recommend_workout_plan(
    fatigue_risk: str,
    sleep_debt_hours: float,
    resting_hr_deviation: float,
    previous_activity_load: float,
) -> WorkoutRecommendation:
    risk = fatigue_risk.lower()

    if risk == "high" and sleep_debt_hours >= 1.25:
        return WorkoutRecommendation(
            action="rest",
            best_window="tomorrow morning",
            reason="high fatigue risk with meaningful sleep debt suggests prioritizing recovery.",
        )

    if risk == "high":
        return WorkoutRecommendation(
            action="reschedule workout",
            best_window="evening or next morning",
            reason="elevated fatigue signal means intense training is better delayed.",
        )

    if risk == "medium" or resting_hr_deviation >= 4:
        return WorkoutRecommendation(
            action="light workout",
            best_window="late afternoon",
            reason="moderate recovery strain suggests lowering workout intensity.",
        )

    if previous_activity_load < 8:
        return WorkoutRecommendation(
            action="normal workout",
            best_window="morning",
            reason="low fatigue and modest recent activity support a normal session.",
        )

    return WorkoutRecommendation(
        action="normal workout",
        best_window="preferred routine window",
        reason="fatigue signals are low enough to continue the planned routine.",
    )
