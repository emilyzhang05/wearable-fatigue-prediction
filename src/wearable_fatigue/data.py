from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    activity_archetype: str
    baseline_sleep_hours: float
    baseline_resting_hr: float
    baseline_steps: int
    workout_preference_hour: int


ARCHETYPES = (
    "consistent_exerciser",
    "weekend_active",
    "sedentary_weekday",
    "high_activity_low_recovery",
)


def generate_synthetic_wearable_data(
    n_users: int = 18,
    days: int = 60,
    random_state: int = 42,
    start_date: date = date(2026, 1, 1),
) -> pd.DataFrame:
    if n_users <= 0:
        raise ValueError("n_users must be positive")
    if days <= 7:
        raise ValueError("days must be greater than 7 for rolling features")

    rng = np.random.default_rng(random_state)
    profiles = [_build_user_profile(index, rng) for index in range(n_users)]
    rows: list[dict[str, object]] = []

    for profile in profiles:
        for day_index in range(days):
            current_date = start_date + timedelta(days=day_index)
            weekday = current_date.weekday()
            is_weekend = weekday >= 5
            routine_noise = rng.normal(0, 1)

            sleep_hours = _sample_sleep_hours(profile, is_weekend, routine_noise, rng)
            workout_intensity = _sample_workout_intensity(profile, is_weekend, rng)
            active_minutes = _sample_active_minutes(profile, workout_intensity, is_weekend, rng)
            steps = _sample_steps(profile, active_minutes, is_weekend, rng)
            resting_heart_rate = _sample_resting_hr(
                profile=profile,
                sleep_hours=sleep_hours,
                workout_intensity=workout_intensity,
                active_minutes=active_minutes,
                rng=rng,
            )

            rows.append(
                {
                    "user_id": profile.user_id,
                    "date": pd.Timestamp(current_date),
                    "weekday": weekday,
                    "is_weekend": int(is_weekend),
                    "activity_archetype": profile.activity_archetype,
                    "sleep_hours": round(sleep_hours, 2),
                    "resting_heart_rate": round(resting_heart_rate, 1),
                    "average_heart_rate": round(resting_heart_rate + rng.normal(18, 6), 1),
                    "steps": int(max(500, steps)),
                    "active_minutes": int(max(0, active_minutes)),
                    "workout_intensity": round(float(np.clip(workout_intensity, 0, 1)), 2),
                    "calories_burned": int(1450 + steps * 0.045 + active_minutes * 4.5 + rng.normal(0, 80)),
                    "workout_preference_hour": profile.workout_preference_hour,
                }
            )

    return pd.DataFrame(rows).sort_values(["user_id", "date"]).reset_index(drop=True)


def _build_user_profile(index: int, rng: np.random.Generator) -> UserProfile:
    archetype = ARCHETYPES[index % len(ARCHETYPES)]
    baseline_steps = int(rng.normal(8200, 1700))

    if archetype == "sedentary_weekday":
        baseline_steps = int(rng.normal(5200, 1000))
    elif archetype == "high_activity_low_recovery":
        baseline_steps = int(rng.normal(10500, 1500))

    return UserProfile(
        user_id=f"user_{index + 1:02d}",
        activity_archetype=archetype,
        baseline_sleep_hours=float(rng.normal(7.2, 0.55)),
        baseline_resting_hr=float(rng.normal(62, 6)),
        baseline_steps=max(2500, baseline_steps),
        workout_preference_hour=int(rng.choice([7, 12, 18, 20])),
    )


def _sample_sleep_hours(
    profile: UserProfile,
    is_weekend: bool,
    routine_noise: float,
    rng: np.random.Generator,
) -> float:
    weekend_boost = 0.45 if is_weekend and profile.activity_archetype != "weekend_active" else 0.1
    low_recovery_penalty = -0.65 if profile.activity_archetype == "high_activity_low_recovery" else 0
    sleep_hours = profile.baseline_sleep_hours + weekend_boost + low_recovery_penalty
    sleep_hours += routine_noise * 0.35 + rng.normal(0, 0.35)
    return float(np.clip(sleep_hours, 3.8, 10.2))


def _sample_workout_intensity(
    profile: UserProfile,
    is_weekend: bool,
    rng: np.random.Generator,
) -> float:
    archetype_base = {
        "consistent_exerciser": 0.62,
        "weekend_active": 0.35 + (0.35 if is_weekend else 0),
        "sedentary_weekday": 0.24 + (0.12 if is_weekend else 0),
        "high_activity_low_recovery": 0.76,
    }[profile.activity_archetype]
    return float(np.clip(rng.normal(archetype_base, 0.18), 0, 1))


def _sample_active_minutes(
    profile: UserProfile,
    workout_intensity: float,
    is_weekend: bool,
    rng: np.random.Generator,
) -> float:
    weekend_effect = 18 if is_weekend and profile.activity_archetype == "weekend_active" else 0
    active_minutes = 18 + workout_intensity * 72 + weekend_effect + rng.normal(0, 14)
    if profile.activity_archetype == "sedentary_weekday" and not is_weekend:
        active_minutes -= 16
    return float(active_minutes)


def _sample_steps(
    profile: UserProfile,
    active_minutes: float,
    is_weekend: bool,
    rng: np.random.Generator,
) -> float:
    weekend_multiplier = 1.22 if is_weekend and profile.activity_archetype == "weekend_active" else 1
    steps = profile.baseline_steps * weekend_multiplier + active_minutes * 42 + rng.normal(0, 950)
    if profile.activity_archetype == "sedentary_weekday" and not is_weekend:
        steps *= 0.82
    return float(steps)


def _sample_resting_hr(
    profile: UserProfile,
    sleep_hours: float,
    workout_intensity: float,
    active_minutes: float,
    rng: np.random.Generator,
) -> float:
    sleep_penalty = max(0, profile.baseline_sleep_hours - sleep_hours) * 1.7
    workout_penalty = workout_intensity * 2.8 + max(0, active_minutes - 70) * 0.035
    recovery_bonus = -1.2 if sleep_hours > profile.baseline_sleep_hours + 0.5 else 0
    return float(profile.baseline_resting_hr + sleep_penalty + workout_penalty + recovery_bonus + rng.normal(0, 1.6))
