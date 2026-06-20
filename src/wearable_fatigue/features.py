from __future__ import annotations

import pandas as pd


WINDOWS = (3, 5, 7)
MEDIUM_FATIGUE_SCORE = 1.75
HIGH_FATIGUE_SCORE = 2.75

MORNING_RAW_FEATURES = [
    "sleep_hours",
    "resting_heart_rate",
    "weekday",
    "is_weekend",
]
MORNING_BASELINE_FEATURES = MORNING_RAW_FEATURES + [
    "sleep_debt_hours",
    "resting_hr_deviation",
]
MORNING_LAG_FEATURES = MORNING_BASELINE_FEATURES + [
    "sleep_lag_1d",
    "previous_activity_load",
    "active_minutes_lag_1d",
    "resting_hr_deviation_lag_1d",
]
MORNING_ROLLING_FEATURES = MORNING_LAG_FEATURES + [
    f"{column}_{window}d_avg"
    for column in ("sleep", "sleep_debt", "prior_activity_load", "prior_active_minutes", "resting_hr_deviation")
    for window in WINDOWS
]
MORNING_EWM_FEATURES = MORNING_LAG_FEATURES + [
    f"{column}_ewm_3d"
    for column in ("sleep", "sleep_debt", "prior_activity_load", "prior_active_minutes", "resting_hr_deviation")
]

ACTIVITY_RAW_FEATURES = [
    "sleep_hours",
    "resting_heart_rate",
    "steps",
    "active_minutes",
    "workout_intensity",
    "weekday",
    "is_weekend",
]
ACTIVITY_BASELINE_FEATURES = ACTIVITY_RAW_FEATURES + [
    "sleep_debt_hours",
    "resting_hr_deviation",
    "activity_deviation_minutes",
]
ACTIVITY_LAG_FEATURES = ACTIVITY_BASELINE_FEATURES + [
    "sleep_lag_1d",
    "previous_activity_load",
    "active_minutes_lag_1d",
]
ACTIVITY_ROLLING_FEATURES = ACTIVITY_LAG_FEATURES + [
    f"{column}_{window}d_avg"
    for column in ("sleep", "sleep_debt", "activity_load", "active_minutes", "resting_hr_deviation")
    for window in WINDOWS
]
ACTIVITY_EWM_FEATURES = ACTIVITY_LAG_FEATURES + [
    f"{column}_ewm_3d"
    for column in ("sleep", "sleep_debt", "activity_load", "active_minutes", "resting_hr_deviation")
]

CLASSIFICATION_FEATURE_SETS = {
    "raw_daily": MORNING_RAW_FEATURES,
    "personal_baselines": MORNING_BASELINE_FEATURES,
    "lagged": MORNING_LAG_FEATURES,
    "rolling_windows": MORNING_ROLLING_FEATURES,
    "exponentially_weighted": MORNING_EWM_FEATURES,
}
REGRESSION_FEATURE_SETS = {
    "raw_daily": ACTIVITY_RAW_FEATURES,
    "personal_baselines": ACTIVITY_BASELINE_FEATURES,
    "lagged": ACTIVITY_LAG_FEATURES,
    "rolling_windows": ACTIVITY_ROLLING_FEATURES,
    "exponentially_weighted": ACTIVITY_EWM_FEATURES,
}

FEATURE_COLUMNS = MORNING_EWM_FEATURES


def build_model_table(raw_data: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "user_id",
        "date",
        "sleep_hours",
        "resting_heart_rate",
        "steps",
        "active_minutes",
        "workout_intensity",
        "weekday",
        "is_weekend",
    }
    missing = required_columns.difference(raw_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = raw_data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["user_id", "date"]).reset_index(drop=True)

    data["sleep_baseline"] = _past_only_expanding_median(data, "sleep_hours")
    data["resting_hr_baseline"] = _past_only_expanding_median(data, "resting_heart_rate")
    data["active_minutes_baseline"] = _past_only_expanding_median(data, "active_minutes")

    data["activity_load"] = (
        data["steps"] / 1000
        + data["active_minutes"] * 0.12
        + data["workout_intensity"] * 12
    )
    data["sleep_debt_hours"] = (data["sleep_baseline"] - data["sleep_hours"]).clip(lower=0)
    data["resting_hr_deviation"] = data["resting_heart_rate"] - data["resting_hr_baseline"]
    data["activity_deviation_minutes"] = data["active_minutes"] - data["active_minutes_baseline"]

    data["sleep_lag_1d"] = _group_shift(data, "sleep_hours")
    data["previous_activity_load"] = _group_shift(data, "activity_load")
    data["active_minutes_lag_1d"] = _group_shift(data, "active_minutes")
    data["resting_hr_deviation_lag_1d"] = _group_shift(data, "resting_hr_deviation")
    data["next_day_active_minutes"] = data.groupby("user_id")["active_minutes"].shift(-1)
    data["next_day_steps"] = data.groupby("user_id")["steps"].shift(-1)

    morning_sources = {
        "sleep": data["sleep_hours"],
        "sleep_debt": data["sleep_debt_hours"],
        "prior_activity_load": _group_shift(data, "activity_load"),
        "prior_active_minutes": _group_shift(data, "active_minutes"),
        "resting_hr_deviation": data["resting_hr_deviation"],
    }
    activity_sources = {
        "sleep": data["sleep_hours"],
        "sleep_debt": data["sleep_debt_hours"],
        "activity_load": data["activity_load"],
        "active_minutes": data["active_minutes"],
        "resting_hr_deviation": data["resting_hr_deviation"],
    }
    for name, values in {**morning_sources, **activity_sources}.items():
        for window in WINDOWS:
            data[f"{name}_{window}d_avg"] = _rolling_user_mean(data, values, window)
        data[f"{name}_ewm_3d"] = _ewm_user_mean(data, values, halflife=3, min_periods=3)

    data["fatigue_score"] = (
        data["sleep_debt_hours"] * 1.15
        + data["resting_hr_deviation"].clip(lower=0) * 0.22
        + data["previous_activity_load"].clip(lower=0) * 0.035
    )
    data["fatigue_risk"] = _label_fatigue_risk(data["fatigue_score"])

    required_history = sorted(set(MORNING_ROLLING_FEATURES + ACTIVITY_ROLLING_FEATURES))
    model_table = data.dropna(subset=required_history + ["next_day_active_minutes", "fatigue_risk"]).copy()
    return model_table.reset_index(drop=True)


def _past_only_expanding_median(data: pd.DataFrame, column: str) -> pd.Series:
    return data.groupby("user_id")[column].transform(
        lambda values: values.expanding(min_periods=1).median().shift(1)
    )


def _group_shift(data: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    return data.groupby("user_id")[column].shift(periods)


def _rolling_user_mean(data: pd.DataFrame, values: pd.Series, window: int) -> pd.Series:
    return (
        values.groupby(data["user_id"])
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _ewm_user_mean(
    data: pd.DataFrame,
    values: pd.Series,
    halflife: float,
    min_periods: int,
) -> pd.Series:
    return values.groupby(data["user_id"], group_keys=False).apply(
        lambda group: group.ewm(halflife=halflife, adjust=False, min_periods=min_periods).mean()
    )


def _label_fatigue_risk(score: pd.Series) -> pd.Series:
    labels = pd.Series("low", index=score.index, dtype="object")
    labels.loc[score >= MEDIUM_FATIGUE_SCORE] = "medium"
    labels.loc[score >= HIGH_FATIGUE_SCORE] = "high"
    labels.loc[score.isna()] = pd.NA
    return labels
