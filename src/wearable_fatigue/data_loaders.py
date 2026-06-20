from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = {
    "user_id",
    "date",
    "sleep_hours",
    "resting_heart_rate",
    "average_heart_rate",
    "steps",
    "active_minutes",
    "workout_intensity",
    "weekday",
    "is_weekend",
}


def load_fitbit_dataset(raw_dir: Path | str, require_heart_rate: bool = False) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    activity_path = _require_file(raw_path / "dailyActivity_merged.csv")
    sleep_path = _require_file(raw_path / "sleepDay_merged.csv")
    heart_rate_path = raw_path / "heartrate_seconds_merged.csv"

    activity = pd.read_csv(activity_path)
    sleep = pd.read_csv(sleep_path)
    data = _prepare_fitbit_activity(activity).merge(
        _prepare_fitbit_sleep(sleep),
        on=["user_id", "date"],
        how="inner",
    )

    if heart_rate_path.exists():
        data = data.merge(_prepare_fitbit_heart_rate(pd.read_csv(heart_rate_path)), on=["user_id", "date"], how="left")
    elif require_heart_rate:
        raise FileNotFoundError(f"Missing optional heart-rate file: {heart_rate_path}")
    else:
        data["average_heart_rate"] = pd.NA
        data["resting_heart_rate"] = pd.NA

    data["average_heart_rate"] = data.groupby("user_id")["average_heart_rate"].transform(_fill_with_group_median)
    data["resting_heart_rate"] = data.groupby("user_id")["resting_heart_rate"].transform(_fill_with_group_median)
    data["average_heart_rate"] = data["average_heart_rate"].fillna(70.0)
    data["resting_heart_rate"] = data["resting_heart_rate"].fillna(62.0)

    return _finalize_canonical(data, source="fitbit")


def load_mmash_dataset(raw_dir: Path | str) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    participant_dirs = sorted(path for path in raw_path.iterdir() if path.is_dir())
    if not participant_dirs:
        raise FileNotFoundError(f"No MMASH participant folders found in {raw_path}")

    rows: list[dict[str, object]] = []
    for participant_dir in participant_dirs:
        sleep_path = _require_file(participant_dir / "sleep.csv")
        actigraph_path = _require_file(participant_dir / "Actigraph.csv")
        questionnaire_path = participant_dir / "questionnaire.csv"

        sleep = pd.read_csv(sleep_path)
        actigraph = pd.read_csv(actigraph_path)
        questionnaire = pd.read_csv(questionnaire_path) if questionnaire_path.exists() else pd.DataFrame()
        if sleep.empty or actigraph.empty:
            continue
        rows.append(_build_mmash_row(participant_dir.name, sleep, actigraph, questionnaire))

    if not rows:
        raise ValueError(f"No complete MMASH participant records found in {raw_path}")

    return _finalize_canonical(pd.DataFrame(rows), source="mmash")


def _prepare_fitbit_activity(activity: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Id",
        "ActivityDate",
        "TotalSteps",
        "VeryActiveMinutes",
        "FairlyActiveMinutes",
        "LightlyActiveMinutes",
        "SedentaryMinutes",
    }
    _require_columns(activity, required, "dailyActivity_merged.csv")
    prepared = activity.copy()
    prepared["user_id"] = prepared["Id"].astype(str)
    prepared["date"] = pd.to_datetime(prepared["ActivityDate"]).dt.normalize()
    prepared["steps"] = prepared["TotalSteps"].astype(float)
    prepared["active_minutes"] = (
        prepared["VeryActiveMinutes"].fillna(0)
        + prepared["FairlyActiveMinutes"].fillna(0)
        + prepared["LightlyActiveMinutes"].fillna(0)
    ).astype(float)
    prepared["workout_intensity"] = (
        (prepared["VeryActiveMinutes"].fillna(0) * 1.0 + prepared["FairlyActiveMinutes"].fillna(0) * 0.6)
        / prepared["active_minutes"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    return prepared[["user_id", "date", "steps", "active_minutes", "workout_intensity"]]


def _prepare_fitbit_sleep(sleep: pd.DataFrame) -> pd.DataFrame:
    _require_columns(sleep, {"Id", "SleepDay", "TotalMinutesAsleep"}, "sleepDay_merged.csv")
    prepared = sleep.copy()
    prepared["user_id"] = prepared["Id"].astype(str)
    prepared["date"] = pd.to_datetime(prepared["SleepDay"], format="mixed").dt.normalize()
    prepared["sleep_hours"] = prepared["TotalMinutesAsleep"].astype(float) / 60.0
    return prepared.groupby(["user_id", "date"], as_index=False)["sleep_hours"].mean()


def _prepare_fitbit_heart_rate(heart_rate: pd.DataFrame) -> pd.DataFrame:
    _require_columns(heart_rate, {"Id", "Time", "Value"}, "heartrate_seconds_merged.csv")
    prepared = heart_rate.copy()
    prepared["user_id"] = prepared["Id"].astype(str)
    prepared["timestamp"] = pd.to_datetime(prepared["Time"], format="mixed")
    prepared["date"] = prepared["timestamp"].dt.normalize()
    prepared["hour"] = prepared["timestamp"].dt.hour
    daily_average = prepared.groupby(["user_id", "date"], as_index=False)["Value"].mean().rename(columns={"Value": "average_heart_rate"})
    resting = (
        prepared[prepared["hour"].between(0, 5)]
        .groupby(["user_id", "date"], as_index=False)["Value"]
        .quantile(0.1)
        .rename(columns={"Value": "resting_heart_rate"})
    )
    return daily_average.merge(resting, on=["user_id", "date"], how="left")


def _build_mmash_row(
    user_id: str,
    sleep: pd.DataFrame,
    actigraph: pd.DataFrame,
    questionnaire: pd.DataFrame,
) -> dict[str, object]:
    _require_columns(sleep, {"Total Sleep Time (TST)"}, "sleep.csv")
    _require_columns(actigraph, {"Steps", "HR", "Vector Magnitude"}, "Actigraph.csv")

    active_threshold = actigraph["Vector Magnitude"].median()
    active_seconds = int((actigraph["Vector Magnitude"] > active_threshold).sum())
    high_intensity_seconds = int((actigraph["HR"] >= actigraph["HR"].quantile(0.75)).sum())
    active_minutes = max(active_seconds / 60.0, float(actigraph["Steps"].sum()) / 100.0)

    row = {
        "user_id": user_id,
        "date": pd.Timestamp("2020-01-01"),
        "sleep_hours": float(sleep["Total Sleep Time (TST)"].iloc[0]) / 60.0,
        "resting_heart_rate": float(actigraph["HR"].quantile(0.1)),
        "average_heart_rate": float(actigraph["HR"].mean()),
        "steps": float(actigraph["Steps"].sum()),
        "active_minutes": active_minutes,
        "workout_intensity": min(1.0, high_intensity_seconds / max(active_seconds, 1)),
        "sleep_fragmentation_index": _optional_first(sleep, "Sleep Fragmentation Index"),
    }

    if not questionnaire.empty:
        row["daily_stress"] = _optional_first(questionnaire, "Daily_stress")
        row["psqi"] = _optional_first(questionnaire, "PSQI")

    return row


def _finalize_canonical(data: pd.DataFrame, source: str) -> pd.DataFrame:
    finalized = data.copy()
    finalized["date"] = pd.to_datetime(finalized["date"]).dt.normalize()
    finalized["weekday"] = finalized["date"].dt.weekday
    finalized["is_weekend"] = (finalized["weekday"] >= 5).astype(int)
    finalized["source_dataset"] = source

    for column in CANONICAL_COLUMNS:
        if column not in finalized.columns:
            finalized[column] = pd.NA

    return finalized.sort_values(["user_id", "date"]).reset_index(drop=True)


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    return path


def _require_columns(data: pd.DataFrame, required: set[str], file_name: str) -> None:
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {sorted(missing)}")


def _optional_first(data: pd.DataFrame, column: str) -> float | None:
    if column not in data.columns:
        return None
    value = data[column].dropna()
    if value.empty:
        return None
    return float(value.iloc[0])


def _fill_with_group_median(values: pd.Series) -> pd.Series:
    if not values.notna().any():
        return values
    return values.fillna(values.median())
