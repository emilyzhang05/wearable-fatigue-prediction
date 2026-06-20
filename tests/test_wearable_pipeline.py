import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearable_fatigue.clustering import cluster_user_profiles
from wearable_fatigue.data import generate_synthetic_wearable_data
from wearable_fatigue.features import FEATURE_COLUMNS, build_model_table
from wearable_fatigue.models import chronological_user_split, evaluate_cross_user_generalization, train_models
from wearable_fatigue.recommendations import recommend_workout_plan


def test_generate_synthetic_wearable_data_has_expected_shape_and_columns():
    data = generate_synthetic_wearable_data(n_users=4, days=14, random_state=7)

    assert len(data) == 56
    assert {
        "user_id",
        "date",
        "sleep_hours",
        "resting_heart_rate",
        "steps",
        "active_minutes",
        "workout_intensity",
    }.issubset(data.columns)
    assert data["user_id"].nunique() == 4


def test_build_model_table_creates_time_series_features_and_labels():
    raw_data = generate_synthetic_wearable_data(n_users=5, days=21, random_state=11)
    model_table = build_model_table(raw_data)

    assert set(["low", "medium", "high"]).issubset(set(model_table["fatigue_risk"]))
    assert model_table[FEATURE_COLUMNS].isna().sum().sum() == 0
    assert "next_day_active_minutes" in model_table.columns
    assert model_table["sleep_7d_avg"].notna().all()
    assert model_table["prior_activity_load_7d_avg"].notna().all()


def test_personal_baselines_use_only_prior_records():
    raw_data = generate_synthetic_wearable_data(n_users=3, days=14, random_state=5)
    model_table = build_model_table(raw_data)
    sample = model_table.iloc[0]
    prior = raw_data[
        (raw_data["user_id"] == sample["user_id"])
        & (pd.to_datetime(raw_data["date"]) < sample["date"])
    ]

    assert sample["sleep_baseline"] == prior["sleep_hours"].median()
    assert sample["resting_hr_baseline"] == prior["resting_heart_rate"].median()


def test_chronological_split_reserves_each_users_latest_records():
    table = build_model_table(generate_synthetic_wearable_data(n_users=5, days=21, random_state=9))
    train_index, test_index = chronological_user_split(table)

    for user_id in table["user_id"].unique():
        user_train = table.loc[train_index][table.loc[train_index, "user_id"] == user_id]
        user_test = table.loc[test_index][table.loc[test_index, "user_id"] == user_id]
        assert user_train["date"].max() < user_test["date"].min()


def test_training_pipeline_returns_model_metrics_and_artifacts():
    raw_data = generate_synthetic_wearable_data(n_users=8, days=20, random_state=13)
    model_table = build_model_table(raw_data)

    results = train_models(model_table, random_state=13, search_iterations=1, outer_repeats=1)

    assert results.classification_metrics["random_forest"]["f1_macro"] > 0
    assert "xgboost" in results.classification_metrics
    assert "xgboost" in results.regression_metrics
    assert results.regression_metrics["random_forest"]["mae"] >= 0
    assert "fatigue_classifier" in results.models
    assert "activity_regressor" in results.models
    assert results.selected_main_classifier.model_name in {"random_forest", "xgboost"}
    assert not results.feature_importance.empty
    assert not results.feature_set_comparison.empty
    assert results.best_params
    assert results.confidence_intervals

    cross_user_metrics = results.cross_user_metrics
    assert cross_user_metrics["random_forest"]["unseen_user_count"] >= 1
    assert cross_user_metrics["xgboost"]["f1_macro"] >= 0


def test_clustering_and_recommendations_are_user_readable():
    raw_data = generate_synthetic_wearable_data(n_users=8, days=24, random_state=17)
    model_table = build_model_table(raw_data)
    clustered = cluster_user_profiles(model_table, n_clusters=3, random_state=17)

    assert "behavior_cluster" in clustered.columns
    assert clustered["behavior_profile"].nunique() >= 2

    sample = clustered.iloc[0]
    recommendation = recommend_workout_plan(
        fatigue_risk=str(sample["fatigue_risk"]),
        sleep_debt_hours=float(sample["sleep_debt_hours"]),
        resting_hr_deviation=float(sample["resting_hr_deviation"]),
        previous_activity_load=float(sample["previous_activity_load"]),
    )

    assert recommendation.action in {
        "normal workout",
        "light workout",
        "rest",
        "reschedule workout",
    }
    assert recommendation.reason
