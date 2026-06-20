import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearable_fatigue.data_loaders import CANONICAL_COLUMNS, load_fitbit_dataset, load_mmash_dataset
from wearable_fatigue.model_selection import select_main_classifier
from wearable_fatigue.reporting import evaluate_hypotheses


def test_fitbit_loader_maps_fixture_to_canonical_schema():
    data = load_fitbit_dataset(ROOT / "tests" / "fixtures" / "fitbit")

    assert CANONICAL_COLUMNS.issubset(data.columns)
    assert len(data) == 6
    assert data["user_id"].nunique() == 2
    assert data["sleep_hours"].notna().all()
    assert data["resting_heart_rate"].notna().all()


def test_fitbit_loader_handles_missing_optional_heart_rate():
    data = load_fitbit_dataset(ROOT / "tests" / "fixtures" / "fitbit", require_heart_rate=False)

    assert "resting_heart_rate" in data.columns
    assert data["resting_heart_rate"].notna().all()


def test_mmash_loader_maps_fixture_to_canonical_schema():
    data = load_mmash_dataset(ROOT / "tests" / "fixtures" / "mmash")

    assert CANONICAL_COLUMNS.issubset(data.columns)
    assert {"daily_stress", "psqi", "sleep_fragmentation_index"}.issubset(data.columns)
    assert len(data) == 2
    assert data["user_id"].nunique() == 2


def test_mmash_loader_skips_participant_with_empty_sleep_file(tmp_path):
    fixture_root = ROOT / "tests" / "fixtures" / "mmash"
    copied_root = tmp_path / "mmash"
    shutil.copytree(fixture_root, copied_root)
    incomplete_user = copied_root / "User_03"
    incomplete_user.mkdir()
    (incomplete_user / "sleep.csv").write_text("Total Sleep Time (TST)\n", encoding="utf-8")
    shutil.copy(fixture_root / "User_01" / "Actigraph.csv", incomplete_user / "Actigraph.csv")

    data = load_mmash_dataset(copied_root)

    assert data["user_id"].nunique() == 2


def test_model_selection_prefers_xgboost_only_after_threshold():
    selected = select_main_classifier(
        {
            "random_forest": {"f1_macro": 0.78},
            "xgboost": {"f1_macro": 0.81},
        }
    )

    assert selected.model_name == "xgboost"

    selected = select_main_classifier(
        {
            "random_forest": {"f1_macro": 0.78},
            "xgboost": {"f1_macro": 0.79},
        }
    )

    assert selected.model_name == "random_forest"


def test_hypothesis_evaluation_reports_supported_statuses():
    hypotheses = evaluate_hypotheses(
        classification_metrics={
            "baseline_most_frequent": {"f1_macro": 0.2},
            "random_forest": {"f1_macro": 0.7},
            "xgboost": {"f1_macro": 0.75},
        },
        regression_metrics={
            "baseline_mean": {"mae": 30.0},
            "random_forest": {"mae": 20.0},
            "xgboost": {"mae": 18.0},
        },
        behavior_profiles=["consistent active user", "sedentary weekday user"],
    )

    assert hypotheses["H1"]["status"] == "supported"
    assert hypotheses["H2"]["status"] == "supported"
    assert hypotheses["H3"]["status"] == "supported"


def test_training_cli_writes_model_selection_and_feature_importance(tmp_path):
    command = [
        sys.executable,
        "-m",
        "wearable_fatigue.train",
        "--dataset",
        "synthetic",
        "--users",
        "8",
        "--days",
        "20",
        "--seed",
        "21",
        "--output-dir",
        str(tmp_path),
        "--search-iterations",
        "1",
        "--outer-repeats",
        "1",
    ]
    subprocess.run(command, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")}, check=True)

    summary = json.loads((tmp_path / "reports" / "model_summary.json").read_text(encoding="utf-8"))

    assert "selected_main_classifier" in summary
    assert summary["data_credits"]["fitbit"]["role"] == "Main modeling dataset"
    assert "feature_importance" in summary
    assert len(summary["feature_importance"]) > 0
    assert "hypotheses" in summary
    assert "xgboost" in summary["classification_metrics"]
    assert "xgboost" in summary["regression_metrics"]
    assert "chronological_evaluation" in summary
    assert summary["selected_feature_sets"]
    assert summary["best_parameters"]
    assert summary["confidence_intervals"]
    assert (tmp_path / "reports" / "feature_set_comparison.csv").exists()
    assert (tmp_path / "reports" / "fold_metrics.csv").exists()
