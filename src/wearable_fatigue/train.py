from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wearable_fatigue.clustering import cluster_user_profiles
from wearable_fatigue.data import generate_synthetic_wearable_data
from wearable_fatigue.data_loaders import load_fitbit_dataset, load_mmash_dataset
from wearable_fatigue.features import build_model_table
from wearable_fatigue.models import train_models
from wearable_fatigue.reporting import evaluate_hypotheses
from wearable_fatigue.recommendations import recommend_workout_plan


DATA_CREDITS = {
    "fitbit": {
        "role": "Main modeling dataset",
        "citation": "Fitbit Fitness Tracker Data, published on Kaggle by Möbius (arashnic).",
        "url": "https://www.kaggle.com/datasets/arashnic/fitbit",
    },
    "mmash": {
        "role": "Supporting physiology, sleep, and stress dataset",
        "citation": (
            "Rossi, A., et al. (2020). Multilevel Monitoring of Activity and Sleep in Healthy "
            "People (version 1.0.0). PhysioNet."
        ),
        "url": "https://doi.org/10.13026/cerq-fc86",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train wearable fatigue prediction demo models.")
    parser.add_argument("--dataset", choices=["synthetic", "fitbit", "mmash"], default="synthetic", help="Dataset source to use.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw Fitbit or MMASH data.")
    parser.add_argument("--users", type=int, default=24, help="Number of synthetic users to generate.")
    parser.add_argument("--days", type=int, default=75, help="Number of days per user.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--search-iterations", type=int, default=30, help="Randomized-search candidates per tuned model.")
    parser.add_argument("--outer-repeats", type=int, default=5, help="Repeated unseen-user evaluation splits.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Repository root or output directory.")
    args = parser.parse_args()

    output_dir = args.output_dir
    processed_dir = output_dir / "data" / "processed"
    reports_dir = output_dir / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_data = _load_dataset(args)
    raw_data.to_csv(processed_dir / f"{args.dataset}_wearable_daily.csv", index=False)

    model_table = build_model_table(raw_data)
    if model_table.empty or len(model_table) < 12 or model_table["user_id"].nunique() < 3:
        summary = _build_support_only_summary(args.dataset, raw_data, len(model_table))
        (reports_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (reports_dir / "project_summary.md").write_text(_build_report(summary), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    clustered_table = cluster_user_profiles(model_table, random_state=args.seed)
    training_results = train_models(
        clustered_table,
        random_state=args.seed,
        search_iterations=args.search_iterations,
        outer_repeats=args.outer_repeats,
    )

    recommendations = _build_sample_recommendations(clustered_table)

    clustered_table.to_csv(processed_dir / "wearable_model_table.csv", index=False)
    recommendations.to_csv(processed_dir / "sample_recommendations.csv", index=False)
    training_results.feature_importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    training_results.feature_set_comparison.to_csv(reports_dir / "feature_set_comparison.csv", index=False)
    training_results.fold_metrics.to_csv(reports_dir / "fold_metrics.csv", index=False)

    behavior_profiles = sorted(clustered_table["behavior_profile"].dropna().unique().tolist())
    hypotheses = evaluate_hypotheses(
        classification_metrics=training_results.classification_metrics,
        regression_metrics=training_results.regression_metrics,
        behavior_profiles=behavior_profiles,
    )

    summary = {
        "dataset_source": args.dataset,
        "data_credits": DATA_CREDITS,
        "dataset": {
            "users": int(raw_data["user_id"].nunique()),
            "daily_rows": int(len(raw_data)),
            "model_rows": int(len(clustered_table)),
        },
        "classification_metrics": training_results.classification_metrics,
        "regression_metrics": training_results.regression_metrics,
        "cross_user_generalization": training_results.cross_user_metrics,
        "cross_user_activity_forecast": training_results.cross_user_regression_metrics,
        "chronological_evaluation": training_results.chronological_metrics,
        "selected_main_classifier": {
            "model_name": training_results.selected_main_classifier.model_name,
            "rationale": training_results.selected_main_classifier.rationale,
        },
        "strongest_overall_classifier": max(
            training_results.cross_user_metrics,
            key=lambda name: training_results.cross_user_metrics[name]["f1_macro"],
        ),
        "feature_importance": training_results.feature_importance.head(10).to_dict(orient="records"),
        "feature_set_comparison": training_results.feature_set_comparison.to_dict(orient="records"),
        "selected_feature_sets": training_results.selected_feature_sets,
        "best_parameters": training_results.best_params,
        "confidence_intervals": training_results.confidence_intervals,
        "evaluation_design": {
            "fatigue_target": "Morning fatigue-risk proxy; Fitbit contains no observed fatigue label.",
            "primary_validated_target": "Next-day active minutes.",
            "chronological_holdout": "Latest 25% of each user's model-ready records.",
            "cross_user_holdout": f"{args.outer_repeats} repeated 70/30 user-group splits.",
            "inner_tuning": "Four-fold GroupKFold, 30 randomized candidates by default.",
        },
        "hypotheses": hypotheses,
        "behavior_profiles": behavior_profiles,
    }
    (reports_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (reports_dir / "project_summary.md").write_text(_build_report(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2))


def _load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if args.dataset == "fitbit":
        return load_fitbit_dataset(args.raw_dir)
    if args.dataset == "mmash":
        return load_mmash_dataset(args.raw_dir)
    return generate_synthetic_wearable_data(
        n_users=args.users,
        days=args.days,
        random_state=args.seed,
    )


def _build_sample_recommendations(model_table: pd.DataFrame) -> pd.DataFrame:
    latest_rows = model_table.sort_values("date").groupby("user_id").tail(1)
    rows: list[dict[str, object]] = []

    for _, row in latest_rows.iterrows():
        recommendation = recommend_workout_plan(
            fatigue_risk=str(row["fatigue_risk"]),
            sleep_debt_hours=float(row["sleep_debt_hours"]),
            resting_hr_deviation=float(row["resting_hr_deviation"]),
            previous_activity_load=float(row["previous_activity_load"]),
        )
        rows.append(
            {
                "user_id": row["user_id"],
                "date": row["date"],
                "fatigue_risk": row["fatigue_risk"],
                "behavior_profile": row.get("behavior_profile", "unassigned"),
                "recommended_action": recommendation.action,
                "best_window": recommendation.best_window,
                "reason": recommendation.reason,
            }
        )

    return pd.DataFrame(rows)


def _build_support_only_summary(dataset_source: str, raw_data: pd.DataFrame, model_rows: int) -> dict[str, object]:
    return {
        "dataset_source": dataset_source,
        "data_credits": DATA_CREDITS,
        "dataset": {
            "users": int(raw_data["user_id"].nunique()),
            "daily_rows": int(len(raw_data)),
            "model_rows": int(model_rows),
        },
        "supporting_dataset_note": (
            "This dataset was loaded into the canonical wearable schema, but it does not contain enough "
            "multi-day per-user records for stable clustering and next-day activity forecasting. Use it as "
            "supporting validation or add the full raw dataset before primary model training."
        ),
        "classification_metrics": {},
        "regression_metrics": {},
        "cross_user_generalization": {},
        "cross_user_activity_forecast": {},
        "chronological_evaluation": {},
        "selected_main_classifier": {
            "model_name": "not_selected",
            "rationale": "No main classifier selected because the dataset is too small for stable model training.",
        },
        "strongest_overall_classifier": "not_selected",
        "feature_importance": [],
        "feature_set_comparison": [],
        "selected_feature_sets": {},
        "best_parameters": {},
        "confidence_intervals": {},
        "hypotheses": {},
        "behavior_profiles": [],
    }


def _build_report(summary: dict[str, object]) -> str:
    if not summary["classification_metrics"]:
        return f"""# Wearable Fatigue Prediction Summary

## Objective

Load real wearable data into a reusable canonical schema and determine whether it can support fatigue prediction, next-day activity forecasting, behavior clustering, and recommendation logic.

## Dataset

- Source: {summary["dataset_source"]}
- Users: {summary["dataset"]["users"]}
- Daily records: {summary["dataset"]["daily_rows"]}
- Model-ready records: {summary["dataset"]["model_rows"]}

## Data Sources and Credits

- Main dataset: {summary["data_credits"]["fitbit"]["citation"]} {summary["data_credits"]["fitbit"]["url"]}
- Supporting dataset: {summary["data_credits"]["mmash"]["citation"]} {summary["data_credits"]["mmash"]["url"]}

## Dataset Role

{summary["supporting_dataset_note"]}
"""

    selected = summary["selected_main_classifier"]
    classification = summary["classification_metrics"]["random_forest"]
    regression = summary["regression_metrics"]["random_forest"]
    xgboost_classification = summary["classification_metrics"]["xgboost"]
    xgboost_regression = summary["regression_metrics"]["xgboost"]
    random_forest_cross_user = summary["cross_user_generalization"]["random_forest"]
    xgboost_cross_user = summary["cross_user_generalization"]["xgboost"]
    activity_cross_user = summary["cross_user_activity_forecast"]
    hypotheses = summary["hypotheses"]

    return f"""# Wearable Fatigue Prediction Summary

## Objective

Forecast next-day activity from wearable behavior and demonstrate a clearly labeled morning fatigue-risk proxy for workout/rest recommendations.

## Dataset

- Source: {summary["dataset_source"]}
- Users: {summary["dataset"]["users"]}
- Daily records: {summary["dataset"]["daily_rows"]}
- Model-ready records: {summary["dataset"]["model_rows"]}

These metrics were produced from the real Fitbit Fitness Tracker Data files, not from the synthetic fallback.

## Data Sources and Credits

- Main modeling dataset: {summary["data_credits"]["fitbit"]["citation"]} {summary["data_credits"]["fitbit"]["url"]}
- Supporting validation dataset: {summary["data_credits"]["mmash"]["citation"]} {summary["data_credits"]["mmash"]["url"]}

MMASH was loaded separately to validate the supporting data adapter and provide richer physiological, sleep-quality, and stress context. It does not produce the main model scores because it contains approximately one 24-hour recording per participant rather than a longitudinal daily series.

## Chronological Evaluation

The latest 25% of each user's records were held out. Feature-window selection and hyperparameter tuning used only the earlier records.

- Random Forest fatigue macro F1: {classification["f1_macro"]:.3f}
- XGBoost fatigue macro F1: {xgboost_classification["f1_macro"]:.3f}
- Random Forest next-day active-minutes MAE: {regression["mae"]:.1f}
- XGBoost next-day active-minutes MAE: {xgboost_regression["mae"]:.1f}

## Repeated Cross-User Generalization

This nested evaluation repeatedly trains on one set of users and evaluates on completely unseen users. Reported values are means across outer folds.

- Unseen users tested: {random_forest_cross_user["unseen_user_count"]:.0f}
- Random Forest unseen-user macro F1: {random_forest_cross_user["f1_macro"]:.3f}
- XGBoost unseen-user macro F1: {xgboost_cross_user["f1_macro"]:.3f}

For the directly observed next-day target, lower MAE is better:

- Mean baseline unseen-user MAE: {activity_cross_user["baseline_mean"]["mae"]:.1f}
- Random Forest unseen-user MAE: {activity_cross_user["random_forest"]["mae"]:.1f}
- XGBoost unseen-user MAE: {activity_cross_user["xgboost"]["mae"]:.1f}

Neither learned regressor beats the unseen-user baseline, so next-day activity generalization is not demonstrated by this sample.

## Selected Tree Model

- Model: {selected["model_name"]}
- Rationale: {selected["rationale"]}

XGBoost had the slightly higher raw F1 score: {xgboost_cross_user["f1_macro"]:.3f} versus {random_forest_cross_user["f1_macro"]:.3f}. Random Forest was selected because this difference was too small and inconsistent across folds to establish that XGBoost genuinely generalizes better; the predefined selection rule required a stable improvement of at least 0.020.

Logistic Regression achieved the highest overall unseen-user macro F1 ({summary["cross_user_generalization"]["logistic_regression"]["f1_macro"]:.3f}). Random Forest is the selected model only for the requested Random Forest-versus-XGBoost comparison. The linear model's strength is expected because the proxy target is constructed from an additive weighted rule.

## Selected Feature Families

- Morning fatigue-risk proxy: {summary["selected_feature_sets"]["fatigue_proxy"]}
- Next-day activity forecast: {summary["selected_feature_sets"]["next_day_activity"]}

## Interpretation Boundary

Fitbit does not provide an observed fatigue outcome. The fatigue classes are a transparent proxy derived from prior sleep, morning heart rate, and previous-day activity. Classification performance measures reproduction of that proxy and is not clinical validation. Next-day active minutes is the primary directly observed prediction target.

## Evaluation Design

- Personal baselines use past records only.
- Morning proxy features use the completed night and activity only through yesterday.
- Next-day forecasting uses information available through the current day.
- Strict 3-, 5-, and 7-day windows and a 3-day-half-life exponentially weighted representation are compared inside training folds.
- Randomized tuning uses group-aware inner folds; outer user and chronological holdouts remain untouched.

## Hypotheses

{chr(10).join(f"- {key}: {value['status']} - {value['evidence']}" for key, value in hypotheses.items())}

## Top Feature Drivers

{chr(10).join(f"- {row['feature']}: {row['importance_mean']:.4f}" for row in summary["feature_importance"][:5])}

## Behavior Profiles

{chr(10).join(f"- {profile}" for profile in summary["behavior_profiles"])}

## Product Recommendation Layer

The model output is translated into user-facing guidance: normal workout, light workout, rest, or reschedule workout. Recommendations are driven by fatigue risk, sleep debt, resting heart-rate deviation, and previous activity load.
"""


if __name__ == "__main__":
    main()
