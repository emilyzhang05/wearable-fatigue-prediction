from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from wearable_fatigue.features import CLASSIFICATION_FEATURE_SETS, REGRESSION_FEATURE_SETS
from wearable_fatigue.model_selection import SelectedModel, select_main_classifier


RF_CLASSIFIER_SPACE = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [2, 4, 8, 12],
    "max_features": [0.5, 0.75, 1.0],
}
RF_REGRESSOR_SPACE = dict(RF_CLASSIFIER_SPACE)
XGB_SPACE = {
    "n_estimators": [100, 200, 350, 450, 600],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.65, 0.8, 1.0],
    "colsample_bytree": [0.65, 0.8, 1.0],
    "reg_lambda": [1, 3, 10],
    "reg_alpha": [0, 0.1, 1],
    "gamma": [0, 0.1, 0.5],
}


@dataclass(frozen=True)
class TrainingResults:
    classification_metrics: dict[str, dict[str, float]]
    regression_metrics: dict[str, dict[str, float]]
    cross_user_metrics: dict[str, dict[str, float]]
    cross_user_regression_metrics: dict[str, dict[str, float]]
    chronological_metrics: dict[str, dict[str, dict[str, float]]]
    selected_main_classifier: SelectedModel
    feature_importance: pd.DataFrame
    feature_set_comparison: pd.DataFrame
    selected_feature_sets: dict[str, str]
    best_params: dict[str, dict[str, Any]]
    fold_metrics: pd.DataFrame
    confidence_intervals: dict[str, dict[str, float]]
    models: dict[str, object]


def train_models(
    model_table: pd.DataFrame,
    random_state: int = 42,
    search_iterations: int = 30,
    outer_repeats: int = 5,
) -> TrainingResults:
    label_encoder = LabelEncoder()
    encoded_labels = pd.Series(
        label_encoder.fit_transform(model_table["fatigue_risk"]),
        index=model_table.index,
    )

    cross_user_metrics, fold_metrics = evaluate_cross_user_generalization(
        model_table,
        random_state=random_state,
        search_iterations=search_iterations,
        outer_repeats=outer_repeats,
        encoded_labels=encoded_labels,
        return_fold_metrics=True,
    )
    selected_main_classifier = select_main_classifier(
        cross_user_metrics,
        fold_metrics=fold_metrics[fold_metrics["task"] == "fatigue_proxy"],
    )

    train_index, test_index = chronological_user_split(model_table, test_size=0.25)
    train_table = model_table.loc[train_index]
    test_table = model_table.loc[test_index]
    train_labels = encoded_labels.loc[train_index]
    test_labels = encoded_labels.loc[test_index]

    class_set, class_comparison = select_feature_set(
        train_table,
        train_labels,
        CLASSIFICATION_FEATURE_SETS,
        task="classification",
        random_state=random_state,
    )
    reg_set, reg_comparison = select_feature_set(
        train_table,
        train_table["next_day_active_minutes"],
        REGRESSION_FEATURE_SETS,
        task="regression",
        random_state=random_state,
    )

    classifier_models, classifier_params = _fit_task_models(
        train_table,
        train_labels,
        CLASSIFICATION_FEATURE_SETS[class_set],
        task="classification",
        random_state=random_state,
        search_iterations=search_iterations,
    )
    regressor_models, regressor_params = _fit_task_models(
        train_table,
        train_table["next_day_active_minutes"],
        REGRESSION_FEATURE_SETS[reg_set],
        task="regression",
        random_state=random_state,
        search_iterations=search_iterations,
    )

    classification_metrics = {
        name: _classification_metrics(test_labels, model.predict(test_table[CLASSIFICATION_FEATURE_SETS[class_set]]))
        for name, model in classifier_models.items()
    }
    regression_metrics = {
        name: _regression_metrics(
            test_table["next_day_active_minutes"],
            model.predict(test_table[REGRESSION_FEATURE_SETS[reg_set]]),
        )
        for name, model in regressor_models.items()
    }

    selected_model = classifier_models[selected_main_classifier.model_name]
    feature_importance = calculate_feature_importance(
        selected_model,
        test_table[CLASSIFICATION_FEATURE_SETS[class_set]],
        test_labels,
        random_state=random_state,
    )
    comparison = pd.concat([class_comparison, reg_comparison], ignore_index=True)
    confidence_intervals = _confidence_intervals(fold_metrics)
    cross_user_regression_metrics = _aggregate_cross_user_regression(fold_metrics)

    return TrainingResults(
        classification_metrics=classification_metrics,
        regression_metrics=regression_metrics,
        cross_user_metrics=cross_user_metrics,
        cross_user_regression_metrics=cross_user_regression_metrics,
        chronological_metrics={
            "fatigue_proxy_classification": classification_metrics,
            "next_day_activity_forecast": regression_metrics,
        },
        selected_main_classifier=selected_main_classifier,
        feature_importance=feature_importance,
        feature_set_comparison=comparison,
        selected_feature_sets={"fatigue_proxy": class_set, "next_day_activity": reg_set},
        best_params={**{f"classifier_{k}": v for k, v in classifier_params.items()}, **{f"regressor_{k}": v for k, v in regressor_params.items()}},
        fold_metrics=fold_metrics,
        confidence_intervals=confidence_intervals,
        models={
            "fatigue_proxy_classifier": selected_model,
            "fatigue_classifier": selected_model,
            "activity_regressor": _select_activity_model(regressor_models, regression_metrics),
        },
    )


def evaluate_cross_user_generalization(
    model_table: pd.DataFrame,
    random_state: int = 42,
    search_iterations: int = 30,
    outer_repeats: int = 5,
    encoded_labels: pd.Series | None = None,
    return_fold_metrics: bool = False,
) -> dict[str, dict[str, float]] | tuple[dict[str, dict[str, float]], pd.DataFrame]:
    if encoded_labels is None:
        encoder = LabelEncoder()
        encoded_labels = pd.Series(encoder.fit_transform(model_table["fatigue_risk"]), index=model_table.index)
    splitter = GroupShuffleSplit(n_splits=outer_repeats, test_size=0.3, random_state=random_state)
    rows: list[dict[str, object]] = []

    for fold, (train_pos, test_pos) in enumerate(
        splitter.split(model_table, encoded_labels, groups=model_table["user_id"]),
        start=1,
    ):
        train_table = model_table.iloc[train_pos]
        test_table = model_table.iloc[test_pos]
        y_train_class = encoded_labels.iloc[train_pos]
        y_test_class = encoded_labels.iloc[test_pos]

        class_set, _ = select_feature_set(
            train_table, y_train_class, CLASSIFICATION_FEATURE_SETS, "classification", random_state + fold
        )
        reg_set, _ = select_feature_set(
            train_table,
            train_table["next_day_active_minutes"],
            REGRESSION_FEATURE_SETS,
            "regression",
            random_state + fold,
        )
        classifiers, _ = _fit_task_models(
            train_table,
            y_train_class,
            CLASSIFICATION_FEATURE_SETS[class_set],
            "classification",
            random_state + fold,
            search_iterations,
        )
        regressors, _ = _fit_task_models(
            train_table,
            train_table["next_day_active_minutes"],
            REGRESSION_FEATURE_SETS[reg_set],
            "regression",
            random_state + fold,
            search_iterations,
        )
        for name, model in classifiers.items():
            metrics = _classification_metrics(
                y_test_class,
                model.predict(test_table[CLASSIFICATION_FEATURE_SETS[class_set]]),
            )
            rows.append({"fold": fold, "task": "fatigue_proxy", "model": name, "feature_set": class_set, "unseen_user_count": test_table["user_id"].nunique(), **metrics})
        for name, model in regressors.items():
            metrics = _regression_metrics(
                test_table["next_day_active_minutes"],
                model.predict(test_table[REGRESSION_FEATURE_SETS[reg_set]]),
            )
            rows.append({"fold": fold, "task": "next_day_activity", "model": name, "feature_set": reg_set, "unseen_user_count": test_table["user_id"].nunique(), **metrics})

    fold_metrics = pd.DataFrame(rows)
    classification_rows = fold_metrics[fold_metrics["task"] == "fatigue_proxy"]
    aggregated: dict[str, dict[str, float]] = {}
    for model_name, group in classification_rows.groupby("model"):
        aggregated[model_name] = {
            "unseen_user_count": float(group["unseen_user_count"].mean()),
            **{metric: float(group[metric].mean()) for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
            "f1_macro_std": float(group["f1_macro"].std(ddof=1) if len(group) > 1 else 0.0),
        }
    if return_fold_metrics:
        return aggregated, fold_metrics
    return aggregated


def chronological_user_split(model_table: pd.DataFrame, test_size: float = 0.25) -> tuple[pd.Index, pd.Index]:
    train_indices: list[int] = []
    test_indices: list[int] = []
    for _, group in model_table.sort_values("date").groupby("user_id"):
        test_count = max(1, ceil(len(group) * test_size))
        if len(group) <= test_count:
            continue
        train_indices.extend(group.index[:-test_count])
        test_indices.extend(group.index[-test_count:])
    if not train_indices or not test_indices:
        raise ValueError("Not enough longitudinal records for a chronological user split")
    return pd.Index(train_indices), pd.Index(test_indices)


def select_feature_set(
    table: pd.DataFrame,
    target: pd.Series,
    feature_sets: dict[str, list[str]],
    task: str,
    random_state: int,
) -> tuple[str, pd.DataFrame]:
    cv = _group_cv(table)
    if task == "classification":
        estimator: BaseEstimator = RandomForestClassifier(
            n_estimators=40, max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=random_state, n_jobs=1
        )
        scoring = "f1_macro"
    else:
        estimator = RandomForestRegressor(
            n_estimators=40, max_depth=5, min_samples_leaf=4, random_state=random_state, n_jobs=1
        )
        scoring = "neg_mean_absolute_error"

    rows = []
    for name, columns in feature_sets.items():
        scores = cross_val_score(estimator, table[columns], target, groups=table["user_id"], cv=cv, scoring=scoring, n_jobs=1)
        reported = scores if task == "classification" else -scores
        rows.append({
            "task": "fatigue_proxy" if task == "classification" else "next_day_activity",
            "feature_set": name,
            "metric": "macro_f1" if task == "classification" else "mae",
            "mean": float(reported.mean()),
            "std": float(reported.std(ddof=1) if len(reported) > 1 else 0.0),
        })
    comparison = pd.DataFrame(rows)
    best_index = comparison["mean"].idxmax() if task == "classification" else comparison["mean"].idxmin()
    return str(comparison.loc[best_index, "feature_set"]), comparison


def calculate_feature_importance(
    selected_model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        selected_model,
        x_test,
        y_test,
        scoring="f1_macro",
        n_repeats=8,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame({
        "feature": x_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _fit_task_models(
    table: pd.DataFrame,
    target: pd.Series,
    columns: list[str],
    task: str,
    random_state: int,
    search_iterations: int,
) -> tuple[dict[str, BaseEstimator], dict[str, dict[str, Any]]]:
    cv = _group_cv(table)
    x = table[columns]
    groups = table["user_id"]
    if task == "classification":
        xgb_objective = "binary:logistic" if target.nunique() == 2 else "multi:softprob"
        models: dict[str, BaseEstimator] = {
            "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]),
        }
        candidates = {
            "random_forest": (RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=1), RF_CLASSIFIER_SPACE),
            "xgboost": (XGBClassifier(objective=xgb_objective, eval_metric="logloss", random_state=random_state, n_jobs=1), XGB_SPACE),
        }
        scoring = "f1_macro"
    else:
        models = {"baseline_mean": DummyRegressor(strategy="mean")}
        candidates = {
            "random_forest": (RandomForestRegressor(random_state=random_state, n_jobs=1), RF_REGRESSOR_SPACE),
            "xgboost": (XGBRegressor(objective="reg:squarederror", random_state=random_state, n_jobs=1), XGB_SPACE),
        }
        scoring = "neg_mean_absolute_error"

    for model in models.values():
        model.fit(x, target)
    best_params: dict[str, dict[str, Any]] = {}
    for offset, (name, (estimator, parameter_space)) in enumerate(candidates.items()):
        if search_iterations <= 2:
            parameter_space = {**parameter_space, "n_estimators": [50]}
        search = RandomizedSearchCV(
            estimator,
            parameter_space,
            n_iter=search_iterations,
            scoring=scoring,
            cv=cv,
            random_state=random_state + offset,
            n_jobs=1,
            refit=True,
        )
        search.fit(x, target, groups=groups)
        models[name] = search.best_estimator_
        best_params[name] = _json_safe_params(search.best_params_)
    return models, best_params


def _group_cv(table: pd.DataFrame) -> GroupKFold:
    user_count = table["user_id"].nunique()
    if user_count < 2:
        raise ValueError("At least two users are required for group-aware validation")
    return GroupKFold(n_splits=min(4, user_count))


def _classification_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
    }


def _regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
    }


def _confidence_intervals(fold_metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for (task, model), group in fold_metrics.groupby(["task", "model"]):
        metric = "f1_macro" if task == "fatigue_proxy" else "mae"
        values = group[metric].dropna().astype(float)
        mean = float(values.mean())
        half_width = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        output[f"{task}:{model}"] = {"metric": metric, "mean": mean, "lower_95": mean - half_width, "upper_95": mean + half_width}
    return output


def _aggregate_cross_user_regression(fold_metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    rows = fold_metrics[fold_metrics["task"] == "next_day_activity"]
    output: dict[str, dict[str, float]] = {}
    for model_name, group in rows.groupby("model"):
        output[model_name] = {
            "unseen_user_count": float(group["unseen_user_count"].mean()),
            "mae": float(group["mae"].mean()),
            "rmse": float(group["rmse"].mean()),
            "mae_std": float(group["mae"].std(ddof=1) if len(group) > 1 else 0.0),
        }
    return output


def _select_activity_model(models: dict[str, BaseEstimator], metrics: dict[str, dict[str, float]]) -> BaseEstimator:
    candidates = {name: value["mae"] for name, value in metrics.items() if name in {"random_forest", "xgboost"}}
    return models[min(candidates, key=candidates.get)]


def _json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: (value.item() if isinstance(value, np.generic) else value) for key, value in params.items()}
