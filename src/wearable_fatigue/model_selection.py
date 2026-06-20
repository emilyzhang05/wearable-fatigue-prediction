from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


XGBOOST_SELECTION_MARGIN = 0.02


@dataclass(frozen=True)
class SelectedModel:
    model_name: str
    rationale: str


def select_main_classifier(
    cross_user_metrics: dict[str, dict[str, float]],
    margin: float = XGBOOST_SELECTION_MARGIN,
    fold_metrics: pd.DataFrame | None = None,
) -> SelectedModel:
    random_forest_f1 = cross_user_metrics.get("random_forest", {}).get("f1_macro", 0.0)
    xgboost_f1 = cross_user_metrics.get("xgboost", {}).get("f1_macro", 0.0)

    stable = True
    stability_note = ""
    if fold_metrics is not None and not fold_metrics.empty:
        paired = fold_metrics.pivot_table(index="fold", columns="model", values="f1_macro", aggfunc="first")
        if {"random_forest", "xgboost"}.issubset(paired.columns):
            differences = (paired["xgboost"] - paired["random_forest"]).dropna()
            if len(differences) > 1:
                lower_95 = float(differences.mean() - 1.96 * differences.std(ddof=1) / np.sqrt(len(differences)))
                win_rate = float((differences > 0).mean())
                stable = lower_95 > 0 and win_rate >= 0.6
                stability_note = f" Paired-fold win rate was {win_rate:.0%} and the improvement's lower 95% bound was {lower_95:.3f}."

    if xgboost_f1 >= random_forest_f1 + margin and stable:
        return SelectedModel(
            model_name="xgboost",
            rationale=(
                f"XGBoost selected because unseen-user macro F1 ({xgboost_f1:.3f}) "
                f"beats Random Forest ({random_forest_f1:.3f}) by at least {margin:.2f} and the gain is stable across folds."
            ),
        )

    return SelectedModel(
        model_name="random_forest",
        rationale=(
            f"Random Forest selected because XGBoost unseen-user macro F1 ({xgboost_f1:.3f}) "
            f"does not deliver a stable improvement over Random Forest ({random_forest_f1:.3f}) above the {margin:.2f} threshold; "
            f"the simpler model is preferred for interpretability and lower overfit risk.{stability_note}"
        ),
    )
