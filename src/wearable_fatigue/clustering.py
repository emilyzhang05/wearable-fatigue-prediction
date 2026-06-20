from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


CLUSTER_FEATURES = [
    "sleep_hours",
    "sleep_debt_hours",
    "resting_hr_deviation",
    "steps",
    "active_minutes",
    "workout_intensity",
    "fatigue_score",
]


def cluster_user_profiles(
    model_table: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    if len(model_table) < n_clusters:
        raise ValueError("n_clusters cannot exceed the number of model rows")

    clustered = model_table.copy()
    user_features = clustered.groupby("user_id")[CLUSTER_FEATURES].mean()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(user_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    user_features["behavior_cluster"] = kmeans.fit_predict(scaled_features)
    profile_names = _name_clusters(user_features)
    user_features["behavior_profile"] = user_features["behavior_cluster"].map(profile_names)

    return clustered.merge(
        user_features[["behavior_cluster", "behavior_profile"]],
        left_on="user_id",
        right_index=True,
        how="left",
    )


def _name_clusters(user_features: pd.DataFrame) -> dict[int, str]:
    profile_names: dict[int, str] = {}
    cluster_summary = user_features.groupby("behavior_cluster").mean(numeric_only=True)

    for cluster_id, row in cluster_summary.iterrows():
        if row["sleep_debt_hours"] > cluster_summary["sleep_debt_hours"].median() and row["steps"] > cluster_summary["steps"].median():
            profile_names[int(cluster_id)] = "high-activity low-recovery user"
        elif row["active_minutes"] > cluster_summary["active_minutes"].median():
            profile_names[int(cluster_id)] = "consistent active user"
        elif row["steps"] < cluster_summary["steps"].median():
            profile_names[int(cluster_id)] = "sedentary weekday user"
        else:
            profile_names[int(cluster_id)] = "balanced routine user"

    return profile_names
