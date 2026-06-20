from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearable_fatigue.clustering import cluster_user_profiles
from wearable_fatigue.data import generate_synthetic_wearable_data
from wearable_fatigue.features import build_model_table
from wearable_fatigue.models import train_models


PROFILE_COLORS = {
    "consistent active user": "#2A9D8F",
    "high-activity low-recovery user": "#E76F51",
    "sedentary weekday user": "#457B9D",
}
FEATURE_LABELS = {
    "sleep_debt_hours": "Sleep debt",
    "previous_activity_load": "Previous activity load",
    "active_minutes_lag_1d": "Previous active minutes",
    "resting_hr_deviation": "Resting HR change",
    "resting_hr_deviation_lag_1d": "Previous resting HR change",
    "sleep_lag_1d": "Previous sleep",
    "sleep_hours": "Sleep duration",
    "weekday": "Day of week",
}
WINDOW_LABELS = {
    "raw_daily": "Current day only",
    "personal_baselines": "Personal baseline",
    "lagged": "Previous day",
    "rolling_windows": "3/5/7-day history",
    "exponentially_weighted": "Recent days weighted more",
}


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, dict[str, object], bool]:
    processed_path = ROOT / "data" / "processed" / "wearable_model_table.csv"
    summary_path = ROOT / "reports" / "model_summary.json"
    if processed_path.exists() and summary_path.exists():
        data = pd.read_csv(processed_path, parse_dates=["date"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return data, summary, False

    raw_data = generate_synthetic_wearable_data(n_users=24, days=75, random_state=42)
    model_table = cluster_user_profiles(build_model_table(raw_data), random_state=42)
    results = train_models(model_table, random_state=42, search_iterations=2, outer_repeats=2)
    summary = {
        "dataset_source": "synthetic fallback",
        "dataset": {
            "users": int(raw_data["user_id"].nunique()),
            "daily_rows": int(len(raw_data)),
            "model_rows": int(len(model_table)),
        },
        "cross_user_generalization": results.cross_user_metrics,
        "cross_user_activity_forecast": results.cross_user_regression_metrics,
        "selected_main_classifier": {
            "model_name": results.selected_main_classifier.model_name,
            "rationale": results.selected_main_classifier.rationale,
        },
        "feature_importance": results.feature_importance.head(10).to_dict(orient="records"),
        "feature_set_comparison": results.feature_set_comparison.to_dict(orient="records"),
        "selected_feature_sets": results.selected_feature_sets,
    }
    return model_table, summary, True


def build_profile_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("behavior_profile")
        .agg(
            users=("user_id", "nunique"),
            avg_sleep=("sleep_hours", "mean"),
            avg_sleep_debt=("sleep_debt_hours", "mean"),
            avg_activity_load=("activity_load", "mean"),
            avg_active_minutes=("active_minutes", "mean"),
            avg_fatigue_score=("fatigue_score", "mean"),
        )
        .reset_index()
    )


st.set_page_config(page_title="Wearable Recovery Study", layout="wide")

data, summary, is_fallback = load_dashboard_data()
dataset = summary.get("dataset", {})
selected_sets = summary.get("selected_feature_sets", {})

st.title("Wearable Recovery & Activity Prediction")
st.markdown(
    "Can everyday Fitbit data tell us when someone may need more recovery, or how active they will be tomorrow?"
)

if is_fallback:
    st.warning("Real processed data was not found, so this page is using the synthetic demonstration dataset.")

st.subheader("What We Found")
st.markdown(
    f"""
- **Model choice:** we used Random Forest because it performed almost the same as XGBoost and was easier to explain.
- **Recovery:** poor sleep and heavy recent activity were the clearest signs that someone may need more recovery.
- **Tomorrow's activity:** the data could not predict this reliably for a new person; a simple average worked slightly better.
- **Different routines:** users naturally separated into consistent, under-recovered, and lower-activity groups.
"""
)

context_cols = st.columns(3)
context_cols[0].metric("Fitbit Users", dataset.get("users", data["user_id"].nunique()))
context_cols[1].metric("Daily Records", dataset.get("daily_rows", len(data)))
context_cols[2].metric("Model-Ready Days", dataset.get("model_rows", len(data)))

st.subheader("What the Project Looks At")
st.markdown(
    "For each person, the project looks at sleep, heart rate, steps, active minutes, and workout "
    "intensity. It compares today with that person's usual routine and checks whether recent days "
    "tell us more than today alone."
)

st.subheader("How Many Past Days Are Useful?")
st.markdown(
    "We compared using only today, yesterday, and several days of history. Taller is better on the "
    "left chart; shorter is better on the right chart. The highlighted bar is the option we used."
)

feature_comparison = pd.DataFrame(summary.get("feature_set_comparison", []))
if not feature_comparison.empty:
    feature_comparison["window"] = feature_comparison["feature_set"].map(WINDOW_LABELS)
    feature_comparison["chosen"] = feature_comparison.apply(
        lambda row: (
            row["feature_set"] == selected_sets.get("fatigue_proxy")
            if row["task"] == "fatigue_proxy"
            else row["feature_set"] == selected_sets.get("next_day_activity")
        ),
        axis=1,
    )
    classification_windows = feature_comparison[feature_comparison["task"] == "fatigue_proxy"]
    activity_windows = feature_comparison[feature_comparison["task"] == "next_day_activity"]

    window_cols = st.columns(2)
    with window_cols[0]:
        classification_chart = px.bar(
            classification_windows,
            x="window",
            y="mean",
            color="chosen",
            error_y="std",
            color_discrete_map={True: "#2A9D8F", False: "#B8C1CC"},
            labels={"mean": "Model score", "window": "Information used", "chosen": "Selected"},
            title="Spotting days that may need more recovery",
        )
        classification_chart.update_layout(showlegend=False)
        st.plotly_chart(classification_chart, width="stretch")
        st.caption("Yesterday was enough to capture most of the useful recovery pattern. Adding more days did not consistently help.")

    with window_cols[1]:
        activity_chart = px.bar(
            activity_windows,
            x="window",
            y="mean",
            color="chosen",
            error_y="std",
            color_discrete_map={True: "#E9C46A", False: "#B8C1CC"},
            labels={"mean": "Prediction error", "window": "Information used", "chosen": "Selected"},
            title="Predicting tomorrow's active minutes",
        )
        activity_chart.update_layout(showlegend=False)
        st.plotly_chart(activity_chart, width="stretch")
        st.caption("Several days of history looked best during development, but the final result still did not work reliably for new people.")

importance = pd.DataFrame(summary.get("feature_importance", []))
if not importance.empty:
    st.subheader("What Mattered Most?")
    st.markdown(
        "Longer bars mean that piece of information mattered more when identifying days that may need extra recovery."
    )
    importance = importance[importance["importance_mean"] > 0].head(8).copy()
    importance["feature_label"] = importance["feature"].map(FEATURE_LABELS).fillna(
        importance["feature"].str.replace("_", " ").str.title()
    )
    importance_chart = px.bar(
        importance,
        x="importance_mean",
        y="feature_label",
        orientation="h",
        error_x="importance_std",
        color_discrete_sequence=["#2A9D8F"],
        labels={"importance_mean": "Relative importance", "feature_label": "Information"},
    )
    importance_chart.update_yaxes(autorange="reversed")
    st.plotly_chart(importance_chart, width="stretch")
    st.markdown(
        "**Main insight:** getting less sleep than usual mattered most, followed by how active the "
        "person had been recently. This is useful for discussing recovery, but it is not a medical diagnosis."
    )

st.subheader("Three Wearable Behavior Profiles")
st.markdown(
    "Each bubble represents a type of routine rather than one person. Further right means more "
    "activity, higher means more signs of needing recovery, and a larger bubble means more missed sleep."
)

profile_summary = build_profile_summary(data)
profile_chart = px.scatter(
    profile_summary,
    x="avg_activity_load",
    y="avg_fatigue_score",
    size="avg_sleep_debt",
    color="behavior_profile",
    text="behavior_profile",
    hover_data={
        "users": True,
        "avg_sleep": ":.1f",
        "avg_active_minutes": ":.0f",
        "avg_sleep_debt": ":.2f",
    },
    color_discrete_map=PROFILE_COLORS,
    labels={
        "avg_activity_load": "Typical activity level",
        "avg_fatigue_score": "Signs of needing recovery",
        "behavior_profile": "Behavior profile",
    },
)
profile_chart.update_traces(textposition="top center")
profile_chart.update_layout(showlegend=False)
st.plotly_chart(profile_chart, width="stretch")

profile_rows = profile_summary.set_index("behavior_profile")
profile_cols = st.columns(3)
for column, profile in zip(profile_cols, PROFILE_COLORS):
    if profile not in profile_rows.index:
        continue
    row = profile_rows.loc[profile]
    with column:
        st.markdown(f"**{profile.title()}**")
        if profile == "consistent active user":
            st.write(f"Active while still sleeping well: about {row['avg_sleep']:.1f} hours per night and the least missed sleep.")
        elif profile == "high-activity low-recovery user":
            st.write("Very active, but also showing the strongest signs of needing recovery. This group may benefit most from rest reminders.")
        else:
            st.write(f"The least active routine, averaging about {row['avg_active_minutes']:.0f} active minutes per day. This group may benefit from gentle activity prompts.")

st.subheader("Important Limitation")
st.markdown(
    "The Fitbit data never asks people whether they actually felt tired. The project therefore "
    "identifies signs associated with needing recovery; it does not measure fatigue or provide "
    "medical advice. The dataset also covers only 24 people for about one month."
)

st.caption(
    "Main data: Fitbit Fitness Tracker Data on Kaggle. Supporting research context: MMASH v1.0.0 "
    "on PhysioNet (Rossi et al., 2020)."
)
