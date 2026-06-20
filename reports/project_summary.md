# Wearable Fatigue Prediction Summary

## Objective

Forecast next-day activity from wearable behavior and demonstrate a clearly labeled morning fatigue-risk proxy for workout/rest recommendations.

## Dataset

- Source: fitbit
- Users: 24
- Daily records: 410
- Model-ready records: 256

These metrics were produced from the real Fitbit Fitness Tracker Data files, not from the synthetic fallback.

## Data Sources and Credits

- Main modeling dataset: Fitbit Fitness Tracker Data, published on Kaggle by Möbius (arashnic). https://www.kaggle.com/datasets/arashnic/fitbit
- Supporting validation dataset: Rossi, A., et al. (2020). Multilevel Monitoring of Activity and Sleep in Healthy People (version 1.0.0). PhysioNet. https://doi.org/10.13026/cerq-fc86

MMASH was loaded separately to validate the supporting data adapter and provide richer physiological, sleep-quality, and stress context. It does not produce the main model scores because it contains approximately one 24-hour recording per participant rather than a longitudinal daily series.

## Chronological Evaluation

The latest 25% of each user's records were held out. Feature-window selection and hyperparameter tuning used only the earlier records.

- Random Forest fatigue macro F1: 0.792
- XGBoost fatigue macro F1: 0.826
- Random Forest next-day active-minutes MAE: 78.7
- XGBoost next-day active-minutes MAE: 78.4

## Repeated Cross-User Generalization

This nested evaluation repeatedly trains on one set of users and evaluates on completely unseen users. Reported values are means across outer folds.

- Unseen users tested: 5
- Random Forest unseen-user macro F1: 0.818
- XGBoost unseen-user macro F1: 0.823

For the directly observed next-day target, lower MAE is better:

- Mean baseline unseen-user MAE: 70.7
- Random Forest unseen-user MAE: 72.9
- XGBoost unseen-user MAE: 72.8

Neither learned regressor beats the unseen-user baseline, so next-day activity generalization is not demonstrated by this sample.

## Selected Tree Model

- Model: random_forest
- Rationale: Random Forest selected because XGBoost unseen-user macro F1 (0.823) does not deliver a stable improvement over Random Forest (0.818) above the 0.02 threshold; the simpler model is preferred for interpretability and lower overfit risk. Paired-fold win rate was 60% and the improvement's lower 95% bound was -0.012.

XGBoost had the slightly higher raw F1 score: 0.823 versus 0.818. Random Forest was selected because this 0.005 difference was too small and inconsistent across folds to establish that XGBoost genuinely generalizes better; the predefined selection rule required a stable improvement of at least 0.020.

Logistic Regression achieved the highest overall unseen-user macro F1 (0.855). Random Forest is the selected model only for the requested Random Forest-versus-XGBoost comparison. The linear model's strength is expected because the proxy target is constructed from an additive weighted rule.

## Selected Feature Families

- Morning fatigue-risk proxy: lagged
- Next-day activity forecast: rolling_windows

## Interpretation Boundary

Fitbit does not provide an observed fatigue outcome. The fatigue classes are a transparent proxy derived from prior sleep, morning heart rate, and previous-day activity. Classification performance measures reproduction of that proxy and is not clinical validation. Next-day active minutes is the primary directly observed prediction target.

## Evaluation Design

- Personal baselines use past records only.
- Morning proxy features use the completed night and activity only through yesterday.
- Next-day forecasting uses information available through the current day.
- Strict 3-, 5-, and 7-day windows and a 3-day-half-life exponentially weighted representation are compared inside training folds.
- Randomized tuning uses group-aware inner folds; outer user and chronological holdouts remain untouched.

## Hypotheses

- H1: supported - Best macro F1 0.877 vs baseline 0.157.
- H2: not supported - Best MAE 78.4 vs baseline 80.9.
- H3: supported - Identified 3 behavior profiles.

## Top Feature Drivers

- sleep_debt_hours: 0.2892
- previous_activity_load: 0.1442
- active_minutes_lag_1d: 0.0214
- resting_hr_deviation: 0.0154
- resting_hr_deviation_lag_1d: 0.0002

## Behavior Profiles

- consistent active user
- high-activity low-recovery user
- sedentary weekday user

## Product Recommendation Layer

The model output is translated into user-facing guidance: normal workout, light workout, rest, or reschedule workout. Recommendations are driven by fatigue risk, sleep debt, resting heart-rate deviation, and previous activity load.
