# Wearable recovery and activity prediction

Can everyday Fitbit data tell us when someone may need more recovery, or how active they will be tomorrow?

This project looks at real Fitbit sleep, heart-rate, and activity records. It turns those records into a simple visual story about recovery, routines, and what wearable data can and cannot predict.

**[Open the live Streamlit dashboard](https://emilyzhang05-wearable-fatigue-prediction-dashboardapp-nfbijn.streamlit.app/)**

```bash
streamlit run dashboard/app.py
```

## What it found

- **Recovery:** getting less sleep than usual and being heavily active recently were the clearest signs that someone may need more recovery.
- **Tomorrow's activity:** the project could not predict this reliably for a new person. A simple average worked slightly better.
- **Different routines:** users separated into three recognizable groups: consistent active, high-activity/low-recovery, and lower-activity weekday users.
- **Model choice:** we compared three common approaches and used Random Forest because it performed almost the same as XGBoost while being easier to explain.

The negative result is intentional. A useful project should say when the data is not strong enough, rather than presenting every prediction as a success.

## How it works

For each person, the project looks at:

- how long they slept;
- whether they slept less than usual;
- whether their resting heart rate changed;
- how many steps and active minutes they recorded;
- how demanding their recent activity was.

It compares today with that person's usual routine. It also tests whether yesterday, several recent days, or a weighted recent history gives a clearer picture.

The project then checks its conclusions on people who were not included during training. This matters because recognizing someone already seen is much easier than working for a genuinely new user.

## Feature engineering and tuning

The original Fitbit files are useful, but they are not ready for prediction on their own. The project first cleans and joins daily activity, sleep, and heart-rate records into one timeline per user. From there, it creates features that are closer to how people actually experience recovery:

- **Personal baselines:** each user's usual sleep, activity, and resting heart rate are estimated from their own past data, so the model compares a person against themselves instead of against a generic average.
- **Recovery signals:** sleep debt and resting heart-rate deviation are used to capture whether the body looks less recovered than usual.
- **Activity load:** steps, active minutes, and workout intensity are combined into a simple activity-load score.
- **Recent history:** the model tests yesterday's values, 3-day, 5-day, and 7-day averages, plus a recency-weighted average, because fatigue and recovery can build up over several days.

For modeling, the project compares Logistic Regression, Random Forest, and XGBoost against simple baselines. Random Forest and XGBoost are also tuned with group-aware cross-validation, meaning the model is tested on users it did not see during training. The tuning focuses on a small set of high-impact choices, such as tree depth, number of trees, learning rate, and regularization, instead of trying every possible setting.

The final model choice is not based only on the highest score. XGBoost performed slightly better in some tests, but the improvement over Random Forest was small and not stable enough across unseen users. Random Forest was selected because it kept nearly the same performance while being easier to explain.

## What is being predicted

There are two questions:

1. **Does this look like a day when the person may need more recovery?** Fitbit does not contain a measured fatigue answer, so this is a transparent warning signal rather than a diagnosis.
2. **How many active minutes will the person record tomorrow?** This is a real measured outcome, but the available data was not strong enough to predict it reliably for new users.

## Why Random Forest

XGBoost scored only slightly higher than Random Forest, and that small difference changed across tests. Random Forest was chosen because it gave almost the same result while being simpler to explain. Logistic Regression was also tested and performed strongly because the recovery warning is based on a straightforward combination of sleep, heart rate, and recent activity.

## What the dashboard shows

- **Past-day comparison:** whether today, yesterday, or several days of history worked best.
- **What mattered most:** which wearable signals contributed most to the recovery result.
- **Behavior map:** how the users separated into three different routine types.
- **Key conclusions and limitations:** what the analysis means in plain language.

## Setup

```bash
pip install -r requirements.txt
pytest -q
streamlit run dashboard/app.py
```

To rebuild the saved Fitbit results:

```bash
PYTHONPATH=src python3 -m wearable_fatigue.train \
  --dataset fitbit \
  --raw-dir data/raw/fitbit \
  --search-iterations 30 \
  --outer-repeats 5 \
  --output-dir .
```

## Important limitation

The Fitbit data never asks people whether they actually felt tired. The project can identify signs associated with needing recovery, but it cannot measure fatigue or provide medical advice.

The dataset is also small: 24 people over roughly one month, leaving 256 days with enough history for the final analysis. A stronger version would collect longer histories and ask users directly about tiredness or readiness.

## Data

The main source is [Fitbit Fitness Tracker Data](https://www.kaggle.com/datasets/arashnic/fitbit), published on Kaggle by Möbius (`arashnic`). It provides 410 joined daily records from 24 users.

[MMASH v1.0.0](https://doi.org/10.13026/cerq-fc86) provides additional sleep, stress, and body-signal context, but its recordings are too short to produce the main results.

Raw datasets are not committed. See [DATA_SOURCES.md](DATA_SOURCES.md) for setup and attribution.

The research behind the history features is linked in [the project report](reports/project_summary.md). The model evaluation code and saved results are available in [`src/wearable_fatigue`](src/wearable_fatigue) and [`reports`](reports).
