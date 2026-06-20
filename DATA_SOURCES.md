# Data Sources

## Main Dataset: Fitbit Fitness Tracker Data

Use the Kaggle Fitbit dataset as the primary modeling dataset:

https://www.kaggle.com/datasets/arashnic/fitbit

Credit: **Fitbit Fitness Tracker Data**, published on Kaggle by Möbius (`arashnic`). The project uses the `4.12.16-5.12.16` Fitabase export for the main model results.

Expected raw files:

- `dailyActivity_merged.csv`
- `sleepDay_merged.csv`
- `heartrate_seconds_merged.csv` if available

Place these files in:

```text
data/raw/fitbit/
```

Then run:

```bash
PYTHONPATH=src python3 -m wearable_fatigue.train --dataset fitbit --raw-dir data/raw/fitbit --output-dir .
```

## Supporting Dataset: PhysioNet MMASH

Use MMASH as a supporting research/validation dataset:

https://physionet.org/content/mmash/1.0.0/

Required citation:

> Rossi, A., Da Pozzo, E., Menicagli, D., Tremolanti, C., Priami, C., Sirbu, A., Clifton, D., Martini, C., & Morelli, D. (2020). Multilevel Monitoring of Activity and Sleep in Healthy People (version 1.0.0). PhysioNet. https://doi.org/10.13026/cerq-fc86

Original publication:

> Rossi, A., et al. (2020). A Public Dataset of 24-h Multi-Levels Psycho-Physiological Responses in Young Healthy Adults. Data, 5(4), 91. https://doi.org/10.3390/data5040091

MMASH is useful because it includes richer physiology, stress, sleep-quality, and actigraphy signals, but it is not the primary prediction dataset because each participant has a short recording window.

Expected structure:

```text
data/raw/mmash/
  DataPaper/
    user_1/
      sleep.csv
      Actigraph.csv
      questionnaire.csv
    user_2/
      sleep.csv
      Actigraph.csv
      questionnaire.csv
```

Then run:

```bash
PYTHONPATH=src python3 -m wearable_fatigue.train --dataset mmash --raw-dir data/raw/mmash/DataPaper --output-dir .
```

## Privacy and Git Policy

Raw datasets are ignored by Git and should not be committed. The repository commits code, tests, documentation, small generated summaries, and the compact derived model table used by the Streamlit demo.
