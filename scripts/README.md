# Inspire classifier training data generator
A script to prepare dataset compliant with inspire-classifier requirements.

### How to run
Run these commands from the repository root:

```
export ES_USERNAME=XXXX
export ES_PASSWORD=XXXX

poetry install
poetry run python scripts/create_dataset.py --year-from 2020 --month-from 1 --year-to 2024 --month-to 12
```

The script writes `inspire_classifier_dataset_2020-01-01_2024-12-01.pkl` in the current directory. The filename includes the requested start and end dates, with the day set to `01`. The optional `--month-from` and `--month-to` arguments default to `1` and `12`, respectively.

The training script reads `inspire_classifier_dataset.pkl` from the current directory. Rename the generated file before starting training:

```
mv inspire_classifier_dataset_2020-01-01_2024-12-01.pkl inspire_classifier_dataset.pkl
poetry run python scripts/train_classifier.py
```
