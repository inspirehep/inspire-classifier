# Inspire classifier training data generator
A script to prepare dataset compliant with inspire-classifier requirements.

### How to run
```
export ES_USERNAME=XXXX
export ES_PASSWORD=XXXX

poetry install
poetry run python create_dataset.py --year-from $YEAR_FROM —year-to $YEAR_TO
```
