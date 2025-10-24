# Inspire Classifier

## About
INSPIRE package aimed to automatically classify the new papers that are added to INSPIRE, such as if they are core or not.

The current implementation uses the ULMfit approach. Universal Language Model Fine-tuning, is a method for training text classifiers by first pre-training a language model on a large corpus to learn general language features (in this case a pre-loaded model, which was trained using the WikiText-103 dataset is used). The pre-trained model is then fine-tuned on the title and abstract of the INSPIRE dataset before training the classifier on top.


## Package Usage
```
from inspire_classifier import Classifier

classifier = Classifier(model_path="PATH/TO/MODEL.h5")

title = "Search for new physics in high-energy particle collisions"
abstract = "We present results from a search for beyond..."

result = classifier.predict_coreness(title, abstract)
print(result) --> {'prediction': 'core', 'scores': {'rejected': 0.1, 'non_core': 0.3, 'core': 0.6}}
```



## Installation for local usage and Training:
* Install and activate `python 3.11` environment (for example using pyenv)
* Install poetry: `pip install poetry==1.8.3`
* Run poetry install: `poetry install`


## Train new classifier model
### 1. Gather training data
Set the environment variables for inspire-prod es database and run the [`create_dataset.py`](scripts/create_dataset.py) file, passing the range of years. This will create a `inspire_classifier_dataset.pkl`, containing the label (core, non-core, rejected) as well as the title and abstract of the fetched records. This data will be used in the next step to train the model. Make sure the generated file is called  `inspire_classifier_dataset.pkl`!

```
export ES_USERNAME=XXXX
export ES_PASSWORD=XXXX

poetry run python scripts/create_dataset.py --year-from $YEAR_FROM --month-from $MONTH_FROM --year-to $YEAR_TO --month-to $MONTH_TO

($MONTH_FROM and $MONTH_TO are optional parameters)
```


### 2. Run training and validate model
The [`train_classifier.py`](scripts/train_classifier.py) script will run the commands to train and validate a new model. Configurations changes like the amount of training epochs as well as the train-test split can be adjusted here. In short, the script first splits the pkl file from the first step into a training and a test dataset inside the `classifier/data` folder. The training set is then used to train the model, while the test set is used to evaluate the model after the training is finished. The model will be saved into `classifier/models/language_model/finetuned_language_model_encoder.h5`

```
poetry run python scripts/train_classifier.py
```


### 3. Upload the model to CERN S3
In order to use the new model in production upload it to CERN S3 and follow [this writeup](https://confluence.cern.ch/display/RCSSIS/Update+Airflow+Base+Image+%28with+classifier+model%29+for+INSPIRE)
