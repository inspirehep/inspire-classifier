# Inspire Classifier

## About
INSPIRE module aimed at automatically classifying the new papers that are added to INSPIRE, such as if they are core or not, or the arXiv category corresponding to each of them.

The current implemntation uses the ULMfit approach. Universal Language Model Fine-tuning, is a method for training text classifiers by first pretraining a language model on a large corpus to learn general language features (in this case a pre-loaded model, which was trained using the WikiText-103 dataset is used). The pretrained model is then fine-tuned on the title and abstract of the inpsire dataset before training the classifier on top.



## Installation:
* Install and activate `python 3.11` enviroment (for exmaple using pyenv)
* Install poetry: `pip install poetry==1.8.3`
* Run poetry install: `poetry install`


## Train and upload new classifier model
### 1. Gather training data
Set the enviroment variables for inspire-prod es database and run the [`create_dataset.py`](scripts/create_dataset.py) file, passing the range of years. This will create a `inspire_classifier_dataset.pkl`, containing the label (core, non-core, rejected) as well as the title and abstract of the fetched records. This data will be used in the next step to train the model.

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


### 3. Upload model
The final step is to upload the new model to the s3 bucket and change the config file in the deployment, so after a redeployment, the pods make use of the new classifier model. Either use `rclone copy` to transfer the file into the s3 bucket or use the `upload_to_s3.py` script (small adjustments like adding the credential are needed).

```
poetry run python scripts/upload_to_s3.py
```



## How to build and deploy new classifier image:
**Currently new images have to deployed on [dockerhub](https://hub.docker.com/r/inspirehep/classifier). This is subject to change as images should go to the harbor registry, but changes in deployment are needed first**

1. Build docker image: `docker build -t inspirehep/classifier:<NEW TAG> .`
2. Login with inspirehep user on dockerhub: `docker login`
3. Push image to dockerhub: `docker push inspirehep/classifier:<NEW TAG>`
4. Change `newTag` in the `kustomization.yml` file in the [k8s repo](https://github.com/cern-sis/kubernetes/tree/master/classifier).




## How to run
For testing, the cli of the classifier can be used via `poetry run inspire-classifier 'example title' 'exmaple abstract'`, with the `-b` flag, the basepath to check for the training data, can be passed (which currently should be `-b classifier`).

In the production, the api is used to predict the 'coreness' of records using the `/api/predict/coreness` endpoint and passing `title` and `abstract` as json fields in a POST request (see [this file](inspire_classifier/app.py) for details).
