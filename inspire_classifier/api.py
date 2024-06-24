# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2014-2024 CERN.
#
# INSPIRE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# INSPIRE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with INSPIRE. If not, see <http://www.gnu.org/licenses/>.
#
# In applying this license, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.
#
# Modified from the fastai library (https://github.com/fastai/fastai).

"""Classifier API."""


from pprint import pprint

import numpy as np
from flask import current_app
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from inspire_classifier.domain.models import Classifier, LanguageModel
from inspire_classifier.domain.preprocessor import split_and_save_data_for_training
from inspire_classifier.utils import path_for


def create_directories():
    """Create the project data and model directories"""
    path_for("data").mkdir(parents=True, exist_ok=True)
    path_for("language_model").mkdir(parents=True, exist_ok=True)
    path_for("classifier_model").mkdir(parents=True, exist_ok=True)


def split_data():
    """
    Splits the data into training and validation set.
    """
    try:
        split_and_save_data_for_training(
            dataframe_path=path_for("dataframe"),
            dest_dir=path_for("train_valid_data"),
            val_fraction=current_app.config["CLASSIFIER_VALIDATION_DATA_FRACTION"],
        )
    except IOError as error:
        raise IOError(
            "Training dataframe not found. Make sure the file is present in the right directory. "
            "Please use the path specified in config.py for CLASSIFIER_DATAFRAME_PATH relative to the "
            "CLASSIFIER_BASE_PATH."
        ) from error


def finetune_and_save_language_model():
    """
    Finetunes the pretrained language model on our dataset.
    """
    try:
        language_model = LanguageModel(
            train_valid_data_dir=path_for("train_valid_data"),
            data_itos_path=path_for("data_itos"),
            cuda_device_id=current_app.config["CLASSIFIER_CUDA_DEVICE_ID"],
            batch_size=current_app.config["CLASSIFIER_LANGUAGE_MODEL_BATCH_SIZE"],
            minimum_word_frequency=current_app.config[
                "CLASSIFIER_MINIMUM_WORD_FREQUENCY"
            ],
            maximum_vocabulary_size=current_app.config[
                "CLASSIFIER_MAXIMUM_VOCABULARY_SIZE"
            ],
        )
    except IOError as error:
        raise IOError(
            "Training files, language model data directory, or data ITOS do not exist."
        ) from error

    try:
        language_model.train(
            finetuned_language_model_encoder_save_path=path_for(
                "finetuned_language_model_encoder"
            ),
            cycle_length=current_app.config["CLASSIFIER_LANGUAGE_MODEL_CYCLE_LENGTH"],
        )
    except IOError as error:
        raise IOError(
            "Unable to save the finetuned language model. Please check that the language model data directory "
            "exists."
        ) from error


def train_and_save_classifier():
    """
    Trains the classifier on our dataset and save the weights.
    """
    try:
        classifier = Classifier(
            cuda_device_id=current_app.config["CLASSIFIER_CUDA_DEVICE_ID"]
        )
    except IOError as error:
        raise IOError("Data ITOS not found.") from error

    try:
        classifier.load_training_and_validation_data(
            train_valid_data_dir=path_for("train_valid_data"),
            data_itos_path=path_for("data_itos"),
            batch_size=current_app.config["CLASSIFIER_CLASSIFIER_BATCH_SIZE"],
        )
    except IOError as error:
        raise IOError(
            "Training and Validation data for Classifier not found."
        ) from error

    classifier.initialize_learner()

    try:
        print(path_for("finetuned_language_model_encoder"))
        classifier.load_finetuned_language_model_weights(
            finetuned_language_model_encoder_path=path_for(
                "finetuned_language_model_encoder"
            )
        )
    except IOError as error:
        raise IOError("Finetuned Language Model Encoder does not exist.") from error

    try:
        classifier.train(
            trained_classifier_save_path=path_for("trained_classifier"),
            cycle_length=current_app.config["CLASSIFIER_CLASSIFIER_CYCLE_LENGTH"],
        )
    except IOError as error:
        raise IOError("Unable to save the trained classifier.") from error


def train():
    """
    Runs the complete training pipeline.
    """
    create_directories()
    split_data()
    finetune_and_save_language_model()
    train_and_save_classifier()


def predict_coreness(title, abstract):
    """
    Predicts class-wise probabilities given the title and abstract.
    """
    text = title + " <ENDTITLE> " + abstract
    categories = ["rejected", "non_core", "core"]
    try:
        classifier = Classifier(
            cuda_device_id=current_app.config["CLASSIFIER_CUDA_DEVICE_ID"]
        )
    except IOError as error:
        raise IOError("Data ITOS not found.") from error

    try:
        classifier.load_trained_classifier_weights(path_for("trained_classifier"))
    except IOError as error:
        raise IOError("Could not load the trained classifier weights.") from error

    class_probabilities = classifier.predict(
        text, temperature=current_app.config["CLASSIFIER_SOFTMAX_TEMPERATUR"]
    )
    assert len(class_probabilities) == 3

    predicted_class = categories[np.argmax(class_probabilities)]
    output_dict = {"prediction": predicted_class}
    output_dict["scores"] = dict(zip(categories, class_probabilities))

    return output_dict


def validate(validation_df):
    classifier = Classifier(
        cuda_device_id=current_app.config["CLASSIFIER_CUDA_DEVICE_ID"]
    )
    try:
        classifier.load_trained_classifier_weights(path_for("trained_classifier"))
    except IOError as error:
        raise IOError("There was a problem loading the classifier model") from error
    predictions = []
    true_labels = []
    validation_df = validation_df.sample(frac=1, random_state=42)
    for _, row in tqdm(
        validation_df.iterrows(), total=len(validation_df.labels.values)
    ):
        predicted_value = classifier.predict(
            row.text, temperature=current_app.config["CLASSIFIER_SOFTMAX_TEMPERATUR"]
        )
        predicted_class = np.argmax(predicted_value)
        predictions.append(predicted_class)
        true_labels.append(row.labels)

    print("f1 score ", f1_score(true_labels, predictions, average="micro"))
    pprint(classification_report(true_labels, predictions))
    pprint(confusion_matrix(true_labels, predictions))
