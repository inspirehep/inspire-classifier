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

import logging
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from inspire_classifier.core.model import Classifier, LanguageModel
from inspire_classifier.core.preprocessor import split_and_save_data_for_training
from inspire_classifier.core.utils import (
    get_classifier_model_path,
    get_data_path,
    get_language_model_path,
)

logger = logging.getLogger(__name__)


def create_directories(base_path):
    """Create the project data and model directories"""
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(get_data_path(base_path, ""), exist_ok=True)
    os.makedirs(get_language_model_path(base_path, ""), exist_ok=True)
    os.makedirs(get_classifier_model_path(base_path, ""), exist_ok=True)


def split_data(base_path, val_fraction=0.1):
    """
    Splits the data into training and validation set.
    """
    try:
        split_and_save_data_for_training(
            dataframe_path=get_data_path(base_path, "train_valid_data.df"),
            dest_dir=get_data_path(base_path, "train_valid_data.csv"),
            val_fraction=val_fraction,
        )
    except IOError as error:
        raise IOError(
            "Training dataframe not found. Make sure the file is present in the right "
            "directory. Please use the path specified in config.py for "
            "CLASSIFIER_DATAFRAME_PATH relative to the CLASSIFIER_BASE_PATH."
        ) from error


def finetune_and_save_language_model(
    base_path,
    cuda_device_id=0,
    language_model_batch_size=64,
    minimum_word_frequency=2,
    maximum_vocabulary_size=60000,
    language_model_cycle_length=15,
):
    """
    Finetunes the pretrained language model on our dataset.
    """
    try:
        language_model = LanguageModel(
            train_valid_data_dir=get_data_path(base_path, "train_valid_data.csv"),
            data_itos_path=get_data_path(base_path, "train_valid_data_itos.pkl"),
            cuda_device_id=cuda_device_id,
            batch_size=language_model_batch_size,
            minimum_word_frequency=minimum_word_frequency,
            maximum_vocabulary_size=maximum_vocabulary_size,
        )
    except IOError as error:
        raise IOError(
            "Training files, language model data directory, or data ITOS do not exist."
        ) from error

    try:
        language_model.train(
            finetuned_language_model_encoder_save_path=get_language_model_path(
                base_path,
            ),
            cycle_length=language_model_cycle_length,
        )
    except IOError as error:
        raise IOError(
            "Unable to save the finetuned language model. Please check that the "
            "language model data directory exists."
        ) from error


def train_and_save_classifier(
    base_path, cuda_device_id=0, classifier_batch_size=128, classifier_cycle_length=15
):
    """
    Trains the classifier on our dataset and save the weights.
    """
    try:
        classifier = Classifier(cuda_device_id=cuda_device_id, train=True)
    except IOError as error:
        raise IOError("Data ITOS not found.") from error

    try:
        classifier.load_training_and_validation_data(
            train_valid_data_dir=get_data_path(base_path, "train_valid_data.csv"),
            data_itos_path=get_data_path(base_path, "train_valid_data_itos.pkl"),
            batch_size=classifier_batch_size,
        )
    except IOError as error:
        raise IOError(
            "Training and Validation data for Classifier not found."
        ) from error

    classifier.initialize_learner()

    try:
        logger.info(
            get_language_model_path(
                base_path,
            )
        )
        classifier.load_finetuned_language_model_weights(
            finetuned_language_model_encoder_path=get_language_model_path(
                base_path,
            )
        )
        classifier.load_finetuned_language_model_weights(
            finetuned_language_model_encoder_path=get_language_model_path(
                base_path,
            )
        )
    except IOError as error:
        raise IOError("Finetuned Language Model Encoder does not exist.") from error

    try:
        classifier.train(
            trained_classifier_save_path=get_classifier_model_path(
                base_path,
            ),
            cycle_length=classifier_cycle_length,
        )
    except IOError as error:
        raise IOError("Unable to save the trained classifier.") from error


def train(
    base_path,
    cuda_device_id,
    val_fraction,
    language_model_batch_size,
    minimum_word_frequency,
    maximum_vocabulary_size,
    language_model_cycle_length,
    classifier_batch_size,
    classifier_cycle_length,
):
    """
    Runs the complete training pipeline.
    """
    create_directories(base_path)
    split_data(base_path, val_fraction)
    finetune_and_save_language_model(
        base_path,
        cuda_device_id,
        language_model_batch_size,
        minimum_word_frequency,
        maximum_vocabulary_size,
        language_model_cycle_length,
    )
    train_and_save_classifier(
        base_path, cuda_device_id, classifier_batch_size, classifier_cycle_length
    )


def validate(validation_df, base_path, cuda_device_id=0, softmax_temperature=0.25):
    classifier = Classifier(cuda_device_id=cuda_device_id)
    try:
        classifier.load_trained_classifier_weights(
            get_classifier_model_path(
                base_path,
            )
        )
    except IOError as error:
        raise IOError("There was a problem loading the classifier model") from error
    predictions = []
    validation_df = validation_df.sample(frac=1, random_state=42)
    for _, row in tqdm(validation_df.iterrows(), total=len(validation_df.label.values)):
        predicted_value = classifier.predict(row.text, temperature=softmax_temperature)
        predicted_class = np.argmax(predicted_value)
        predictions.append(predicted_class)

    validation_df.insert(2, "predicted_label", predictions)
    validation_df.to_csv(
        f"{os.path.join(base_path, 'data')}/validation_results.csv", index=False
    )
    f1_validation_score = f1_score(
        validation_df["label"], validation_df["predicted_label"], average="micro"
    )
    logger.info(f"f1 score: {f1_validation_score}")

    classification_rep = classification_report(
        validation_df["label"], validation_df["predicted_label"]
    )
    logger.info(f"Classification report:\n{classification_rep}")

    confusion_mat = confusion_matrix(
        validation_df["label"], validation_df["predicted_label"]
    )
    logger.info(f"Confusion matrix:\n{confusion_mat}")
