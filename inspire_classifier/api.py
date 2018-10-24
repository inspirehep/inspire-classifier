# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2014-2018 CERN.
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


from flask import current_app
from inspire_classifier.domain.models import (
    Classifier,
    LanguageModel
)
from inspire_classifier.domain.preprocessor import (
    generate_and_save_classifier_tokens,
    generate_and_save_language_model_tokens,
    map_and_save_tokens_to_ids_for_classifier,
    map_and_save_tokens_to_ids_for_language_model,
    split_and_save_data_for_language_model_and_classifier
)
from inspire_classifier.utils import path_for
import numpy as np
import requests


def create_directories():
    """Create the project data and model directories"""
    path_for('classifier_data').mkdir(parents=True, exist_ok=True)
    path_for('language_model_data').mkdir(parents=True, exist_ok=True)
    path_for('classifier_model').mkdir(parents=True, exist_ok=True)
    (path_for('language_model') / 'wikitext_103').mkdir(exist_ok=True, parents=True)


def preprocess_and_save_data():
    """
    Prepares the data for training.
    """
    try:
        split_and_save_data_for_language_model_and_classifier(
            dataframe_path=path_for('dataframe'), language_model_data_dir=path_for('language_model_data'),
            classifier_data_dir=path_for('classifier_data'),
            val_fraction=current_app.config['CLASSIFIER_VALIDATION_DATA_FRACTION']
        )
    except IOError as error:
        raise IOError('Training dataframe not found.') from error

    try:
        generate_and_save_language_model_tokens(language_model_data_dir=path_for('language_model_data'))
    except IOError as error:
        raise IOError('Language Model data directory does not exist.') from error

    try:
        map_and_save_tokens_to_ids_for_language_model(
            language_model_data_dir=path_for('language_model_data'), data_itos_path=path_for('data_itos'),
            max_vocab_size=current_app.config['CLASSIFIER_MAXIMUM_VOCABULARY_SIZE'],
            minimum_frequency=current_app.config['CLASSIFIER_MINIMUM_WORD_FREQUENCY']
        )
    except IOError as error:
        raise IOError('Language Model data directory or the data directory do not exist.') from error

    try:
        generate_and_save_classifier_tokens(classifier_data_dir=path_for('classifier_data'))
    except IOError as error:
        raise IOError('Classifier data directory does not exist.') from error

    try:
        map_and_save_tokens_to_ids_for_classifier(classifier_data_dir=path_for('classifier_data'),
                                                  data_itos_path=path_for('data_itos'))
    except IOError as error:
        raise IOError('Classifier data directory or the data ITOS does not exist.') from error


def finetune_and_save_language_model():
    """
    Finetunes the pretrained (on wikitext103) language model on our dataset.
    """
    try:
        language_model = LanguageModel(
            training_data_ids_path=path_for('language_model_data') / 'training_token_ids.npy',
            validation_data_ids_path=path_for('language_model_data') / 'validation_token_ids.npy',
            language_model_model_dir=path_for('language_model_data'),
            data_itos_path=path_for('data_itos'), cuda_device_id=current_app.config['CLASSIFIER_CUDA_DEVICE_ID'],
            batch_size=current_app.config['CLASSIFIER_LANGUAGE_MODEL_BATCH_SIZE']
        )
    except IOError as error:
        raise IOError('Training files, language model data directory, or data ITOS do not exist.') from error

    if not path_for('pretrained_language_model').exists():
        wikitext103_language_model_response = requests.get(
            current_app.config['CLASSIFIER_WIKITEXT103_LANGUAGE_MODEL_URL'], allow_redirects=True)
        wikitext103_language_model_response.raise_for_status()
        with open(path_for('pretrained_language_model'), 'wb') as fd:
            fd.write(wikitext103_language_model_response.content)
    if not path_for('wikitext103_itos').exists():
        wikitext103_itos_response = requests.get(current_app.config['CLASSIFIER_WIKITEXT103_ITOS_URL'],
                                                 allow_redirects=True)
        wikitext103_itos_response.raise_for_status()
        with open(path_for('wikitext103_itos'), 'wb') as fd:
            fd.write(wikitext103_itos_response.content)

    try:
        language_model.load_pretrained_language_model_weights(
            pretrained_language_model_path=path_for('pretrained_language_model'),
            wikitext103_itos_path=path_for('wikitext103_itos')
        )
    except IOError as error:
        raise IOError('Wikitext103 pretrained language model and Wikitext103 ITOS do not exist.') from error

    try:
        language_model.train(finetuned_language_model_encoder_save_path=path_for('finetuned_language_model_encoder'),
                             cycle_length=current_app.config['CLASSIFIER_LANGUAGE_MODEL_CYCLE_LENGTH'])
    except IOError as error:
        raise IOError('Unable to save the finetuned language model.') from error


def train_and_save_classifier():
    """
    Trains the classifier on our dataset and save the weights.
    """
    try:
        classifier = Classifier(data_itos_path=path_for('data_itos'),
                                number_of_classes=3, cuda_device_id=current_app.config['CLASSIFIER_CUDA_DEVICE_ID'])
    except IOError as error:
        raise IOError('Data ITOS not found.') from error

    try:
        classifier.load_training_and_validation_data(
            training_data_ids_path=path_for('classifier_data') / 'training_token_ids.npy',
            training_data_labels_path=path_for('classifier_data') / 'training_labels.npy',
            validation_data_ids_path=path_for('classifier_data') / 'validation_token_ids.npy',
            validation_data_labels_path=path_for('classifier_data') / 'validation_labels.npy',
            classifier_data_dir=path_for('classifier_data'),
            batch_size=current_app.config['CLASSIFIER_CLASSIFIER_BATCH_SIZE']
        )
    except IOError as error:
        raise IOError('Training and Validation data for Classifier not found.') from error

    classifier.initialize_learner()

    try:
        classifier.load_finetuned_language_model_weights(
            finetuned_language_model_encoder_path=path_for('finetuned_language_model_encoder')
        )
    except IOError as error:
        raise IOError('Finetuned Language Model Encoder does not exist.') from error

    try:
        classifier.train(trained_classifier_save_path=path_for('trained_classifier'),
                         cycle_length=current_app.config['CLASSIFIER_CLASSIFIER_CYCLE_LENGTH'])
    except IOError as error:
        raise IOError('Unable to save the trained classifier.') from error


def train():
    """
    Runs the complete training pipeline.
    """
    create_directories()
    preprocess_and_save_data()
    finetune_and_save_language_model()
    train_and_save_classifier()


def predict_coreness(title, abstract):
    """
    Predicts class-wise probabilities given the title and abstract.
    """
    text = title + abstract
    categories = ['rejected', 'non_core', 'core']
    try:
        classifier = Classifier(data_itos_path=path_for('data_itos'),
                                number_of_classes=3, cuda_device_id=current_app.config['CLASSIFIER_CUDA_DEVICE_ID'])
    except IOError as error:
        raise IOError('Data ITOS not found.') from error

    try:
        classifier.load_trained_classifier_weights(path_for('trained_classifier'))
    except IOError as error:
        raise IOError('Could not load the trained classifier weights.') from error

    class_probabilities = classifier.predict(text)
    assert len(class_probabilities) == 3

    predicted_class = categories[np.argmax(class_probabilities)]
    output_dict = {'prediction': predicted_class}
    output_dict['scores'] = dict(zip(categories, class_probabilities))

    return output_dict
