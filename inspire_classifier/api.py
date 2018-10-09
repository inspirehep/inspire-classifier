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

"""Classifier API."""

from __future__ import absolute_import, division, print_function

from flask import current_app
from inspire_classifier.core.ml.models import (
    Classifier,
    LanguageModel
)
from inspire_classifier.core.preprocessor.preprocessor import (
    generate_and_save_classifier_tokens,
    generate_and_save_language_model_tokens,
    map_and_save_tokens_to_ids_for_classifier,
    map_and_save_tokens_to_ids_for_language_model,
    split_and_save_data_for_language_model_and_classifier
)
import os


def preprocess_and_save_data():
    '''
    Calls the preprocessor functions and processes and saves to disk the training and validation data for the language
    model and the classifier.
    '''
    split_and_save_data_for_language_model_and_classifier(
        current_app.config['CLASSIFIER_DATAFRAME_PATH'], current_app.config['CLASSIFIER_LANGUAGE_MODEL_DATA_DIR'],
        current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], current_app.config['CLASSIFIER_VALIDATION_DATA_FRACTION'],
        current_app.config['CLASSIFIER_CLASSIFICATION_CLASSES']
    )
    generate_and_save_language_model_tokens(current_app.config['CLASSIFIER_LANGUAGE_MODEL_DATA_DIR'])
    map_and_save_tokens_to_ids_for_language_model(
        current_app.config['CLASSIFIER_LANGUAGE_MODEL_DATA_DIR'], current_app.config['CLASSIFIER_DATA_ITOS_PATH'],
        current_app.config['CLASSIFIER_MAXIMUM_VOCABULARY_SIZE'], current_app.config['CLASSIFIER_MINIMUM_WORD_FREQUENCY']
    )
    generate_and_save_classifier_tokens(current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'])
    map_and_save_tokens_to_ids_for_classifier(
        current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], app.config['CLASSIFIER_DATA_ITOS_PATH']
    )


def finetune_and_save_language_model():
    '''
    Finetunes the pretrained (on wikitext103) language model on our dataset.
    '''
    language_model = LanguageModel(
        os.path.join(current_app.config['CLASSIFIER_LANGUAGE_MODEL_DATA_DIR'], 'trn_ids.npy'),
        os.path.join(current_app.config['CLASSIFIER_LANGUAGE_MODEL_DATA_DIR'], 'val_ids.npy'),
        current_app.config['CLASSIFIER_LANGUAGE_MODEL_DATA_DIR'], current_app.config['CLASSIFIER_DATA_ITOS_PATH']
    )
    language_model.load_pretrained_language_model_weights(
        current_app.config['CLASSIFIER_PRETRAINED_LANGUAGE_MODEL_PATH'], app.config['CLASSIFIER_WIKITEXT103_ITOS_PATH']
    )
    language_model.train(
        current_app.config['CLASSIFIER_FINETUNED_LANGUAGE_MODEL_PATH'],
        current_app.config['CLASSIFIER_FINETUNED_LANGUAGE_MODEL_ENCODER_PATH']
    )


def train_and_save_classifier():
    '''
    Trains the classifier on our dataset and save the weights.
    '''
    classifier = Classifier(
        os.path.join(current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], 'trn_ids.npy'),
        os.path.join(current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], 'lbl_trn.npy'),
        os.path.join(current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], 'val_ids.npy'),
        os.path.join(current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], 'lbl_val.npy'),
        current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], current_app.config['CLASSIFIER_DATA_ITOS_PATH']
    )
    classifier.load_finetuned_language_model_weights(
        current_app.config['CLASSIFIER_FINETUNED_LANGUAGE_MODEL_ENCODER_PATH']
    )
    classifier.train(current_app.config['CLASSIFIER_TRAINED_CLASSIFIER_PATH'])


def preprocess_data_and_finetune_language_model_and_train_classifier():
    '''
    Preprocesses the data, finetunes the wikitext103 pretrained language model on it, and trains the classifier on top
    of it; meanwhile saving all the intermediate files and the trained classifier.
    '''
    preprocess_and_save_data()
    finetune_and_save_language_model()
    train_and_save_classifier()


def predict_with_classifier(text):
    '''
    :param text: The string input to classify.
    :return: The class-wise classification scores for the input text.
    '''
    classifier = Classifier(
        os.path.join(current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], 'trn_ids.npy'),
        os.path.join(current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], 'lbl_trn.npy'),
        os.path.join(current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], 'val_ids.npy'),
        os.path.join(current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], 'lbl_val.npy'),
        current_app.config['CLASSIFIER_CLASSIFIER_DATA_DIR'], current_app.config['CLASSIFIER_DATA_ITOS_PATH']
    )
    classifier.load_trained_classifier_weights(current_app.config['CLASSIFIER_TRAINED_CLASSIFIER_PATH'])

    return classifier.predict(text)