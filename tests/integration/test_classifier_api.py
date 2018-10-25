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

from inspire_classifier.api import (
    predict_coreness,
)
from inspire_classifier.utils import path_for
from math import isclose
import numpy as np
import pandas as pd
import pickle


TEST_TITLE = 'Pre-images of extreme points of the numerical range, and applications'
TEST_ABSTRACT = 'We extend the pre-image representation of exposed points of the numerical range of a matrix to all \
extreme points. With that we characterize extreme points which are multiply generated, having at least two linearly \
independent pre-images, as the extreme points which are Hausdorff limits of flat boundary portions on numerical ranges \
of a sequence converging to the given matrix. These studies address the inverse numerical range map and the \
maximum-entropy inference map which are continuous functions on the numerical range except possibly at certain \
multiply generated extreme points. This work also allows us to describe closures of subsets of 3-by-3 matrices having \
the same shape of the numerical range.'


def test_create_directories(trained_pipeline):
    assert path_for('classifier_data').exists()
    assert path_for('language_model_data').exists()
    assert path_for('classifier_model').exists()
    assert (path_for('language_model') / 'wikitext_103').exists()


def test_preprocess_and_save_data(app, trained_pipeline):
    dataframe = pd.read_pickle(path_for('dataframe'))

    # Test core/preprocessor:split_and_save_data_for_language_model_and_classifier
    classifier_training_csv = pd.read_csv(path_for('classifier_data') / 'training_data.csv')
    assert isclose(len(classifier_training_csv),
                   len(dataframe) * (1 - app.config['CLASSIFIER_VALIDATION_DATA_FRACTION']), abs_tol=1)
    classifier_validation_csv = pd.read_csv(path_for('classifier_data') / 'validation_data.csv')
    assert isclose(len(classifier_validation_csv), len(dataframe) * app.config['CLASSIFIER_VALIDATION_DATA_FRACTION'],
                   abs_tol=1)

    language_model_training_csv = pd.read_csv(path_for('language_model_data') / 'training_data.csv')
    assert isclose(len(language_model_training_csv),
                   len(dataframe) * (1 - app.config['CLASSIFIER_VALIDATION_DATA_FRACTION']), abs_tol=1)
    language_model_validation_csv = pd.read_csv(path_for('language_model_data') / 'validation_data.csv')
    assert isclose(len(language_model_validation_csv),
                   len(dataframe) * app.config['CLASSIFIER_VALIDATION_DATA_FRACTION'], abs_tol=1)

    # Test core/preprocessor:generate_and_save_language_model_tokens
    language_model_training_tokens = np.load(path_for('language_model_data') / 'training_tokens.npy')
    assert isclose(len(language_model_training_tokens),
                   len(dataframe) * (1 - app.config['CLASSIFIER_VALIDATION_DATA_FRACTION']), abs_tol=1)
    language_model_validation_tokens = np.load(path_for('language_model_data') / 'validation_tokens.npy')
    assert isclose(len(language_model_validation_tokens),
                   len(dataframe) * app.config['CLASSIFIER_VALIDATION_DATA_FRACTION'], abs_tol=1)
    language_model_training_labels = np.load(path_for('language_model_data') / 'training_labels.npy')
    assert isclose(len(language_model_training_labels),
                   len(dataframe) * (1 - app.config['CLASSIFIER_VALIDATION_DATA_FRACTION']), abs_tol=1)
    language_model_validation_labels = np.load(path_for('language_model_data') / 'validation_labels.npy')
    assert isclose(len(language_model_validation_labels),
                   len(dataframe) * app.config['CLASSIFIER_VALIDATION_DATA_FRACTION'], abs_tol=1)

    # Test core/preprocessor:map_and_save_tokens_to_ids_for_language_model
    data_itos = pickle.load(open(path_for('data_itos'), 'rb'))
    assert len(data_itos) == app.config['CLASSIFIER_MAXIMUM_VOCABULARY_SIZE'] + 2

    language_model_training_ids = np.load(path_for('language_model_data') / 'training_token_ids.npy')
    assert isclose(len(language_model_training_ids),
                   len(dataframe) * (1 - app.config['CLASSIFIER_VALIDATION_DATA_FRACTION']), abs_tol=1)
    language_model_validation_ids = np.load(path_for('language_model_data') / 'validation_token_ids.npy')
    assert isclose(len(language_model_validation_ids),
                   len(dataframe) * app.config['CLASSIFIER_VALIDATION_DATA_FRACTION'], abs_tol=1)

    # Test core/preprocessor:generate_and_save_classifier_tokens
    classifier_training_tokens = np.load(path_for('classifier_data') / 'training_tokens.npy')
    assert isclose(len(classifier_training_tokens),
                   len(dataframe) * (1 - app.config['CLASSIFIER_VALIDATION_DATA_FRACTION']), abs_tol=1)
    classifier_validation_tokens = np.load(path_for('classifier_data') / 'validation_tokens.npy')
    assert isclose(len(classifier_validation_tokens),
                   len(dataframe) * app.config['CLASSIFIER_VALIDATION_DATA_FRACTION'], abs_tol=1)
    classifier_training_labels = np.load(path_for('classifier_data') / 'training_labels.npy')
    assert isclose(len(classifier_training_labels),
                   len(dataframe) * (1 - app.config['CLASSIFIER_VALIDATION_DATA_FRACTION']), abs_tol=1)
    classifier_validation_labels = np.load(path_for('classifier_data') / 'validation_labels.npy')
    assert isclose(len(classifier_validation_labels),
                   len(dataframe) * app.config['CLASSIFIER_VALIDATION_DATA_FRACTION'], abs_tol=1)

    # Test core/preprocessor:map_and_save_tokens_to_ids_for_classifier
    classifier_training_ids = np.load(path_for('classifier_data') / 'training_token_ids.npy')
    assert isclose(len(classifier_training_ids),
                   len(dataframe) * (1 - app.config['CLASSIFIER_VALIDATION_DATA_FRACTION']), abs_tol=1)
    classifier_validation_ids = np.load(path_for('classifier_data') / 'validation_token_ids.npy')
    assert isclose(len(classifier_validation_ids), len(dataframe) * app.config['CLASSIFIER_VALIDATION_DATA_FRACTION'],
                   abs_tol=1)


def test_finetune_and_save_language_model(trained_pipeline):
    assert path_for('pretrained_language_model').exists()
    assert path_for('wikitext103_itos').exists()
    assert path_for('finetuned_language_model_encoder').exists()


def test_train_and_save_classifier(trained_pipeline):
    assert path_for('trained_classifier').exists()


def test_predict_coreness(trained_pipeline):
    assert path_for('data_itos').exists()
    assert path_for('trained_classifier').exists()
    output_dict = predict_coreness(title=TEST_TITLE, abstract=TEST_ABSTRACT)

    assert set(output_dict.keys()) == {'prediction', 'scores'}
    assert output_dict['prediction'] in {'rejected', 'non_core', 'core'}
    assert set(output_dict['scores'].keys()) == {'rejected', 'non_core', 'core'}
    assert isclose(output_dict['scores']['rejected'] + output_dict['scores']['non_core'] + output_dict['scores']['core'],
                   1.0, abs_tol=1e-2)
