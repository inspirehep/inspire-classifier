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

"""Classifier Configuration."""

from __future__ import absolute_import, division, print_function

import os


CLASSIFIER_MAXIMUM_VOCABULARY_SIZE = 60000
CLASSIFIER_MINIMUM_WORD_FREQUENCY = 2
CLASSIFIER_VALIDATION_DATA_FRACTION = 0.1
CLASSIFIER_CLASSIFICATION_CLASSES = ['Rejected', 'NonCore', 'Core']
CLASSIFIER_LANGUAGE_MODEL_CYCLE_LENGTH = 15
CLASSIFIER_CLASSIFIER_CYCLE_LENGTH = 14

### FIXME: Check and change the classifier base path
CLASSIFIER_BASE_PATH = os.path.join(app.instance_path, 'inspire-classifier')
CLASSIFIER_DATA_DIR = os.path.join(CLASSIFIER_BASE_PATH, 'data')
CLASSIFIER_MODELS_DIR = os.path.join(CLASSIFIER_BASE_PATH, 'models')
CLASSIFIER_LANGUAGE_MODEL_DIR = os.path.join(CLASSIFIER_MODELS_DIR, 'language_model')
CLASSIFIER_CLASSIFIER_MODEL_DIR = os.path.join(CLASSIFIER_MODELS_DIR, 'classifier_model')
CLASSIFIER_LANGUAGE_MODEL_DATA_DIR = os.path.join(CLASSIFIER_DATA_DIR, 'language_model_data')
CLASSIFIER_CLASSIFIER_DATA_DIR = os.path.join(CLASSIFIER_DATA_DIR, 'classifier_data')

CLASSIFIER_DATAFRAME_PATH =  os.path.join(CLASSIFIER_DATA_DIR, 'inspire_data.df')
CLASSIFIER_PRETRAINED_LANGUAGE_MODEL_PATH = os.path.join(
    CLASSIFIER_LANGUAGE_MODEL_DIR, 'wikitext_103', 'fwd_wt103.h5')
CLASSIFIER_FINETUNED_LANGUAGE_MODEL_ENCODER_PATH = os.path.join(
    CLASSIFIER_LANGUAGE_MODEL_DIR, 'finetuned_language_encoder_model.h5')
CLASSIFIER_TRAINED_CLASSIFIER_PATH = os.path.join(
    CLASSIFIER_CLASSIFIER_MODEL_DIR, 'trained_classifier_model.h5')
CLASSIFIER_WIKITEXT103_ITOS_PATH = os.path.join(
    CLASSIFIER_LANGUAGE_MODEL_DIR, 'wikitext_103', 'itos_wt103.pkl')
CLASSIFIER_DATA_ITOS_PATH = os.path.join(CLASSIFIER_DATA_DIR, 'data_itos.pkl')
CLASSIFIER_WIKITEXT103_LANGUAGE_MODEL_URL = 'http://files.fast.ai/models/wt103/fwd_wt103.h5'
CLASSIFIER_WIKITEXT103_ITOS_URL = 'http://files.fast.ai/models/wt103/itos_wt103.pkl'