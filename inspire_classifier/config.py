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

"""Classifier Configuration."""


CLASSIFIER_MAXIMUM_VOCABULARY_SIZE = 60000
CLASSIFIER_MINIMUM_WORD_FREQUENCY = 2
CLASSIFIER_VALIDATION_DATA_FRACTION = 0.1
CLASSIFIER_LANGUAGE_MODEL_CYCLE_LENGTH = 15
CLASSIFIER_CLASSIFIER_CYCLE_LENGTH = 15
CLASSIFIER_LANGUAGE_MODEL_BATCH_SIZE = 64
CLASSIFIER_CLASSIFIER_BATCH_SIZE = 128
CLASSIFIER_SOFTMAX_TEMPERATUR = 0.25
CLASSIFIER_CUDA_DEVICE_ID = 0  # set to 0 to use a GPU

CLASSIFIER_DATA_PATH = "data"
CLASSIFIER_LANGUAGE_MODEL_PATH = "models/language_model"
CLASSIFIER_CLASSIFIER_MODEL_PATH = "models/classifier_model"
CLASSIFIER_DATAFRAME_PATH = "data/train_valid_data.df"
CLASSIFIER_TRAIN_VALID_DATA_PATH = "data/train_valid_data.csv"
CLASSIFIER_FINETUNED_LANGUAGE_MODEL_ENCODER_PATH = (
    "models/language_model/finetuned_language_model_encoder.h5"
)
CLASSIFIER_TRAINED_CLASSIFIER_PATH = (
    "models/classifier_model/trained_classifier_model.h5"
)
CLASSIFIER_DATA_ITOS_PATH = "data/train_valid_data_itos.pkl"
PROMETHEUS_ENABLE_EXPORTER_FLASK = False
