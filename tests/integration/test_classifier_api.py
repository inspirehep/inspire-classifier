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

import pickle
from math import isclose

import pandas as pd
import pytest

from inspire_classifier.api import initialize_classifier, predict_coreness
from inspire_classifier.utils import path_for

TEST_TITLE = "Pre-images of extreme points of the numerical range, and applications"
TEST_ABSTRACT =\
    ("We extend the pre-image representation of exposed points of the numerical range "
     "of a matrix to all extreme points. With that we characterize extreme points which"
     " are multiply generated, having at least two linearly independent pre-images,"
     " as the extreme points which are Hausdorff limits of flat boundary portions on"
     " numerical ranges of a sequence converging to the given matrix."
     " These studies address the inverse numerical range map and the maximum-entropy "
     "inference map which are continuous functions on the numerical range except "
     "possibly at certain multiply generated extreme points. This work also allows us"
     " to describe closures of subsets of 3-by-3 matrices having the same shape of the"
     " numerical range.")


@pytest.mark.usefixtures("_trained_pipeline")
def test_create_directories():
    assert path_for("classifier_model").exists()


@pytest.mark.usefixtures("_trained_pipeline")
def test_preprocess_and_save_data(app):
    dataframe = pd.read_pickle(path_for("dataframe"))

    training_valid__csv = pd.read_csv(path_for("train_valid_data"))
    training_csv = training_valid__csv[~training_valid__csv["is_valid"]]
    validation_csv = training_valid__csv[training_valid__csv["is_valid"]]

    assert isclose(
        len(training_csv),
        len(dataframe) * (1 - app.config["CLASSIFIER_VALIDATION_DATA_FRACTION"]),
        abs_tol=1,
    )
    assert isclose(
        len(validation_csv),
        len(dataframe) * app.config["CLASSIFIER_VALIDATION_DATA_FRACTION"],
        abs_tol=1,
    )

@pytest.mark.usefixtures("_trained_pipeline")
def test_vocab(app):
    with open(path_for("data_itos"), "rb") as file:
        data_itos = pickle.load(file)
    # For performance when using mixed precision, the vocabulary is always made of
    # size a multiple of 8, potentially by adding xxfake tokens.
    adjusted_max_vocab = (
        app.config["CLASSIFIER_MAXIMUM_VOCABULARY_SIZE"]
        + 8
        - app.config["CLASSIFIER_MAXIMUM_VOCABULARY_SIZE"] % 8
    )
    assert len(data_itos) == adjusted_max_vocab


@pytest.mark.usefixtures("_trained_pipeline")
def test_save_language_model():
    assert path_for("finetuned_language_model_encoder").exists()


@pytest.mark.usefixtures("_trained_pipeline")
def test_train_and_save_classifier():
    assert path_for("trained_classifier").exists()


@pytest.mark.usefixtures("_trained_pipeline")
def test_predict_coreness():
    assert path_for("data_itos").exists()
    assert path_for("trained_classifier").exists()
    classifier = initialize_classifier()
    output_dict = predict_coreness(classifier, TEST_TITLE, TEST_ABSTRACT)

    assert set(output_dict.keys()) == {"prediction", "scores"}
    assert output_dict["prediction"] in {"rejected", "non_core", "core"}
    assert set(output_dict["scores"].keys()) == {"rejected", "non_core", "core"}
    assert isclose(
        output_dict["scores"]["rejected"]
        + output_dict["scores"]["non_core"]
        + output_dict["scores"]["core"],
        1.0,
        abs_tol=1e-2,
    )
