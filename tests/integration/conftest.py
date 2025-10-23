# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2014-2017 CERN.
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

import os
import shutil
from pathlib import Path

import pytest
from fastai.text.all import Learner
from mock import patch

from inspire_classifier.core.api import create_directories, train
from inspire_classifier.core.utils import get_data_path


class Mock_Learner(Learner):
    """
    Mocks the fit method of the Learner.

    This is done to reduce the model training time during testing by making the fit
    run once (as opposed to 2 times and 3 times for the LanguageModel and Classifier
    respectively). It stores the result of the first run and then returns the same
    result for the other times fit is run.
    """

    def fit(self, *args, **kwargs):
        try:
            return self._fit_result
        except AttributeError:
            self._fit_result = super().fit(*args, **kwargs)
            return self._fit_result


@pytest.fixture(scope="session")
@patch("fastai.text.learner.text_classifier_learner", Mock_Learner)
def _trained_pipeline(tmp_path_factory):
    create_directories(os.path.join(os.getcwd(), "inspire_classifier_testing"))
    shutil.copy(
        Path(__file__).parent / "fixtures" / "inspire_test_data.df",
        get_data_path(
            os.path.join(os.getcwd(), "inspire_classifier_testing"),
            "train_valid_data.df",
        ),
    )
    train(
        base_path=os.path.join(os.getcwd(), "inspire_classifier_testing"),
        cuda_device_id=-1,
        language_model_cycle_length=1,
        classifier_cycle_length=1,
        maximum_vocabulary_size=500,
        minimum_word_frequency=1,
        language_model_batch_size=10,
        classifier_batch_size=10,
        val_fraction=0.2,
    )


@pytest.fixture(scope="session", autouse=True)
def _after_all_tests():
    yield
    shutil.rmtree(os.path.join(os.getcwd(), "inspire_classifier_testing"))
