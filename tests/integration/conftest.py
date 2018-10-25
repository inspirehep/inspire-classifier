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

from inspire_classifier.app import create_app
from inspire_classifier.api import (
    create_directories,
    train
)
from inspire_classifier.domain.models import RNN_Learner
from inspire_classifier.utils import path_for
from mock import patch
from pathlib import Path
import pytest
import shutil


@pytest.fixture(autouse=True, scope='session')
def app():
    app = create_app()
    with app.app_context():
        app.config['CLASSIFIER_MAXIMUM_VOCABULARY_SIZE'] = 500
        app.config['CLASSIFIER_MINIMUM_WORD_FREQUENCY'] = 1
        app.config['CLASSIFIER_LANGUAGE_MODEL_CYCLE_LENGTH'] = 1
        app.config['CLASSIFIER_CLASSIFIER_CYCLE_LENGTH'] = 1
        app.config['CLASSIFIER_LANGUAGE_MODEL_BATCH_SIZE'] = 10
        app.config['CLASSIFIER_CLASSIFIER_BATCH_SIZE'] = 10
        app.config['CLASSIFIER_VALIDATION_DATA_FRACTION'] = 0.2
        yield app


# TODO: all fixtures using ``app`` must be replaced by ones that use ``isolated_app``.
@pytest.fixture()
def app_client(app):
    """Flask test client for the application.
    See: http://flask.pocoo.org/docs/0.12/testing/#keeping-the-context-around.
    """
    with app.test_client() as client:
        yield client


class Mock_RNN_Learner(RNN_Learner):
    """
    Mocks the fit method of the RNN_Learner.

    This is done to reduce the model training time during testing by making the fit run once (as opposed to 2 times and
    3 times for the LanguageModel and Classifier respectively). It stores the result of the first run and then returns
    the same result for the other times fit is run.
    """
    def fit(self, *args, **kwargs):
        try:
            return self._fit_result
        except AttributeError:
            self._fit_result = super().fit(*args, **kwargs)
            return self._fit_result


@pytest.fixture(scope='session')
@patch(
    'fastai.text.RNN_Learner', Mock_RNN_Learner
)
@patch(
    'inspire_classifier.domain.models.RNN_Learner', Mock_RNN_Learner
)
def trained_pipeline(app, tmpdir_factory):
    app.config['CLASSIFIER_BASE_PATH'] = str(tmpdir_factory)
    create_directories()
    shutil.copy(Path(__file__).parent / 'fixtures' / 'inspire_test_data.df', path_for('dataframe'))
    train()
