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

from click.exceptions import MissingParameter
from inspire_classifier.cli import (
    predict,
    train_classifier
)
import pytest


def test_classifier_predict_cli_with_classifier_base_path():
    input_arguments = predict.make_context('predict-coreness', args=['foo', 'bar', '-b', '.'])
    assert input_arguments.params['title'] == 'foo'
    assert input_arguments.params['abstract'] == 'bar'
    assert input_arguments.params['base_path'] == '.'


def test_classifier_predict_cli_without_classifier_base_path():
    input_arguments = predict.make_context('predict-coreness', args=['foo', 'bar'])
    assert input_arguments.params['title'] == 'foo'
    assert input_arguments.params['abstract'] == 'bar'
    assert input_arguments.params['base_path'] is None


def test_classifier_predict_cli_fails_without_title_and_abstract():
    with pytest.raises(MissingParameter):
        predict.make_context('predict-coreness', args=[])


def test_classifier_train_cli_correctly_parses_arguments():
    input_arguments = train_classifier.make_context('train', args=['-l', '15', '-c', '14', '-b', '.'])
    assert input_arguments.params['language_model_epochs'] == 15
    assert input_arguments.params['classifier_epochs'] == 14
    assert input_arguments.params['base_path'] == '.'


def test_classifier_train_cli_works_with_no_arguments():
    assert train_classifier.make_context('train', args=[])
