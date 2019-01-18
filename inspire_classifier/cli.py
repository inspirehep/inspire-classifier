# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2014-2019 CERN.
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

import click
import click_spinner
from flask import current_app
from flask.cli import FlaskGroup, with_appcontext
from inspire_classifier.api import (
    train,
    predict_coreness
)
from inspire_classifier.app import create_app


@click.group(cls=FlaskGroup, create_app=create_app)
def inspire_classifier():
    "INSPIRE Classifier commands"


@inspire_classifier.command('predict-coreness')
@with_appcontext
@click.argument('title', type=str, required=True, nargs=1)
@click.argument('abstract', type=str, required=True, nargs=1)
@click.option('-b', '--base-path', type=click.Path(exists=True), required=False, nargs=1)
def predict(title, abstract, base_path):
    with click_spinner.spinner():
        with current_app.app_context():
            if base_path:
                current_app.config['CLASSIFIER_BASE_PATH'] = base_path
            click.echo(predict_coreness(title, abstract))


@inspire_classifier.command('train')
@with_appcontext
@click.option('-l', '--language-model-epochs', type=int, required=False, nargs=1)
@click.option('-c', '--classifier-epochs', type=int, required=False, nargs=1)
@click.option('-b', '--base-path', type=click.Path(exists=True), required=False, nargs=1)
def train_classifier(language_model_epochs, classifier_epochs, base_path):
    with click_spinner.spinner():
        with current_app.app_context():
            if language_model_epochs:
                current_app.config['CLASSIFIER_LANGUAGE_MODEL_CYCLE_LENGTH'] = language_model_epochs
            if classifier_epochs:
                current_app.config['CLASSIFIER_CLASSIFIER_CYCLE_LENGTH'] = classifier_epochs
            if base_path:
                current_app.config['CLASSIFIER_BASE_PATH'] = base_path
            train()
