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

from __future__ import absolute_import, division, print_function

import json


def test_health_check(app_client):
    assert app_client.get('/api/health').status_code == 200


def test_classifier_accepts_only_post(app_client):
    assert app_client.post('/api/classifier', data=dict(title='foo bar', abstract='foobar foobar')).status_code == 200
    assert app_client.get('/api/classifier').status_code == 405


def test_classifier(app_client):
    response = app_client.post('/api/classifier', data=dict(title='foo bar', abstract='foobar foobar'))

    result = json.loads(response.data)

    expected = {
        "prediction": "core",
        "score1": 0.1,
        "score2": 0.2,
        "score3": 0.7,
    }

    assert response.status_code == 200
    assert expected == result


def test_classifier_serializes_input(app_client):
    assert app_client.post('/api/classifier', data=dict(title='foo bar')).status_code == 422
    assert app_client.post('/api/classifier', data=dict(abstract='foo bar')).status_code == 422


def test_classifier_accepts_extra_fields(app_client):
    assert app_client.post('/api/classifier', data=dict(title='foo bar', abstract='foo bar', author='foo')).status_code == 200
