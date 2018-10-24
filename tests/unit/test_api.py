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

from inspire_classifier import serializers

from marshmallow.exceptions import ValidationError

import pytest


def test_output_serializer():
    output_serializer = serializers.ClassifierOutputSerializer()

    scores = {
        "prediction": "core",
        "scores": {
            "rejected": 0.1,
            "non_core": 0.2,
            "core": 0.7
        }
    }

    assert output_serializer.load(scores)


def test_output_serializer_raises_exception():
    output_serializer = serializers.ClassifierOutputSerializer()

    scores = {
        "prediction": "core",
        "scores": {
            "rejected": 0.1,
            "non_core": 0.2
        }
    }

    with pytest.raises(ValidationError):
        output_serializer.load(scores)


def test_output_serializer_does_not_accept_extra_fields():
    output_serializer = serializers.ClassifierOutputSerializer()

    scores = {
        "prediction": "core",
        "scores": {
            "rejected": 0.1,
            "non_core": 0.2,
            "core": 0.7,
            "score4": 0.0
        }
    }

    with pytest.raises(ValidationError):
        output_serializer.load(scores)


def test_output_accepts_only_certain_values_for_prediction():
    output_serializer = serializers.ClassifierOutputSerializer()

    scores = {
        "prediction": "non-rejected",
        "scores": {
            "rejected": 0.1,
            "non_core": 0.2,
            "core": 0.7
        }
    }

    with pytest.raises(ValidationError):
        output_serializer.load(scores)
