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

from marshmallow.exceptions import ValidationError

import pytest


def test_input_serializer():
    input_serializer = serializers.ClassifierInputSerializer()

    request = {
        "title": "Alice in Wonderland",
        "abstract": "The reader is conveyed to Wonderland, a world that has no apparent connection with reality...",
    }

    assert input_serializer.load(request)


def test_output_serializer():
    output_serializer = serializers.ClassifierOutputSerializer()

    scores = {
        "score1": 0.1,
        "score2": 0.2,
        "score3": 0.7,
    }

    assert output_serializer.load(scores)


def test_input_serializer_accepts_extra_fields():
    input_serializer = serializers.ClassifierInputSerializer()

    request = {
        "title": "Alice in Wonderland",
        "abstract": "The reader is conveyed to Wonderland, a world that has no apparent connection with reality...",
        "author": "Lewis Carroll",
    }

    assert input_serializer.load(request)


def test_input_serializer_raises_exception():
    input_serializer = serializers.ClassifierInputSerializer()

    request = {
        "title": "Alice in Wonderland",
    }

    with pytest.raises(ValidationError):
        input_serializer.load(request)


def test_output_serializer_raises_exception():
    output_serializer = serializers.ClassifierOutputSerializer()

    scores = {
        "score1": 0.1,
        "score2": 0.2,
    }

    with pytest.raises(ValidationError):
        output_serializer.load(scores)


def test_output_serializer_doe_not_accept_extra_fields():
    output_serializer = serializers.ClassifierOutputSerializer()

    scores = {
        "score1": 0.1,
        "score2": 0.2,
        "score3": 0.7,
        "score4": 0.0,
    }

    with pytest.raises(ValidationError):
        output_serializer.load(scores)
