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

import datetime

from flask import Flask, jsonify, request, Response
from marshmallow.exceptions import ValidationError

from . import serializers
from .domain.models import CoreClassifier


class JsonResponse(Response):
    """"By creaitng this Response class, we force the response to always be in json, getting rid of the jsonify function."""

    @classmethod
    def force_type(cls, rv, environ=None):
        if isinstance(rv, dict):
            rv = jsonify(rv)
        return super(JsonResponse, cls).force_type(rv, environ)


classifier = CoreClassifier()
Flask.response_class = JsonResponse
app = Flask(__name__)


@app.route("/api/health")
def date():
    """Basic endpoint that returns the date, used to check if everything is up and working"""
    now = datetime.datetime.now()
    return jsonify(now)


@app.route("/api/classifier", methods=["POST"])
def core_classifier():
    """Endpoint for the CORE classifier.

    Accepts only POST requests, as we have to send data (title and abstract) to the classifier.

    Returns an array with three float values that correspond to the probability of the record being Rejected, Non-Core and Core."""

    input_serializer = serializers.ClassifierInputSerializer()
    output_serializer = serializers.ClassifierOutputSerializer()

    try:
        data = input_serializer.load(request.get_json(force=True))
    except ValidationError as exc:
        return {
            "errors": [
                exc.messages
            ]
        }, 400

    classifier.predict(data['title'], data['abstract'])

    return output_serializer.dump(classifier)


@app.errorhandler(404)
def page_not_found(e):
    return {
        "errors": [
            str(e)
        ]
    }, 404


if __name__ == '__main__':
    app.run(host='0.0.0.0')
