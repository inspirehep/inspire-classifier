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

import datetime

from flask import Flask, jsonify, Response
from flask_apispec import use_kwargs, marshal_with, FlaskApiSpec
from inspire_classifier.api import predict_coreness
from marshmallow import fields

from . import serializers


class JsonResponse(Response):
    """"By creaitng this Response class, we force the response to always be in json, getting rid of the jsonify function."""

    @classmethod
    def force_type(cls, rv, environ=None):
        if isinstance(rv, dict):
            rv = jsonify(rv)
        return super(JsonResponse, cls).force_type(rv, environ)


def create_app():
    Flask.response_class = JsonResponse
    app = Flask(__name__)
    app.config['CLASSIFIER_BASE_PATH'] = app.instance_path
    app.config.from_object('inspire_classifier.config')
    docs = FlaskApiSpec(app)

    @app.route("/api/health")
    def date():
        """Basic endpoint that returns the date, used to check if everything is up and working."""
        now = datetime.datetime.now()
        return jsonify(now)

    docs.register(date)

    @app.route("/api/predict/coreness", methods=["POST"])
    @use_kwargs({'title': fields.Str(required=True), 'abstract': fields.Str(required=True)})
    @marshal_with(serializers.ClassifierOutputSerializer)
    def core_classifier(**kwargs):
        """Endpoint for the CORE classifier."""

        return predict_coreness(kwargs['title'], kwargs['abstract'])

    docs.register(core_classifier)

    return app

    @app.errorhandler(404)
    def page_not_found(e):
        return {
            "errors": [
                str(e)
            ]
        }, 404


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0')
