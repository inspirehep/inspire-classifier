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

import datetime

from flask import Flask, Response, jsonify
from marshmallow import fields
from webargs.flaskparser import use_args

from inspire_classifier import serializers
from inspire_classifier.api import initialize_classifier, predict_coreness


class JsonResponse(Response):
    """ By creating this Response class, we force the response to always be in json,
    getting rid of the jsonify function."""

    @classmethod
    def force_type(cls, rv, environ=None):
        if isinstance(rv, dict):
            rv = jsonify(rv)
        return super(JsonResponse, cls).force_type(rv, environ)


def create_app(instance_path=None):
    Flask.response_class = JsonResponse
    coreness_schema = serializers.ClassifierOutputSerializer()
    # TODO instance path should be removed... but needs changes in deployment file
    app = Flask(
        __name__,
        instance_relative_config=True,
        instance_path= instance_path if instance_path else
        "/opt/conda/var/inspire_classifier.app-instance",
    )
    app.config["CLASSIFIER_BASE_PATH"] = app.instance_path
    app.config.from_object("inspire_classifier.config")
    app.config.from_pyfile("classifier.cfg", silent=True)
    with app.app_context():
        classifier = initialize_classifier()

    @app.route("/api/health")
    def date():
        """Basic endpoint that returns the date, used to check if everything is up
        and working."""
        now = datetime.datetime.now()
        return jsonify(now)

    @app.route("/api/predict/coreness", methods=["POST"])
    @use_args(
        {"title": fields.Str(required=True), "abstract": fields.Str(required=True)},
        location="json",
    )
    def core_classifier(args):
        """Endpoint for the CORE classifier."""
        prediction = predict_coreness(classifier, args["title"], args["abstract"])
        response = coreness_schema.dump(prediction)
        return response

    return app

    @app.errorhandler(404)
    def page_not_found(e):
        return {"errors": [str(e)]}, 404
