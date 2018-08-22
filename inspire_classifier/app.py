import datetime

from flask import Flask, jsonify, request
from marshmallow.exceptions import ValidationError

from . import serializers
from .domain.models import CoreClassifier


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

    # TODO: make it always return JSON content in case of errors (like when it's a GET); it currently returns HTML.

    input_serializer = serializers.ClassifierInputSerializer()
    output_serializer = serializers.ClassifierOutputSerializer()

    # Parse the input data.
    try:
        data = input_serializer.load(request.get_json(force=True))
    except ValidationError as exc:
        # TODO: we do not want to fail when extra data are sent as input.
        return jsonify(exc.messages), 400

    # Run the domain model.
    classifier = CoreClassifier(data['title'], data['abstract'])
    classifier.classify()

    # Build the response.
    return jsonify(output_serializer.dump(classifier))
