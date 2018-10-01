import datetime

from flask import Flask, jsonify, request, Response
from marshmallow.exceptions import ValidationError

import .serializers as serializers
from .domain import CoreClassifier


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
        data = input_serializer.load(request.get_json(force=True))[0]
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
