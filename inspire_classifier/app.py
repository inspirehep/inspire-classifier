from flask import Flask
from flask import jsonify, request

import datetime


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
    result = request.get_json(force=True)

    return jsonify(result)
