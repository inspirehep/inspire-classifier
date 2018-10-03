#!/bin/sh
exec gunicorn -b :5000 --access-logfile - --error-logfile - inspire_classifier.app:app