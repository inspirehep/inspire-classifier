import logging

from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics

from inspire_classifier.app import create_app

app = create_app()

if app.config.get("PROMETHEUS_ENABLE_EXPORTER_FLASK"):
    logging.info("Starting prometheus metrics exporter")
    metrics = GunicornInternalPrometheusMetrics.for_app_factory()
    metrics.init_app(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
