from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, render_template, request

from webapp.services.data_service import DATA
from webapp.services.history_service import HISTORY
from webapp.services.prediction_service import PREDICTOR

logger = logging.getLogger(__name__)

def create_app() -> Flask:
    app = Flask(__name__, static_folder = "static", template_folder = "templates")

    logger.info("Loading data tables...")
    DATA.load()
    logger.info("Loading models...")
    PREDICTOR.load()
    logger.info("Webapp ready (data source: %s, best model: %s)", DATA.source, PREDICTOR.best_model_name)

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            best_model = PREDICTOR.best_model_name,
            best_metrics = PREDICTOR.best_metrics,
            data_source = DATA.source
        )

    @app.route("/api/options")
    def api_options():
        return jsonify(DATA.options())

    @app.route("/api/dashboard")
    def api_dashboard():
        return jsonify(DATA.dashboard_stats())

    @app.route("/api/compare")
    def api_compare():
        return jsonify({
            "best_model": PREDICTOR.best_model_name,
            "metrics": PREDICTOR.compare_metrics()
        })

    @app.route("/api/predict", methods = ["POST"])
    def api_predict():
        payload = request.get_json(force = True)
        try:
            result = PREDICTOR.predict(payload)
        except Exception as e:
            logger.exception("Prediction failed")
            return jsonify({"error": str(e)}), 400

        item = HISTORY.add(payload, result)
        result["history_id"] = item["id"]
        return jsonify(result)

    @app.route("/api/history")
    def api_history():
        return jsonify({
            "items": HISTORY.list(),
            "stats": HISTORY.stats()
        })

    @app.route("/api/history/<int:item_id>/actual", methods = ["POST"])
    def api_history_actual(item_id: int):
        payload = request.get_json(force = True)
        actual = float(payload["actual_delay_minutes"])
        item = HISTORY.update_actual(item_id, actual)
        if item is None:
            return jsonify({"error": "Item not found"}), 404
        return jsonify(item)

    return app

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )
    app = create_app()
    app.run(host = "0.0.0.0", port = 5000, debug = False)
