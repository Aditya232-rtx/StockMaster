"""Flask application serving a lightweight UI to exercise the ML models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from test_model import ModelTester, create_sample_data

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

MODEL_TESTER: ModelTester | None = None
MODEL_LOAD_ERROR: str | None = None


def get_tester() -> ModelTester:
    """Return a lazily instantiated ModelTester instance."""
    global MODEL_TESTER, MODEL_LOAD_ERROR

    if MODEL_TESTER is None and MODEL_LOAD_ERROR is None:
        try:
            MODEL_TESTER = ModelTester()
        except Exception as exc:  # pragma: no cover - defensive guard
            MODEL_LOAD_ERROR = str(exc)
    if MODEL_TESTER is None:
        raise RuntimeError(MODEL_LOAD_ERROR or "ModelTester failed to initialise.")
    return MODEL_TESTER


def coerce_numeric_features(features: Dict[str, Any]) -> Dict[str, float]:
    """Ensure all feature values can be parsed as floats."""
    if not isinstance(features, dict):
        raise ValueError("Features payload must be a dictionary.")

    coerced: Dict[str, float] = {}
    for key, value in features.items():
        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise ValueError(f"Feature '{key}' is required.")
        if isinstance(value, (int, float)):
            coerced[key] = float(value)
            continue
        try:
            coerced[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{key}' must be numeric.") from exc
    return coerced


@app.get("/")
def serve_index():
    """Serve the micro frontend entry point."""
    index_path = Path(app.static_folder) / "index.html"
    if not index_path.exists():
        return ("Frontend not built yet. Please add 'frontend/index.html'.", 404)
    return send_from_directory(app.static_folder, "index.html")


@app.get("/<path:asset>")
def serve_static(asset: str):
    """Serve other static assets (JS/CSS)."""
    asset_path = Path(app.static_folder) / asset
    if not asset_path.exists() or not asset_path.is_file():
        return ("Asset not found", 404)
    return send_from_directory(app.static_folder, asset)


@app.get("/api/sample-data")
def api_sample_data():
    """Provide sample product and vendor features for the UI."""
    try:
        product_features, vendor_features = create_sample_data()
        return jsonify(
            {
                "status": "success",
                "product_features": product_features,
                "vendor_features": vendor_features,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.post("/api/predict/demand")
def api_predict_demand():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        tester = get_tester()
        product_features = coerce_numeric_features(payload.get("product_features", {}))
        prediction = tester.predict_demand(product_features)
        return jsonify({"status": "success", "prediction": float(prediction)})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.post("/api/predict/vendor-price")
def api_predict_vendor_price():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        tester = get_tester()
        vendor_features = coerce_numeric_features(payload.get("vendor_features", {}))
        prediction = tester.predict_price(vendor_features)
        return jsonify({"status": "success", "prediction": float(prediction)})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.post("/api/recommendation")
def api_recommendation():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        tester = get_tester()
        product_id = payload.get("product_id", "").strip() or "UNKNOWN"
        product_features = coerce_numeric_features(payload.get("product_features", {}))
        vendor_features = coerce_numeric_features(payload.get("vendor_features", {}))
        recommendation = tester.get_inventory_recommendation(
            product_id=product_id,
            product_features=product_features,
            vendor_features=vendor_features,
        )
        return jsonify({"status": "success", "recommendation": recommendation})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.get("/api/health")
def api_health():
    """Simple health endpoint for monitoring."""
    if MODEL_LOAD_ERROR:
        return jsonify({"status": "error", "message": MODEL_LOAD_ERROR}), 500
    try:
        get_tester()
        return jsonify({"status": "ok"})
    except Exception as exc:  # pragma: no cover - defensive guard
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
