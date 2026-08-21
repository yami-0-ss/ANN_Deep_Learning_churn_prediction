"""
🚀 Production Flask API for Keras/ANN Model Deployment
📦 Serves single and batch predictions for a 10-feature input ANN
"""

import os
import pickle
import logging
import numpy as np
from flask import Flask, request, jsonify

# ---------------------------------------------------------
# 🛠️ Configuration & Logging Setup
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] ⚡ %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "ANN.pkl")
EXPECTED_FEATURES = 10
DECISION_THRESHOLD = 0.5

# ---------------------------------------------------------
# 🧠 Model Loader
# ---------------------------------------------------------
model = None

def load_prediction_model():
    """Loads the serialized Keras model from disk."""
    global model
    try:
        logger.info(f"🔄 Loading model from `{MODEL_PATH}`...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("✅ ANN Model successfully loaded and ready for inference!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise e

# Load model at startup
load_prediction_model()

# ---------------------------------------------------------
# 🌐 API Endpoints
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def health_check():
    """📡 AWS ALB / ECS / EB Health Check Endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "ANN Inference Engine",
        "expected_features": EXPECTED_FEATURES,
        "model_loaded": model is not None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    🎯 Main Prediction Endpoint
    
    Accepts JSON payloads:
    1. Single sample:  {"features": [1.0, 2.0, ..., 10.0]}
    2. Batch samples:  {"features": [[1.0, ..., 10.0], [2.0, ..., 10.0]]}
    """
    if not request.is_json:
        return jsonify({"error": "Invalid Content-Type. Expected `application/json`."}), 415

    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({
            "error": "Missing `features` key in payload.",
            "example": {"features": [0.0] * EXPECTED_FEATURES}
        }), 400

    try:
        raw_features = data["features"]
        input_array = np.array(raw_features, dtype=np.float32)

        # Reshape 1D inputs (single sample) to 2D (batch_size=1, features=10)
        if input_array.ndim == 1:
            input_array = np.expand_dims(input_array, axis=0)

        # Validate feature vector shape
        if input_array.ndim != 2 or input_array.shape[1] != EXPECTED_FEATURES:
            return jsonify({
                "error": f"Invalid feature dimensions. Expected shape (N, {EXPECTED_FEATURES}), received {input_array.shape}."
            }), 422

        # ⚡ Run Model Inference
        raw_predictions = model.predict(input_array, verbose=0)
        probabilities = raw_predictions.flatten().tolist()
        predicted_classes = [int(p >= DECISION_THRESHOLD) for p in probabilities]

        response = {
            "status": "success",
            "samples_processed": len(probabilities),
            "probabilities": probabilities,
            "predictions": predicted_classes
        }

        # Unpack single-sample responses for cleaner output
        if len(probabilities) == 1:
            response["probability"] = probabilities[0]
            response["prediction"] = predicted_classes[0]

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"🔥 Prediction failed: {str(e)}")
        return jsonify({"error": "Internal inference error", "details": str(e)}), 500


# ---------------------------------------------------------
# 🚀 Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
