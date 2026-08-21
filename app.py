import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# ---------------------------------------------------------
# 🚀 Flask App Initialization & Model Loading
# ---------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ANN.pkl")

print("⏳ Loading Artificial Neural Network model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("✅ Model loaded successfully!")

# ---------------------------------------------------------
# 🎨 Modern Responsive UI Template (HTML + CSS)
# ---------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 ANN Prediction Engine</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #22c55e;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: var(--bg); color: var(--text-main); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem 1rem; }
        .container { background-color: var(--card-bg); width: 100%; max-width: 650px; padding: 2.5rem; border-radius: 16px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); border: 1px solid var(--border); }
        .header { text-align: center; margin-bottom: 2rem; }
        .header h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }
        .header p { color: var(--text-muted); font-size: 0.95rem; }
        .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
        .input-group { display: flex; flex-direction: column; }
        label { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.4rem; }
        input { background-color: #0f172a; border: 1px solid var(--border); color: #fff; padding: 0.75rem; border-radius: 8px; font-size: 0.95rem; transition: border-color 0.2s; }
        input:focus { outline: none; border-color: var(--primary); }
        button { width: 100%; padding: 0.9rem; border: none; border-radius: 8px; background: linear-gradient(135deg, var(--primary), var(--primary-hover)); color: #fff; font-size: 1rem; font-weight: 600; cursor: pointer; transition: transform 0.1s ease, opacity 0.2s; }
        button:hover { opacity: 0.95; }
        button:active { transform: scale(0.98); }
        .result-card { margin-top: 1.5rem; padding: 1.25rem; border-radius: 8px; background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.2); text-align: center; display: none; }
        .result-card h3 { color: var(--accent); font-size: 1.1rem; margin-bottom: 0.3rem; }
        .result-card p { font-size: 0.9rem; color: var(--text-muted); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 ANN Model Inference</h1>
            <p>Enter the 10 numerical features to compute neural predictions</p>
        </div>
        <form id="predictionForm">
            <div class="grid">
                {% for i in range(1, 11) %}
                <div class="input-group">
                    <label for="f{{i}}">Feature {{i}} 📊</label>
                    <input type="number" step="any" id="f{{i}}" name="f{{i}}" placeholder="0.0" required>
                </div>
                {% endfor %}
            </div>
            <button type="submit">✨ Run Inference</button>
        </form>
        <div class="result-card" id="resultBox">
            <h3 id="resultTitle">Prediction Result</h3>
            <p id="resultText"></p>
        </div>
    </div>
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const features = [];
            for (let i = 1; i <= 10; i++) {
                features.push(parseFloat(document.getElementById(`f${i}`).value));
            }
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: features })
                });
                const data = await response.json();
                if (data.status === 'success') {
                    const resultBox = document.getElementById('resultBox');
                    document.getElementById('resultTitle').innerText = `🎯 Output Class: ${data.prediction}`;
                    document.getElementById('resultText').innerText = `Confidence / Probability: ${(data.probability * 100).toFixed(2)}%`;
                    resultBox.style.display = 'block';
                }
            } catch (err) {
                alert('⚠️ Prediction failed: ' + err.message);
            }
        });
    </script>
</body>
</html>
"""

# ---------------------------------------------------------
# 🌐 Routes & Endpoints
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Renders the frontend dashboard."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check route for Render monitoring."""
    return jsonify({"status": "healthy", "service": "ANN Inference Service"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Accepts JSON payload with 10 features and returns the prediction."""
    try:
        data = request.get_json(force=True)
        features = data.get("features", [])

        if len(features) != 10:
            return jsonify({
                "status": "error",
                "message": f"Expected 10 features, received {len(features)}"
            }), 400

        # Preprocess and infer
        input_data = np.array([features], dtype=np.float32)
        raw_output = model.predict(input_data, verbose=0)
        probability = float(raw_output[0][0])
        predicted_class = int(probability >= 0.5)

        return jsonify({
            "status": "success",
            "prediction": predicted_class,
            "probability": round(probability, 4)
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
