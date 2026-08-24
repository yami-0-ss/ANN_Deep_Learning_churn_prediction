import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# --- Model Loading ---
MODEL_PATH = "ANN.pkl"
model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- Combined HTML/CSS Template with Modern UI ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 ANN Prediction Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #022c22);
            background-size: 400% 400%;
            animation: gradientShift 14s ease infinite;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
            color: #f8fafc;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            width: 100%;
            max-width: 850px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: #94a3b8;
            font-size: 0.95rem;
            letter-spacing: 0.5px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #cbd5e1;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .input-group input:focus {
            border-color: #38bdf8;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
            background: rgba(15, 23, 42, 0.85);
        }

        .btn-submit {
            width: 100%;
            padding: 1rem;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #0284c7, #9333ea);
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 20px -5px rgba(147, 51, 234, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(147, 51, 234, 0.6);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(56, 189, 248, 0.3);
            text-align: center;
            animation: fadeIn 0.5s ease-out forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 0.4rem;
        }

        .result-score {
            font-size: 2rem;
            font-weight: 700;
            color: #38bdf8;
        }

        .result-status {
            margin-top: 0.5rem;
            font-size: 1.1rem;
            font-weight: 600;
        }

        .status-high { color: #4ade80; }
        .status-low { color: #f87171; }
        .error-msg { color: #fb7185; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Neural Network Engine</h1>
            <p>✨ Real-Time 10-Feature Binary Inference</p>
        </div>

        <form method="POST" action="/">
            <div class="form-grid">
                {% for i in range(1, 11) %}
                <div class="input-group">
                    <label>⚡ Feature {{ i }}</label>
                    <input type="number" step="any" name="f{{ i }}" placeholder="0.00" required 
                           value="{{ inputs['f' ~ i] if inputs else '' }}">
                </div>
                {% endfor %}
            </div>
            <button type="submit" class="btn-submit">🚀 Run Model Inference</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            <div class="result-title">Model Confidence Score</div>
            <div class="result-score">{{ "%.4f"|format(prediction) }}</div>
            <div class="result-status {{ 'status-high' if prediction >= 0.5 else 'status-low' }}">
                {% if prediction >= 0.5 %}
                    🌟 Classification: Positive Class (≥ 0.50)
                {% else %}
                    ❄️ Classification: Negative Class (&lt; 0.50)
                {% endif %}
            </div>
        </div>
        {% elif error %}
        <div class="result-box">
            <p class="error-msg">⚠️ {{ error }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    inputs = None

    if request.method == "POST":
        inputs = request.form
        try:
            # Extract 10 input values
            feature_values = [float(request.form.get(f"f{i}", 0)) for i in range(1, 11)]
            
            # Format input array for (batch_size, 10)
            input_tensor = np.array([feature_values], dtype=np.float32)

            # Inference
            if model is not None:
                pred = model.predict(input_tensor)
                prediction = float(pred[0][0])
            else:
                error = "Model is not loaded on server."
        except Exception as err:
            error = f"Inference Failed: {str(err)}"

    return render_template_string(
        HTML_TEMPLATE, 
        prediction=prediction, 
        error=error, 
        inputs=inputs
    )

if __name__ == "__main__":
    # AWS Elastic Beanstalk / EC2 production port default
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
