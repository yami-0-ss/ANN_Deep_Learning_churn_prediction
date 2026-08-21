import threading
import pickle
import time
import requests
import numpy as np
from flask import Flask, request, jsonify
import streamlit as st

# ==========================================
# 1. FLASK BACKEND SERVICE
# ==========================================
server = Flask(__name__)

# Load Keras ANN model from pickle
with open('ANN.pkl', 'rb') as f:
    model = pickle.load(f)

@server.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'online'}), 200

@server.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json(force=True)
        features = data.get('features', [])
        
        if len(features) != 10:
            return jsonify({'error': f'Expected 10 features, got {len(features)}'}), 400
        
        input_data = np.array(features, dtype=np.float32).reshape(1, 10)
        prob = float(model.predict(input_data, verbose=0)[0][0])
        
        return jsonify({
            'status': 'success',
            'probability': prob,
            'prediction': int(prob >= 0.5)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_flask():
    server.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

# Start Flask daemon thread if not already running
if not any(t.name == "FlaskServerThread" for t in threading.enumerate()):
    flask_thread = threading.Thread(target=run_flask, name="FlaskServerThread", daemon=True)
    flask_thread.start()
    time.sleep(0.5)

# ==========================================
# 2. STREAMLIT FRONTEND
# ==========================================
st.set_page_config(
    page_title="ANN Inference Portal",
    page_icon="⚡",
    layout="wide"
)

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
    }
    .metric-container {
        background: #1e293b;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Neural Network Deployment Engine")
st.caption("Flask REST API running on port 5000 + Streamlit Dashboard UI")
st.write("---")

with st.sidebar:
    st.header("⚙️ Model Architecture")
    st.markdown("""
    * **Input Features**: 10
    * **Hidden Layers**: Dense (8, ReLU) → Dense (7, ReLU)
    * **Output Layer**: Dense (1, Sigmoid)
    """)
    st.divider()
    threshold = st.slider("Classification Cutoff", 0.0, 1.0, 0.5, 0.05)

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Input Numerical Vectors")
    grid_cols = st.columns(2)
    inputs = []
    
    for i in range(10):
        with grid_cols[i % 2]:
            val = st.number_input(
                f"Feature {i+1}",
                value=0.0,
                step=0.1,
                format="%.4f",
                key=f"feat_{i}"
            )
            inputs.append(val)

with right_col:
    st.subheader("Inference Result")
    
    if st.button("Run Prediction"):
        try:
            # Send payload directly to the embedded Flask endpoint
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json={"features": inputs},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                prob = result["probability"]
                assigned_class = 1 if prob >= threshold else 0
                badge_color = "#22c55e" if assigned_class == 1 else "#ef4444"

                st.markdown(f"""
                    <div class="metric-container">
                        <span style="color: #94a3b8; font-size: 0.9rem;">CONFIDENCE SCORE</span>
                        <h1 style="color: #f8fafc; margin: 8px 0;">{prob * 100:.2f}%</h1>
                        <hr style="border-color: #334155; margin: 16px 0;">
                        <span style="color: #94a3b8; font-size: 0.9rem;">FINAL OUTPUT</span>
                        <h2 style="color: {badge_color}; margin: 8px 0;">Class {assigned_class}</h2>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as err:
            st.error(f"Failed to connect to Flask API: {err}")
