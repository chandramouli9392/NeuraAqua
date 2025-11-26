import streamlit as st
import numpy as np
import joblib
import json
import os
import google.generativeai as genai

# ---------------------------------------------------------
# LOAD MODEL + SCALER + META
# ---------------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = f"{MODEL_DIR}/pump_model.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
META_PATH = f"{MODEL_DIR}/feature_meta.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
meta = json.load(open(META_PATH))

label_map = meta["label_map"]
inv_label_map = {v: k for k, v in label_map.items()}

# ---------------------------------------------------------
# GEMINI AI HYPOTHESIS GENERATOR
# ---------------------------------------------------------
def generate_hypothesis(vibration, temperature, current, status, risk):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "⚠ Gemini API key not set — using fallback hypothesis.\n\n"
            "Likely Cause:\n"
            "• High vibration → bearing wear / misalignment\n"
            "• High temperature → friction / lubrication issue\n"
            "• High current → overload / electrical issue"
        )

    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        prompt = f"""
Generate a mechanical pump failure hypothesis.

Sensor Inputs:
• Vibration: {vibration} mm/s
• Temperature: {temperature} °C
• Current: {current} A

ML Output:
• Status: {status}
• Risk: {risk}

Your task:
1. Explain the MOST LIKELY root cause
2. Provide mechanical reasoning
3. Give recommended maintenance actions
4. Provide urgency level
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"⚠ Gemini Model Error: {str(e)}"


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="PumpGuard AI", page_icon="🛠", layout="wide")

st.title("🛠 PumpGuard AI — Intelligent Pump Health Analyzer")

st.markdown(
    """
PumpGuard AI predicts pump health, failure risk,  
and generates Gemini-powered diagnostic hypotheses.
"""
)

# INPUTS
col1, col2, col3 = st.columns(3)

vibration = col1.number_input("Vibration (mm/s)", min_value=0.0, step=0.1)
temperature = col2.number_input("Temperature (°C)", min_value=0.0, step=0.1)
current = col3.number_input("Motor Current (A)", min_value=0.0, step=0.1)

if st.button("🔍 Analyze Pump Health"):
    # Preprocess
    X = np.array([[vibration, temperature, current]])
    X_scaled = scaler.transform(X)

    # Prediction
    proba = model.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(proba))
    status = inv_label_map[pred_idx]

    risk_score = float(proba[2]) if len(proba) == 3 else float(max(proba))

    color = {"HEALTHY": "green", "WARNING": "orange", "CRITICAL": "red"}[status]

    st.markdown(
        f"## Pump Status: <span style='color:{color}'>{status}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"### Failure Risk Score: **{risk_score:.3f}**")

    # Gemini Hypothesis
    st.subheader("🤖 AI-Generated Hypothesis")
    hypothesis = generate_hypothesis(vibration, temperature, current, status, risk_score)
    st.write(hypothesis)

    # Maintenance logic
    st.subheader("🔧 Recommended Maintenance")
    recs = []

    if vibration > 6:
        recs.append("• Inspect bearings & alignment (high vibration).")
    if temperature > 70:
        recs.append("• Check lubrication & cooling system (overheating).")
    if current > 12:
        recs.append("• Inspect motor load or electrical faults (high current).")

    if not recs:
        recs.append("• No immediate issues detected — continue monitoring.")

    st.write("\n".join(recs))