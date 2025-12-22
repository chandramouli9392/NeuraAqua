import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# LOAD MODEL + SCALER + META
# ---------------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = f"{MODEL_DIR}/pump_model.pkl"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
META_PATH = f"{MODEL_DIR}/feature_meta.json"

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(META_PATH)):
    st.error("❌ Model files missing. Please run train_model.py and ensure /model contains: pump_model.pkl, scaler.pkl, feature_meta.json")
    st.stop()

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
            "⚠ Gemini API key missing.\n"
            "Fallback hypothesis:\n"
            "- High vibration → misalignment or bearing wear\n"
            "- High temperature → lubrication or cooling issue\n"
            "- High current → overload or electrical fault"
        )

    prompt = f"""
You are an expert mechanical engineer diagnosing pump faults.

Inputs:
• Vibration: {vibration} mm/s
• Temperature: {temperature} °C
• Motor Current: {current} A
• Status: {status}
• Failure Risk: {risk:.3f}

Provide:
1) Most likely root cause
2) Mechanical reasoning
3) 3 recommended maintenance steps
4) Urgency level
"""

    try:
      import google.generativeai as genai

        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel("gemini-pro")
        response = gmodel.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            return response.text

        if hasattr(response, "candidates") and response.candidates:
            try:
                return response.candidates[0].content.parts[0].text
            except:
                return str(response)

        return str(response)

    except Exception as e:
        return f"⚠ Gemini Error: {e}\n\nFallback hypothesis:\n- Inspect bearings, lubrication, cooling, and electrical load."

# ---------------------------------------------------------
# HISTORY STATE
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

def add_history(v, t, c, status, risk, hypothesis):
    st.session_state.history.append({
        "timestamp": int(time.time()),
        "vibration": float(v),
        "temperature": float(t),
        "current": float(c),
        "status": status,
        "risk": float(risk),
        "hypothesis": hypothesis
    })

def history_df():
    if not st.session_state.history:
        return pd.DataFrame(columns=["timestamp","vibration","temperature","current","status","risk","hypothesis"])
    df = pd.DataFrame(st.session_state.history)
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")
    return df

def clear_history():
    st.session_state.history = []

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="PumpGuard AI", page_icon="🛠", layout="wide")

st.title("🛠 PumpGuard AI — Intelligent Pump Health Analyzer")
st.markdown("Predict pump health, failure risk, and view visual dashboards.")

# INPUT AREA
col1, col2, col3, col4 = st.columns([2,2,2,1])
vibration = col1.number_input("Vibration (mm/s)", min_value=0.0, step=0.1, value=3.0)
temperature = col2.number_input("Temperature (°C)", min_value=0.0, step=0.1, value=35.0)
current = col3.number_input("Motor Current (A)", min_value=0.0, step=0.1, value=6.0)
analyze = col4.button("🔍 Analyze")

# ---------------------------------------------------------
# ANALYSIS BLOCK
# ---------------------------------------------------------
if analyze:
    X = np.array([[vibration, temperature, current]])
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0]
    idx = int(np.argmax(proba))
    status = inv_label_map[idx]
    risk = float(proba[2]) if len(proba) > 2 else float(max(proba))

    color = {"HEALTHY":"green","WARNING":"orange","CRITICAL":"red"}.get(status, "black")

    st.markdown(f"## Pump Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    st.markdown(f"### Failure Risk Score: **{risk:.3f}**")

    # Hypothesis
    st.subheader("🤖 AI-Generated Hypothesis")
    hypothesis = generate_hypothesis(vibration, temperature, current, status, risk)
    st.write(hypothesis)

    # Maintenance
    st.subheader("🔧 Maintenance Recommendations")
    recs = []
    if vibration > 6: recs.append("• Inspect bearings & alignment (high vibration).")
    if temperature > 70: recs.append("• Check lubrication & cooling system (overheating).")
    if current > 12: recs.append("• Inspect motor load or electrical faults (high current).")
    if not recs: recs.append("• No immediate issues detected — continue monitoring.")
    st.write("\n".join(recs))

    # Save history
    add_history(vibration, temperature, current, status, risk, hypothesis)

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
st.header("📊 Graphical Analysis Dashboard")

df = history_df()
if df.empty:
    st.info("Perform at least one analysis to populate the dashboard.")
else:
    import altair as alt

    # Scatter Chart
    st.subheader("Vibration vs Risk")
    scatter = (
        alt.Chart(df)
        .mark_circle(size=80)
        .encode(
            x="vibration",
            y="risk",
            color="status",
            tooltip=["time", "vibration", "temperature", "current", "status", "risk"]
        )
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)

    # Time Series
    st.subheader("Risk Over Time")
    line = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="time:T",
            y="risk:Q",
            color="status:N",
            tooltip=["time", "risk", "status"]
        )
        .interactive()
    )
    st.altair_chart(line, use_container_width=True)

    # Histogram
    st.subheader("Risk Distribution")
    hist = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X("risk:Q", bin=alt.Bin(maxbins=12)),
            y="count()",
            color="status:N",
        )
    )
    st.altair_chart(hist, use_container_width=True)

    # History Table
    st.subheader("History Table")
    status_filter = st.multiselect("Filter by status", df["status"].unique(), default=df["status"].unique())
    st.dataframe(df[df["status"].isin(status_filter)], use_container_width=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Clear History"):
        clear_history()
        st.success("History cleared.")

    if not df.empty:
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "pumpguard_history.csv",
            "text/csv"
        )

    st.markdown("— PumpGuard AI by Tenet Σ")


