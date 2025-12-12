# app.py - PumpGuard AI + Graphical Dashboard
import os
import json
import joblib
import time
import io
import csv
import numpy as np
import pandas as pd
import streamlit as st

# matplotlib for plotting
import matplotlib.pyplot as plt

# -------------------------
# Load ML Model + Scaler
# -------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "pump_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "feature_meta.json")

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(META_PATH)):
    st.error(
        "❌ Model files missing. Please run train_model.py and ensure /model contains: "
        "pump_model.pkl, scaler.pkl, feature_meta.json"
    )
    st.stop()

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
meta = json.load(open(META_PATH, "r"))
label_map = meta["label_map"]
inv_label_map = {v: k for k, v in label_map.items()}

# -------------------------
# Gemini Hypothesis Generator (env var only)
# -------------------------
def generate_hypothesis(vibration, temperature, current, status, risk):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "⚠ Gemini API key missing. Set GEMINI_API_KEY environment variable to enable AI hypotheses.\n\n"
            "Fallback hypothesis:\n"
            "- High vibration -> possible bearing wear / misalignment\n"
            "- High temperature -> lubrication or cooling issue\n"
            "- High current -> motor overload or electrical fault"
        )

    prompt = f"""
You are an expert mechanical engineer diagnosing pump faults.
Inputs:
- Vibration (mm/s): {vibration}
- Temperature (°C): {temperature}
- Motor Current (A): {current}
- ML Status: {status}
- Failure Risk: {risk:.3f}

Provide:
1) Most likely root cause
2) Mechanical reasoning
3) 3 recommended maintenance steps
4) Urgency (immediate/within 24h/routine)
"""
    # use new SDK style (most common), fallback to simple error text
    try:
        from google import generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        # different SDKs expose text differently
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            # candidate structure may vary
            try:
                return response.candidates[0].content.parts[0].text
            except Exception:
                return str(response)
        return str(response)
    except Exception as e:
        # If Gemini fails, return a helpful fallback message
        return f"⚠ Gemini Error: {e}\n\nFallback hypothesis:\n- Check bearings, alignment, lubrication, cooling, and motor load."

# -------------------------
# Session state: history
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # each item: dict with vib,temp,current,status,risk,timestamp,hypothesis

def add_history(vib, temp, curr, status, risk, hypothesis):
    st.session_state.history.append({
        "timestamp": int(time.time()),
        "vibration": float(vib),
        "temperature": float(temp),
        "current": float(curr),
        "status": status,
        "risk": float(risk),
        "hypothesis": hypothesis
    })

def clear_history():
    st.session_state.history = []

def history_to_df():
    if not st.session_state.history:
        return pd.DataFrame(columns=["timestamp","vibration","temperature","current","status","risk","hypothesis"])
    df = pd.DataFrame(st.session_state.history)
    df["time_readable"] = pd.to_datetime(df["timestamp"], unit="s")
    return df[["time_readable","vibration","temperature","current","status","risk","hypothesis"]]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PumpGuard AI", page_icon="🛠", layout="wide")
st.title("🛠 PumpGuard AI — Pump Health + Dashboard")

st.markdown(
    "Enter pump readings below. Use the dashboard to visualize risk vs vibration, track history, and download results."
)

# Main input area
with st.container():
    col1, col2, col3, col4 = st.columns([2,2,2,1])
    vibration = col1.number_input("Vibration (mm/s)", min_value=0.0, step=0.1, value=3.0, key="vib_input")
    temperature = col2.number_input("Temperature (°C)", min_value=0.0, step=0.1, value=35.0, key="temp_input")
    current = col3.number_input("Motor Current (A)", min_value=0.0, step=0.1, value=6.0, key="curr_input")
    analyze_btn = col4.button("🔍 Analyze")

if analyze_btn:
    # ML Prediction
    X = np.array([[vibration, temperature, current]])
    try:
        Xs = scaler.transform(X)
        proba = clf.predict_proba(Xs)[0]
        idx = int(np.argmax(proba))
        status = inv_label_map.get(idx, "UNKNOWN")
        # choose risk as probability of CRITICAL if mapping contains that, else max prob
        risk = float(proba[2]) if len(proba) > 2 else float(max(proba))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Gemini hypothesis (env var only)
    hypothesis = generate_hypothesis(vibration, temperature, current, status, risk)

    # display results
    color = {"HEALTHY":"green","WARNING":"orange","CRITICAL":"red"}.get(status, "black")
    st.markdown(f"## Pump Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    st.markdown(f"### Failure Risk Score: **{risk:.3f}**")
    st.subheader("🤖 AI-Generated Hypothesis")
    st.write(hypothesis)

    # Maintenance suggestions
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

    # add to history
    add_history(vibration, temperature, current, status, risk, hypothesis)

# -------------------------
# Sidebar controls & Data
# -------------------------
with st.sidebar:
    st.header("Dashboard Controls")
    st.write("History entries:", len(st.session_state.history))
    if st.button("Clear History"):
        clear_history()
        st.success("History cleared.")
    if st.session_state.history:
        df = history_to_df()
        # download link
        csv_buf = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download history CSV", data=csv_buf, file_name="pump_history.csv", mime="text/csv")
    st.markdown("---")
    st.markdown("Tips:\n• Train the model locally before running the app.\n• Set GEMINI_API_KEY env var to enable AI hypotheses.")

# -------------------------
# Dashboard visuals
# -------------------------
st.header("📊 Graphical Analysis Dashboard")

df = history_to_df()

if df.empty:
    st.info("No history yet — perform an analysis to populate the dashboard.")
else:
    # Scatter: Vibration vs Risk
    st.subheader("Vibration vs Failure Risk (scatter)")
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["vibration"], df["risk"])
    ax1.set_xlabel("Vibration (mm/s)")
    ax1.set_ylabel("Failure Risk (0-1)")
    ax1.set_title("Vibration vs Risk")
    ax1.grid(True)
    st.pyplot(fig1)

    # Time-series: Risk over time
    st.subheader("Risk over Time")
    fig2, ax2 = plt.subplots()
    ax2.plot(df["time_readable"], df["risk"], marker="o")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Failure Risk (0-1)")
    ax2.set_title("Risk over Time")
    ax2.grid(True)
    fig2.autofmt_xdate()
    st.pyplot(fig2)

    # Histogram: Risk distribution
    st.subheader("Risk Distribution")
    fig3, ax3 = plt.subplots()
    ax3.hist(df["risk"], bins=10)
    ax3.set_xlabel("Failure Risk")
    ax3.set_ylabel("Count")
    ax3.set_title("Risk Distribution")
    ax3.grid(True)
    st.pyplot(fig3)

    # Optional: show raw table with ability to filter by status
    st.subheader("History Table")
    status_filter = st.multiselect("Filter by status", options=df["status"].unique().tolist(), default=df["status"].unique().tolist())
    filtered = df[df["status"].isin(status_filter)]
    st.dataframe(filtered.sort_values("time_readable", ascending=False), use_container_width=True)
