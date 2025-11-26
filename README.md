🛠⚡ PumpGuard AI — Industrial Pump Health Prediction System

🔥 Powered by Machine Learning • 📊 Predictive Maintenance • 💥 Built by Tenet Σ

PumpGuard AI is a smart ML-based system designed to analyze industrial pump health using only 3 core parameters:

💠 Vibration (mm/s)

🔥 Temperature (°C)

⚡ Motor Current (A)

It predicts whether a pump is:

🟢 HEALTHY

🟠 WARNING

🔴 CRITICAL

This system helps industries reduce downtime, detect risks early, and maintain operational safety — all using simple numerical inputs + ML intelligence.

🌟 ✨ Features (Sigma Edition)

🧠 ML-powered Pump Health Classification

📈 Failure Risk Score

📊 Vibration–Temperature–Current based prediction

🛠 Actionable maintenance recommendations

🎛 Clean & modern Streamlit UI

⚡ Lightweight & deployable to Streamlit Cloud

🔒 Safe — No external API dependence

🚀 Offline compatible (uses only your trained model)

📁 Project Structure (Σ Organized)
PumpGuard-AI/
│── app.py                # Streamlit interface
│── train_model.py        # ML training script
│── requirements.txt      # Dependencies
│── data/
│     └── pumphealth.csv  # Your dataset
│── model/
│     ├── pump_model.pkl
│     ├── scaler.pkl
│     └── feature_meta.json
│── README.md

🧠 How PumpGuard AI Works
1️⃣ Training the ML Model

Uses RandomForestClassifier to learn pump conditions from:

⚙️ vibration

🌡 temperature

🔌 current

🏷 label (HEALTHY/WARNING/FAIL)

Run the training:

python train_model.py --csv data/pumphealth.csv --out model


This creates:

model/
  pump_model.pkl
  scaler.pkl
  feature_meta.json

2️⃣ Running the Streamlit App

Start the UI:

streamlit run app.py


Enter your parameters:

Vibration

Temperature

Motor Current

Then PumpGuard AI outputs:

🟢🟠🔴 Pump Status

📈 Failure Risk Probability

🛠 Maintenance Suggestions

📦 Installation (Σ Simple)

Install required libraries:

pip install -r requirements.txt


Requirements:

streamlit
scikit-learn
pandas
numpy
joblib


(No external API needed ✔)
(No internet dependency ✔)

🎨 UI Highlights

⚡ Minimal & fast

🔢 Easy numeric inputs

🟩🟧🟥 Color-coded output

🛠 Clear maintenance advice

🎯 Industrial-ready

🚀 Deploy to Streamlit Cloud

Push your project folder to GitHub

Go to https://streamlit.io/cloud

Choose your repo

Click Deploy

Boom — PumpGuard AI goes live. ⚡🔥

🧪 Model Training Script Summary (train_model.py)

Loads CSV

Encodes labels (HEALTHY/WARNING/FAIL)

Scales features

Trains RandomForest

Saves model + scaler + metadata

Simple, clean, fast. ⚙️

🏆 Why PumpGuard AI?

🔮 Predict pump failures earlier

🛠 Reduce repair cost

⚙️ Improve reliability

🧠 Use AI for smart maintenance

💸 Zero API cost

🟢 Works even offline

👨‍💻 Developer (Σ Authority Mode)

Built with precision and intelligence by Tenet Σ
ML • Data Science • AI Systems • Industrial Automation
