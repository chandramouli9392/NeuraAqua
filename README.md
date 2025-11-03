# 💧 NeuraAqua — AI Guardian of Industrial Flow  

### 🧠 Powered by IBM watsonx Orchestrate | Agentic AI for Sustainable Water Infrastructure  

---

## 🌍 Overview  
**AquaNeuraX** is an **AI-driven orchestration system** that brings neural intelligence to **industrial water management**.  
It monitors pump data — such as vibration, motor current, and temperature — to predict failures, prevent water waste, and ensure sustainable industrial performance.  

The system uses **IBM watsonx Orchestrate** to automate data analysis, reporting, and maintenance recommendations through **agentic AI workflows**.  
Each agent acts autonomously but collaborates intelligently — turning reactive maintenance into **proactive sustainability**.  

---

## ⚙️ Core Features  

| Feature | Description |
|----------|-------------|
| 🧩 **Agentic AI** | Autonomous agents analyze, predict, and act using watsonx Orchestrate. |
| 💧 **Smart Pump Analysis** | Detects inefficiencies, classifies pump health (HEALTHY / WARNING / FAIL). |
| 📈 **Predictive Maintenance** | Identifies early risk trends to prevent costly downtime. |
| 🧾 **Automated Reporting** | Generates daily PDF summaries and sends email notifications. |
| ☁️ **IBM Cloud Integration** | Seamlessly integrates with Cloud Object Storage, Cloud Functions, and Cloudant DB. |
| 🔍 **Explainable Insights** | Provides transparent reasoning for each AI decision. |

---

## 🧠 Agent Architecture  

1. **Data Analyzer Agent**  
   - Processes CSV or real-time IoT data.  
   - Applies rules/ML to classify pump status.  

2. **Predictive Maintenance Agent**  
   - Detects early warnings and recommends preventive actions.  

3. **Report Agent**  
   - Generates formatted daily summaries in PDF.  

4. **Advisory Agent**  
   - Interacts with users conversationally via watsonx Orchestrate.  

---

## 🔗 Workflow with IBM watsonx Orchestrate  

1️⃣ Upload pump dataset (CSV or IoT feed).  
2️⃣ Orchestrate triggers **Data Analyzer Agent**.  
3️⃣ Analyzer outputs JSON results → sent to **Maintenance Agent**.  
4️⃣ **Report Agent** compiles insights into a PDF summary.  
5️⃣ **Advisory Agent** communicates results & recommendations to the user.  

---

## 🧩 Tech Stack  

- **IBM watsonx Orchestrate** 🧠  
- **IBM Cloud Object Storage (COS)** ☁️  
- **IBM Cloud Functions** ⚡  
- **IBM Cloudant / Db2** 💾  
- **Watson Studio (optional ML scoring)** 📊  
- **Python + Pandas** for local testing 🐍  

---

## 💼 Use Case Alignment  

**🌐 UN SDG 9:** Industry, Innovation & Infrastructure  
**💦 UN SDG 6:** Clean Water & Sanitation  

AquaNeuraX enhances infrastructure resilience by preventing breakdowns, conserving water, and ensuring sustainable industrial practices.  

---

## 🚀 How to Run (Hackathon Demo Steps)  

1️⃣ Open **IBM watsonx Orchestrate** dashboard.  
2️⃣ Import the AquaNeuraX agent configuration.  
3️⃣ Upload your sample dataset (`water_pump_health_data.csv`).  
4️⃣ Trigger workflow: `Run Daily Pump Analysis`.  
5️⃣ View generated report → check email or COS bucket for output.  
6️⃣ Ask the agent questions like:  
   - “Analyze today’s pump performance.”  
   - “Show pumps at risk of failure.”  
   - “Summarize energy efficiency trends.”  

---

## 🧩 Example Query & Response  

**Query:**  
> “AquaNeuraX, can you check the latest vibration trend for Pump ID P-301?”

**Response:**  
> ✅ Pump P-301 shows an upward vibration trend (3.1 → 5.4 mm/s).  
> ⚠️ Early imbalance detected — schedule lubrication within 48 hours.  
> 📊 Classification: *WARNING*  

---

## 🤖 Team  
- Built by innovators for the **IBM Tech Exchange 2025 Hackathon**  
- Category: *Industry, Innovation, and Infrastructure (SDG 9)*  

---

## 💬 Example Greetings  
> 💧 “Hello, I’m AquaNeuraX — the Guardian of Industrial Flow. Upload your pump data, and I’ll analyze, predict, and optimize for sustainable water management.”  

---

## 🏆 Recognition Goals  
This project demonstrates how **Agentic AI** + **IBM watsonx Orchestrate** can:  
- Reduce industrial water wastage 🌊  
- Automate maintenance with zero downtime ⚙️  
- Inspire scalable, sustainable infrastructure 🌍  

---

### 🪄 License  
© 2025 AquaNeuraX Team | Built for the IBM watsonx Orchestrate Hackathon 💧  

---
