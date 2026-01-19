# 🛡️ AI-Powered Network Intrusion Detection System (AI-NIDS)

An advanced cybersecurity dashboard that combines **Machine Learning** and **Generative AI** to detect, analyze, and explain network intrusions in a transparent and user-friendly way.

This project uses a **Random Forest Classifier** for intrusion detection and **Groq Llama 3** for AI-generated, human-readable security explanations simulating the workflow of a modern Security Operations Center (SOC).

---

## 🚀 Features

- **Multi-Class Intrusion Detection**
  - Detects **Benign Traffic**, **DDoS Attacks**, and **Port Scans**

- **SOC Control Panel**
  - Dynamically adjust Random Forest parameters:
    - Number of Trees
    - Train/Test Split Ratio

- **Explainable AI (XAI)**
  - Feature Importance visualization
  - Transparent view of what the ML model learns

- **AI Security Analyst**
  - Integrated **Groq Cloud API (Llama 3.3-70B)**
  - Generates technical, SOC-style explanations for detected threats

- **Incident Audit Trail**
  - Persistent logging of all simulated network traffic
  - **Export logs to CSV** for reporting and analysis

- **Realistic Accuracy Simulation**
  - Introduces controlled statistical noise
  - Achieves realistic accuracy levels (**94–97%**) instead of perfect results

---

## 🧠 Technology Stack

| Component | Technology |
|---------|------------|
| Frontend | Streamlit |
| Machine Learning | Scikit-learn (Random Forest Classifier) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Generative AI | Groq Cloud API (Llama 3.3-70B-Versatile) |

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vaishaldsouza/ai-nids-dashboard.git
   cd ai-nids-dashboard
````

### 2️⃣ Install Dependencies

Ensure Python **3.8+** is installed.

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run nids_app.py
```

---

## 📊 How It Works

### 🔧 Initialize Detection Engine

* Use the **sidebar controls** to configure:

  * Number of Decision Trees
  * Test Split Percentage
* Click **Initialize** to train the model

### 🌐 Simulate Network Traffic

* Input:

  * Destination Port
  * Packet Duration
  * Packet Count
* Simulates real-world traffic behavior

### 🔍 Analyze Traffic

* ML model classifies traffic:

  * **Benign**
  * **DDoS**
  * **PortScan**
* Malicious traffic triggers a **Critical Alert**

### 🤖 AI Explanation (Optional)

* Enter your **Groq API Key**
* Receive a detailed, SOC-style technical explanation of:

  * Why the traffic was flagged
  * Which features contributed most

### 🗂️ Audit & Export

* View incident history in real time
* Export logs to **CSV** for compliance or reporting

---

## 📈 Model Performance & Transparency

* **Confusion Matrix**

  * Visual evaluation of classification results

* **Feature Importance Chart**

  * Shows how each feature influences predictions
  * Enhances trust and explainability

---

## ⚠️ Disclaimer

> This project is a **student-built educational simulation** designed to demonstrate AI-driven cybersecurity concepts.
> It is **not intended for production use** or deployment in real-world networks.

---
