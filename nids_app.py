import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
from datetime import datetime
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Professional AI-NIDS Dashboard", page_icon="🛡️", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: #00e5ff !important;}
.stMetric {background-color:#161b22; padding:15px; border-radius:10px; border: 1px solid #30363d;}
div.stButton > button {width: 100%; border-radius: 5px; height: 3em; background-color: #00e5ff; color: black; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if 'history' not in st.session_state: st.session_state['history'] = []
if 'model' not in st.session_state: st.session_state['model'] = None

# ---------------- DATA GENERATION ----------------
@st.cache_data
def load_realistic_nids_data():
    np.random.seed(42)
    n_samples = 5000
    data = {
        'Destination_Port': np.random.randint(1, 65535, n_samples),
        'Flow_Duration': np.random.randint(10, 100000, n_samples),
        'Total_Fwd_Packets': np.random.randint(1, 100, n_samples),
        'Packet_Length_Mean': np.random.uniform(10, 1500, n_samples),
        'Active_Mean': np.random.uniform(0, 1000, n_samples),
        'Label': np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    }
    df = pd.DataFrame(data)

    df.loc[df['Label'] == 1, 'Total_Fwd_Packets'] += np.random.randint(50, 300, size=len(df[df['Label']==1]))
    df.loc[df['Label'] == 1, 'Flow_Duration'] = np.random.randint(1, 500, size=len(df[df['Label']==1]))
    df.loc[df['Label'] == 2, 'Destination_Port'] = np.random.choice([21, 22, 23, 80, 443, 8080], size=len(df[df['Label']==2]))
    
    noise_indices = df[df['Label'] == 0].sample(frac=0.05).index
    df.loc[noise_indices, 'Total_Fwd_Packets'] = np.random.randint(200, 400)
    return df

df = load_realistic_nids_data()

# ---------------- SIDEBAR: SOC CONTROL PANEL ----------------
st.sidebar.title("🛡️ SOC Control Panel")
groq_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="gsk_...")
st.sidebar.caption("Add key for AI Analyst explanations.")

st.sidebar.markdown("---")
n_trees = st.sidebar.slider("Random Forest Trees", 50, 200, 100)
test_split = st.sidebar.slider("Test Data Split", 0.1, 0.4, 0.2)

if st.sidebar.button("🚀 Initialize AI Engine"):
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    model.fit(X_train, y_train)
    
    st.session_state['model'] = model
    st.session_state['accuracy'] = accuracy_score(y_test, model.predict(X_test))
    st.session_state['features'] = X.columns.tolist()
    st.session_state['cm'] = confusion_matrix(y_test, model.predict(X_test))
    st.sidebar.success("Engine Online")

# ---------------- MAIN DASHBOARD ----------------
st.title("🛡️ AI-Powered Network Intrusion Detection System")

if st.session_state['model'] is None:
    st.info("Welcome, Analyst. Please configure the SOC Panel and Initialize the Engine.")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*ZFuMI83pcBy9S_S_v_O3WA.png", width=700)
else:
    # 1. METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("System Accuracy", f"{st.session_state['accuracy']*100:.2f}%")
    m2.metric("Dataset Size", len(df))
    m3.metric("Trees in Forest", n_trees)
    m4.metric("Engine Status", "Active")

    st.divider()

    # 2. SIMULATOR & VERDICT
    col_sim, col_verdict = st.columns([1, 1.5])

    with col_sim:
        st.subheader("📡 Traffic Simulator")
        p_port = st.number_input("Dest Port", 1, 65535, 80)
        p_dur = st.number_input("Flow Duration (ms)", 0, 100000, 500)
        p_pck = st.number_input("Total Packets", 1, 1000, 10)
        p_len = st.number_input("Packet Length Mean", 0, 1500, 64)
        
        if st.button("🔍 Analyze Packet"):
            input_df = pd.DataFrame([[p_port, p_dur, p_pck, p_len, 50]], columns=st.session_state['features'])
            pred = st.session_state['model'].predict(input_df)[0]
            prob = st.session_state['model'].predict_proba(input_df)[0][pred]
            mapping = {0: "BENIGN", 1: "DDoS ATTACK", 2: "PORTSCAN"}
            st.session_state['last_res'] = {"type": mapping[pred], "data": input_df.iloc[0].to_dict(), "conf": prob}
            st.session_state['history'].append({"Time": datetime.now().strftime("%H:%M:%S"), "Event": mapping[pred], "Conf": f"{prob*100:.1f}%"})

    with col_verdict:
        st.subheader("🤖 AI Verdict & Analyst")
        if 'last_res' in st.session_state:
            res_obj = st.session_state['last_res']
            if res_obj['type'] == "BENIGN":
                st.success(f"### {res_obj['type']} DETECTED (Confidence: {res_obj['conf']*100:.1f}%)")
            else:
                st.error(f"### {res_obj['type']} DETECTED (Confidence: {res_obj['conf']*100:.1f}%)")
            
            if st.button("✨ Explain with AI"):
                if not groq_key:
                    st.warning("⚠️ Add Groq Key in the sidebar for AI analysis.")
                    st.info(f"Offline logic suggests this is {res_obj['type']} based on current feature weights.")
                else:
                    try:
                        client = Groq(api_key=groq_key)
                        prompt = f"Explain why a network packet with {res_obj['data']} is {res_obj['type']} in 3 short points."
                        chat = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}])
                        st.info(chat.choices[0].message.content)
                    except Exception as e: st.error(f"AI Error: {e}")
        else: st.info("Run analysis to see results.")

    st.divider()

    # 3. VISUALS
    col_fi, col_cm = st.columns(2)
    with col_fi:
        st.subheader("📊 Feature Importance")
        importances = st.session_state['model'].feature_importances_
        fi_df = pd.DataFrame({'Feature': st.session_state['features'], 'Importance': importances}).sort_values('Importance', ascending=False)
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax1)
        st.pyplot(fig1)

    with col_cm:
        st.subheader("🎯 Confusion Matrix")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(st.session_state['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'DDoS', 'PortScan'], yticklabels=['Benign', 'DDoS', 'PortScan'], ax=ax2)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig2)

    st.divider()

    # 4. LOGS & EXPORT
    st.subheader("📜 Recent Incident Logs")
    if st.session_state['history']:
        h_df = pd.DataFrame(st.session_state['history'])
        st.table(h_df.iloc[::-1].head(5))
        
        # EXPORT FEATURE
        csv = h_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full Audit Log to CSV",
            data=csv,
            file_name=f'nids_audit_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
        )
    else:
        st.write("No incidents logged yet.")

st.markdown("---")
st.caption("AI-NIDS Professional Instance | SOC Dashboard V2.0")