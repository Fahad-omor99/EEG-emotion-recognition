# Phase 12: Streamlit real-time emotion dashboard
import streamlit as st
import numpy as np
import joblib
import time
import sys
import os
sys.path.insert(0, 'src')

from load_data import load_all_subjects
from label_utils import convert_labels, LABEL_NAMES
from realtime_sim import EEGBuffer, RealTimeSimulator

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Emotion Recognition",
    page_icon="🧠",
    layout="wide"
)

# ── Emotion styling ────────────────────────────────────────────
EMOTION_COLORS = {
    0: "#2ECC71",   # Happy  → green
    1: "#E74C3C",   # Fear   → red
    2: "#3498DB",   # Sad    → blue
    3: "#95A5A6",   # Neutral→ grey
}

EMOTION_EMOJIS = {
    0: "😊",
    1: "😨",
    2: "😢",
    3: "😐",
}

# ── Header ─────────────────────────────────────────────────────
st.title("🧠 Real-Time EEG Emotion Recognition")
st.markdown("**DEAP Dataset Simulation** — SVM + Differential Entropy Features")
st.divider()

# ── Sidebar controls ───────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
subject_id = st.sidebar.slider("Subject", 1, 32, 1) - 1
trial_id   = st.sidebar.slider("Trial",   1, 40, 1) - 1
speed      = st.sidebar.selectbox("Playback speed", [1, 2, 5, 10], index=2)
start_btn  = st.sidebar.button("▶ Start Simulation", type="primary")
st.sidebar.divider()
st.sidebar.markdown("### About")
st.sidebar.markdown("""
- **Model**: SVM + RBF kernel
- **Features**: Differential Entropy
- **Bands**: δ θ α β γ
- **Accuracy**: 65%+
- **Classes**: Happy / Fear / Sad / Neutral
""")

# ── Layout ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown("### 🎯 Current Prediction")
    emotion_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Confidence Scores")
    confidence_placeholder = st.empty()

with col3:
    st.markdown("### ✅ True Label")
    true_placeholder = st.empty()

st.divider()
st.markdown("### 📈 Prediction History")
history_placeholder = st.empty()
st.divider()
log_placeholder = st.empty()

# ── Loading model and data ────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('models/svm_model.pkl')

@st.cache_resource
def load_data():
    all_eeg, all_labels = load_all_subjects()
    return all_eeg, convert_labels(all_labels)

# ── Running simulation ─────────────────────────────────────────────
if start_btn:
    model = load_model()
    all_eeg, discrete_labels = load_data()

    true_label   = discrete_labels[subject_id, trial_id]
    true_name    = LABEL_NAMES[true_label]
    true_emoji   = EMOTION_EMOJIS[true_label]
    true_color   = EMOTION_COLORS[true_label]

    true_placeholder.markdown(
        f"<div style='text-align:center; padding:20px; "
        f"background:{true_color}22; border-radius:10px; "
        f"border: 2px solid {true_color}'>"
        f"<h1>{true_emoji}</h1>"
        f"<h3>{true_name}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

    simulator  = RealTimeSimulator(model, subject_id, trial_id)
    simulator.load_trial(all_eeg, discrete_labels)

    history       = []
    log_lines     = []
    chunk_size    = 64
    step          = 0

    while True:
        chunk = simulator.get_next_chunk()
        if chunk is None:
            st.success(f"✅ Trial complete! Most predicted: "
                      f"{LABEL_NAMES[max(set(history), key=history.count)] if history else 'N/A'}")
            break

        simulator.buffer.add_chunk(chunk)

        if simulator.buffer.is_ready:
            window              = simulator.buffer.get_window()
            prediction, probs   = simulator.predict(window)
            pred_name           = LABEL_NAMES[prediction]
            pred_emoji          = EMOTION_EMOJIS[prediction]
            pred_color          = EMOTION_COLORS[prediction]
            confidence          = probs.max() * 100
            step               += 1
            history.append(prediction)

            # ── Current emotion display ────────────────────────
            emotion_placeholder.markdown(
                f"<div style='text-align:center; padding:30px; "
                f"background:{pred_color}22; border-radius:15px; "
                f"border: 3px solid {pred_color}'>"
                f"<h1 style='font-size:80px'>{pred_emoji}</h1>"
                f"<h2 style='color:{pred_color}'>{pred_name}</h2>"
                f"<p style='font-size:18px'>Step {step}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            # ── Confidence bars ────────────────────────────────
            conf_html = ""
            for i, (lid, lname) in enumerate(LABEL_NAMES.items()):
                pct   = probs[i] * 100
                color = EMOTION_COLORS[lid]
                conf_html += (
                    f"<div style='margin:8px 0'>"
                    f"<span style='width:80px;display:inline-block'>"
                    f"{EMOTION_EMOJIS[lid]} {lname}</span>"
                    f"<div style='display:inline-block;width:60%;background:#eee;"
                    f"border-radius:5px;height:20px;vertical-align:middle'>"
                    f"<div style='width:{pct:.0f}%;background:{color};"
                    f"height:100%;border-radius:5px'></div></div>"
                    f" <span style='color:{color}'><b>{pct:.1f}%</b></span>"
                    f"</div>"
                )
            confidence_placeholder.markdown(conf_html, unsafe_allow_html=True)

            # ── History chart ──────────────────────────────────
            if len(history) > 1:
                import pandas as pd
                df = pd.DataFrame({
                    'Step'   : range(1, len(history)+1),
                    'Emotion': [LABEL_NAMES[h] for h in history]
                })
                history_placeholder.dataframe(
                    df.tail(10),
                    use_container_width=True,
                    hide_index=True
                )

            # ── Log ───────────────────────────────────────────
            log_lines.append(
                f"Step {step:03d} | {pred_name:8s} | {confidence:.1f}%"
            )
            log_placeholder.code("\n".join(log_lines[-8:]))

        time.sleep(chunk_size / 128 / speed)