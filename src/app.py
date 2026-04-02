"""
Streamlit frontend for Phishing Email Detection.
Run with: streamlit run src/app.py
"""

import os
import sys
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_text, load_tokenizer, MAX_SEQUENCE_LENGTH
from src.model import AttentionLayer, build_attention_extraction_model
from src.paths import trained_model_path, MODEL_DIR, RESULTS_DIR

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = trained_model_path()
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")

    if not os.path.exists(model_path):
        return None, None, None

    model = tf.keras.models.load_model(
        model_path, custom_objects={"AttentionLayer": AttentionLayer}
    )
    try:
        model.get_layer("attention")
        attn_model = build_attention_extraction_model(model)
    except ValueError:
        attn_model = None

    tokenizer = load_tokenizer(tokenizer_path)
    return model, attn_model, tokenizer


model, attn_model, tokenizer = load_model()

# ---------------------------------------------------------------------------
# Sidebar – model performance
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model Performance")

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{metrics.get('compile_metrics', 0):.2%}")
        col2.metric("F1 Score", f"{metrics.get('f1_score', 0):.2%}")

    st.divider()
    st.subheader("Evaluation Plots")

    plot_files = [
        ("Confusion Matrix", "confusion_matrix.png"),
        ("ROC Curve", "roc_curve.png"),
        ("Precision-Recall", "precision_recall_curve.png"),
        ("Training History", "training_history.png"),
    ]
    for label, fname in plot_files:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            with st.expander(label):
                st.image(path, use_container_width=True)

    st.divider()
    st.subheader("Dataset Insights")
    insight_files = [
        ("Label Distribution", "label_distribution.png"),
        ("Text Length Analysis", "text_length_analysis.png"),
        ("Top Words", "top_words.png"),
        ("Phishing Indicators", "phishing_indicators.png"),
    ]
    for label, fname in insight_files:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            with st.expander(label):
                st.image(path, use_container_width=True)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🛡️ Phishing Email Detector")
st.caption("BiLSTM + Attention model trained on 82,486 emails")

if model is None:
    st.error("Model not found. Run training first (`python -m src.train`).")
    st.stop()

# Sample emails for quick testing
SAMPLES = {
    "-- paste your own --": "",
    "Phishing: account verification": (
        "Dear user, your account has been compromised. "
        "Click here immediately to verify your identity and "
        "reset your password or your account will be suspended."
    ),
    "Phishing: prize scam": (
        "Congratulations! You have been selected as the winner of our "
        "$1,000,000 prize draw. To claim your prize, reply with your "
        "full name, address, and bank account details."
    ),
    "Legitimate: meeting invite": (
        "Hi team, just a reminder that we have a project sync tomorrow "
        "at 10 AM in Conference Room B. Please bring your status updates. "
        "Thanks, Sarah"
    ),
    "Legitimate: IT update": (
        "Hello everyone, we will be performing scheduled maintenance on "
        "the email servers this Saturday from 2-4 AM. No action is required "
        "on your part. — IT Support"
    ),
}

sample = st.selectbox("Try a sample email", options=list(SAMPLES.keys()))

email_text = st.text_area(
    "Paste email text below",
    value=SAMPLES[sample],
    height=200,
    placeholder="Enter the email content you want to analyze...",
)

if st.button("Analyze Email", type="primary", use_container_width=True):
    if not email_text.strip():
        st.warning("Please enter some email text.")
        st.stop()

    with st.spinner("Analyzing..."):
        cleaned = preprocess_text(email_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

        if attn_model is not None:
            prediction, attention_weights = attn_model.predict(padded, verbose=0)
            phishing_prob = float(prediction[0][0])
            tokens = cleaned.split()[:MAX_SEQUENCE_LENGTH]
            attn_vals = attention_weights[0].flatten()[: len(tokens)]
        else:
            phishing_prob = float(model.predict(padded, verbose=0)[0][0])
            tokens, attn_vals = [], np.array([])

    is_phishing = phishing_prob >= 0.5
    confidence = phishing_prob if is_phishing else 1 - phishing_prob

    # --- Result banner ---
    if is_phishing:
        st.error(f"⚠️ **PHISHING** — {confidence:.1%} confidence")
    else:
        st.success(f"✅ **LEGITIMATE** — {confidence:.1%} confidence")

    # --- Probability bar ---
    st.progress(phishing_prob, text=f"Phishing probability: {phishing_prob:.2%}")

    # --- Attention highlights ---
    if len(tokens) > 0 and len(attn_vals) > 0 and attn_vals.max() > 0:
        st.subheader("Attention Analysis")
        st.caption("Words the model focused on most (higher = more influential)")

        norm_attn = attn_vals / attn_vals.max()
        top_indices = np.argsort(attn_vals)[-15:][::-1]

        # Bar chart of top words
        top_words = [tokens[i] for i in top_indices if i < len(tokens)]
        top_scores = [float(attn_vals[i]) for i in top_indices if i < len(tokens)]

        st.bar_chart(
            data=dict(zip(top_words, top_scores)),
            horizontal=True,
        )

        # Highlighted text
        st.subheader("Highlighted Email Text")
        html_parts = []
        for i, token in enumerate(tokens):
            if i < len(norm_attn):
                opacity = float(norm_attn[i])
                if opacity > 0.3:
                    color = "rgba(255, 75, 75," if is_phishing else "rgba(33, 195, 84,"
                    html_parts.append(
                        f'<span style="background:{color} {opacity:.2f}); '
                        f'padding:2px 4px; border-radius:3px; margin:1px;">'
                        f"{token}</span>"
                    )
                else:
                    html_parts.append(token)
            else:
                html_parts.append(token)

        st.markdown(" ".join(html_parts), unsafe_allow_html=True)
