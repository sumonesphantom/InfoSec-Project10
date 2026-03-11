"""
FastAPI REST API for Phishing Email Detection.
Provides endpoints for real-time email classification using the trained
BiLSTM + Attention model.
"""

import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_text, load_tokenizer, MAX_SEQUENCE_LENGTH
from src.model import AttentionLayer, build_attention_extraction_model

app = FastAPI(
    title="Phishing Email Detection API",
    description="BiLSTM + Attention model for enterprise email phishing detection",
    version="1.0.0"
)

# Global model and tokenizer
model = None
attn_model = None
tokenizer = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')


class EmailRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Dear user, your account has been compromised. Click here to verify your identity immediately."
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    phishing_probability: float
    top_attention_words: list


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup."""
    global model, attn_model, tokenizer

    model_path = os.path.join(MODEL_DIR, 'best_model.keras')
    tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')

    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}. Train the model first.")
        return

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    attn_model = build_attention_extraction_model(model)
    tokenizer = load_tokenizer(tokenizer_path)
    print("Model and tokenizer loaded successfully.")


@app.get("/")
async def root():
    return {"message": "Phishing Email Detection API", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(email: EmailRequest):
    """Classify an email as phishing or legitimate."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    # Preprocess
    cleaned = preprocess_text(email.text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH,
                           padding='post', truncating='post')

    # Predict with attention
    prediction, attention_weights = attn_model.predict(padded, verbose=0)
    phishing_prob = float(prediction[0][0])
    is_phishing = phishing_prob >= 0.5

    # Get top attended words
    tokens = cleaned.split()[:MAX_SEQUENCE_LENGTH]
    attn_vals = attention_weights[0].flatten()[:len(tokens)]
    if len(tokens) > 0 and attn_vals.max() > 0:
        top_indices = np.argsort(attn_vals)[-10:][::-1]
        top_words = [{"word": tokens[i], "attention": float(attn_vals[i])}
                     for i in top_indices if i < len(tokens)]
    else:
        top_words = []

    return PredictionResponse(
        prediction="Phishing" if is_phishing else "Legitimate",
        confidence=phishing_prob if is_phishing else 1 - phishing_prob,
        phishing_probability=phishing_prob,
        top_attention_words=top_words
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
