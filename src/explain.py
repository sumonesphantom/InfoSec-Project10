"""
Explainability Module using LIME and SHAP for Phishing Detection Model.

Provides interpretability for the BiLSTM + Attention model predictions.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lime
import lime.lime_text
import shap
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_tokenizer, preprocess_text, MAX_SEQUENCE_LENGTH
from src.model import AttentionLayer, build_attention_extraction_model
from src.paths import trained_model_path, MODEL_DIR, RESULTS_DIR


def load_model_and_tokenizer():
    """Load trained model and tokenizer."""
    path = trained_model_path()
    model = tf.keras.models.load_model(
        path,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    tokenizer = load_tokenizer(os.path.join(MODEL_DIR, 'tokenizer.pkl'))
    return model, tokenizer


def create_prediction_function(model, tokenizer):
    """Create a prediction function compatible with LIME."""
    def predict_proba(texts):
        cleaned = [preprocess_text(t) for t in texts]
        sequences = tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,
                               padding='post', truncating='post')
        preds = model.predict(padded, verbose=0).flatten()
        return np.column_stack([1 - preds, preds])
    return predict_proba


def explain_with_lime(model, tokenizer, text, save_path=None):
    """
    Generate LIME explanation for a single email prediction.

    Args:
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        text: Raw email text
        save_path: Path to save the explanation HTML
    """
    predict_fn = create_prediction_function(model, tokenizer)

    explainer = lime.lime_text.LimeTextExplainer(
        class_names=['Legitimate', 'Phishing'],
        split_expression=r'\W+',
        random_state=42
    )

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=20,
        num_samples=500
    )

    if save_path:
        explanation.save_to_file(save_path)
        print(f"LIME explanation saved to {save_path}")

    # Print top features
    print("\nLIME Top Features:")
    for feature, weight in explanation.as_list():
        direction = "PHISHING" if weight > 0 else "LEGITIMATE"
        print(f"  {feature:25s} -> {direction} (weight: {weight:+.4f})")

    return explanation


def explain_with_shap(model, tokenizer, texts, save_dir=None):
    """
    Generate SHAP explanations for email predictions.

    Args:
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        texts: List of raw email texts (small sample, e.g. 10-50)
        save_dir: Directory to save SHAP plots
    """
    predict_fn = create_prediction_function(model, tokenizer)

    # Use KernelExplainer with a small background dataset
    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(
        lambda x: predict_fn(x)[:, 1],
        masker=masker,
        output_names=['Phishing Score']
    )

    shap_values = explainer(texts[:10])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Text plot
        plt.figure()
        shap.plots.text(shap_values[0], display=False)
        plt.savefig(os.path.join(save_dir, 'shap_text_plot.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Bar plot of mean absolute SHAP values
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_bar_plot.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"SHAP plots saved to {save_dir}")

    return shap_values


def visualize_attention(model, tokenizer, text, save_path=None):
    """
    Visualize attention weights for a given email text.

    Args:
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        text: Raw email text
        save_path: Path to save attention visualization
    """
    try:
        model.get_layer('attention')
    except ValueError:
        print("Skipping attention visualization (conv model has no attention layer).")
        return None, None, None

    # Build attention extraction model
    attn_model = build_attention_extraction_model(model)

    # Preprocess
    cleaned = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH,
                           padding='post', truncating='post')

    # Get predictions and attention weights
    prediction, attention_weights = attn_model.predict(padded, verbose=0)
    attention_weights = attention_weights[0].flatten()

    # Get tokens
    tokens = cleaned.split()[:MAX_SEQUENCE_LENGTH]
    attn_values = attention_weights[:len(tokens)]

    # Normalize attention for visualization
    if attn_values.max() > 0:
        attn_values = attn_values / attn_values.max()

    # Plot top-20 attended tokens
    top_k = min(20, len(tokens))
    top_indices = np.argsort(attn_values)[-top_k:]
    top_tokens = [tokens[i] for i in top_indices]
    top_weights = [attn_values[i] for i in top_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_k), top_weights, color='steelblue')
    plt.yticks(range(top_k), top_tokens)
    plt.xlabel('Attention Weight')
    plt.title(f'Top {top_k} Attended Words (Pred: {"Phishing" if prediction[0][0] > 0.5 else "Legitimate"}, '
              f'Score: {prediction[0][0]:.4f})')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Attention visualization saved to {save_path}")
    plt.close()

    return tokens, attn_values, prediction[0][0]


def run_explainability_analysis():
    """Run full explainability analysis on sample test emails."""
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # Load test data
    X_test = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(MODEL_DIR, 'y_test.npy'))

    # Get predictions for test data
    preds = model.predict(X_test[:100], verbose=0).flatten()

    # Reverse tokenize to get text back
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

    def sequences_to_text(seq):
        return ' '.join([reverse_word_index.get(idx, '') for idx in seq if idx > 0])

    # Find examples: correct phishing, correct legitimate, misclassified
    phishing_idx = np.where((y_test[:100] == 1) & (preds > 0.5))[0]
    legit_idx = np.where((y_test[:100] == 0) & (preds < 0.5))[0]

    os.makedirs(os.path.join(RESULTS_DIR, 'explanations'), exist_ok=True)

    # LIME explanations
    if len(phishing_idx) > 0:
        text = sequences_to_text(X_test[phishing_idx[0]])
        print(f"\n--- LIME: Phishing Email (True Positive) ---")
        explain_with_lime(model, tokenizer, text,
                          os.path.join(RESULTS_DIR, 'explanations', 'lime_phishing.html'))
        visualize_attention(model, tokenizer, text,
                            os.path.join(RESULTS_DIR, 'explanations', 'attention_phishing.png'))

    if len(legit_idx) > 0:
        text = sequences_to_text(X_test[legit_idx[0]])
        print(f"\n--- LIME: Legitimate Email (True Negative) ---")
        explain_with_lime(model, tokenizer, text,
                          os.path.join(RESULTS_DIR, 'explanations', 'lime_legitimate.html'))
        visualize_attention(model, tokenizer, text,
                            os.path.join(RESULTS_DIR, 'explanations', 'attention_legitimate.png'))

    print("\nExplainability analysis complete!")


if __name__ == '__main__':
    run_explainability_analysis()
