"""
Standalone Evaluation Script.

Loads a saved model and generates evaluation metrics, plots, and reports
without retraining. Use this after training has been completed.
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import AttentionLayer
from src.train import evaluate_model
from src.paths import trained_model_path, RESULTS_DIR, MODEL_DIR


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = trained_model_path()
    if not os.path.exists(model_path):
        print("ERROR: No trained model found. Run 'python src/train.py' first.")
        sys.exit(1)

    print("Loading saved model...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    model.summary()

    print("Loading test data...")
    X_test = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(MODEL_DIR, 'y_test.npy'))
    print(f"Test set: {X_test.shape[0]} samples")

    metrics = evaluate_model(model, X_test, y_test, RESULTS_DIR)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
