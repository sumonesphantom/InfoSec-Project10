"""
Training Pipeline for BiLSTM Phishing Email Detection Model.

Handles model training, evaluation, and result visualization.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score
)
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import (
    load_and_preprocess_data, save_tokenizer,
    MAX_SEQUENCE_LENGTH, MAX_VOCAB_SIZE
)
from src.model import build_bilstm_attention_model, build_conv_pool_model
from src.paths import (
    PROJECT_ROOT, MODEL_DIR, RESULTS_DIR,
    trained_model_path, milestone_checkpoint_pattern, model_arch,
)

DATA_PATH = os.path.join(PROJECT_ROOT, 'Dataset', 'phishing_email.csv')
_ATTN = {'AttentionLayer': __import__('src.model', fromlist=['AttentionLayer']).AttentionLayer}


def _training_batch_size():
    default = '512' if model_arch() == 'conv' else '128'
    return int(os.environ.get('TRAIN_BATCH_SIZE', default))


def _jit_compile():
    return os.environ.get('TF_JIT', '').strip() == '1'


def _recurrent_dropout_bilstm():
    return 0.0 if os.environ.get('FAST_RNN', '').strip() == '1' else 0.1


def configure_tensorflow_for_training():
    """Threads, device logging, optional mixed precision when a GPU is visible."""
    intra = os.environ.get('TF_INTRA_OP_THREADS', '').strip()
    inter = os.environ.get('TF_INTER_OP_THREADS', '').strip()
    if intra.isdigit():
        tf.config.threading.set_intra_op_parallelism_threads(int(intra))
    if inter.isdigit():
        tf.config.threading.set_inter_op_parallelism_threads(int(inter))

    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print(f"TensorFlow devices — GPU: {gpus}, CPU: {cpus}")

    if os.environ.get('TF_MIXED_PRECISION', '').strip() == '1':
        if gpus:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled (TF_MIXED_PRECISION=1).")
        else:
            print("TF_MIXED_PRECISION=1 ignored: no GPU visible.")

    if _jit_compile():
        print("TF_JIT=1: training step will use jit_compile=True (XLA where supported).")


def make_tf_dataset(X, y, batch_size, shuffle=False, seed=42):
    y = np.asarray(y, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 10000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def plot_training_history(history, save_dir):
    """Plot and save training/validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision & Recall
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].plot(history.history['recall'], label='Train Recall')
    axes[1, 0].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # AUC
    axes[1, 1].plot(history.history['auc'], label='Train AUC')
    axes[1, 1].plot(history.history['val_auc'], label='Val AUC')
    axes[1, 1].set_title('AUC-ROC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"Training history plot saved.")


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Confusion matrix saved.")


def plot_roc_curve(y_true, y_prob, save_dir):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    print(f"ROC curve saved.")


def plot_precision_recall_curve(y_true, y_prob, save_dir):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'), dpi=150)
    plt.close()
    print(f"Precision-recall curve saved.")


def evaluate_model(model, X_test, y_test, save_dir):
    """Evaluate model and generate all reports and plots."""
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)

    # Predictions
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    # Test loss and metrics
    test_results = model.evaluate(X_test, y_test, verbose=0)
    metrics_names = model.metrics_names
    print("\nTest Metrics:")
    for name, value in zip(metrics_names, test_results):
        print(f"  {name}: {value:.4f}")

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=['Legitimate', 'Phishing'],
        digits=4
    )
    print(f"\nClassification Report:\n{report}")

    # Save classification report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write("BiLSTM + Attention Phishing Detection - Test Results\n")
        f.write("="*60 + "\n\n")
        f.write("Test Metrics:\n")
        for name, value in zip(metrics_names, test_results):
            f.write(f"  {name}: {value:.4f}\n")
        f.write(f"\nClassification Report:\n{report}\n")

    # Save metrics as JSON
    metrics_dict = {
        name: float(value) for name, value in zip(metrics_names, test_results)
    }
    metrics_dict['f1_score'] = float(f1_score(y_test, y_pred))
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    # Generate plots
    plot_confusion_matrix(y_test, y_pred, save_dir)
    plot_roc_curve(y_test, y_prob, save_dir)
    plot_precision_recall_curve(y_test, y_prob, save_dir)

    return metrics_dict


def train():
    """Main training pipeline."""
    configure_tensorflow_for_training()
    arch = model_arch()
    if arch not in ('bilstm', 'conv'):
        raise ValueError(f"Unknown MODEL_ARCH={arch!r}; use 'bilstm' or 'conv'.")
    batch_size = _training_batch_size()
    ckpt_path = trained_model_path()
    print(f"Model architecture: {arch} (set MODEL_ARCH=bilstm|conv).")
    print(f"Checkpoint file: {os.path.basename(ckpt_path)}")
    print(f"Training batch size: {batch_size} (override with TRAIN_BATCH_SIZE).")
    if arch == 'bilstm' and os.environ.get('FAST_RNN', '').strip() == '1':
        print("FAST_RNN=1: LSTM recurrent_dropout=0 for faster steps.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 1: Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = \
        load_and_preprocess_data(DATA_PATH)

    # Save tokenizer
    save_tokenizer(tokenizer, os.path.join(MODEL_DIR, 'tokenizer.pkl'))

    # Save test data for later use with explainability
    np.save(os.path.join(MODEL_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(MODEL_DIR, 'y_test.npy'), y_test)

    # Step 2: Build model
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    jc = _jit_compile()
    if arch == 'conv':
        model = build_conv_pool_model(
            vocab_size=vocab_size,
            embedding_dim=128,
            conv_filters=256,
            kernel_size=5,
            dense_units=64,
            dropout_rate=0.3,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            learning_rate=0.001,
            jit_compile=jc,
        )
    else:
        model = build_bilstm_attention_model(
            vocab_size=vocab_size,
            embedding_dim=128,
            lstm_units=128,
            attention_units=128,
            dense_units=64,
            dropout_rate=0.3,
            recurrent_dropout=_recurrent_dropout_bilstm(),
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            learning_rate=0.001,
            jit_compile=jc,
        )

    print("\nModel Summary:")
    model.summary()

    # Step 3: Check if a previously trained model exists to resume from
    pretrained_path = ckpt_path
    if os.path.exists(pretrained_path):
        print(f"\nFound existing model at {pretrained_path}")
        print("Loading pre-trained weights to resume training...")
        model = tf.keras.models.load_model(pretrained_path, custom_objects=_ATTN)
        # Quick evaluation to decide if retraining is needed
        val_results = model.evaluate(X_val, y_val, verbose=0)
        val_acc = val_results[1]  # accuracy is the second metric
        print(f"Existing model validation accuracy: {val_acc:.4f}")
        if val_acc >= 0.95:
            print("Model already meets accuracy threshold (>=95%). Skipping training.")
            print("To force retraining, delete the model file and run again.")
            # Still evaluate on test set and generate results
            metrics = evaluate_model(model, X_test, y_test, RESULTS_DIR)
            return model, None, metrics

    # Step 3: Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model based on validation loss
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Also save checkpoints at accuracy milestones
        ModelCheckpoint(
            filepath=milestone_checkpoint_pattern(),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # Step 4: Train (tf.data prefetch overlaps host prep with device work)
    train_ds = make_tf_dataset(X_train, y_train, batch_size, shuffle=True)
    val_ds = make_tf_dataset(X_val, y_val, batch_size, shuffle=False)

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    # Step 5: Plot training history
    plot_training_history(history, RESULTS_DIR)

    # Step 6: Evaluate on test set
    print("\nLoading best model for evaluation...")
    best_model = tf.keras.models.load_model(ckpt_path, custom_objects=_ATTN)
    metrics = evaluate_model(best_model, X_test, y_test, RESULTS_DIR)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*60)

    return model, history, metrics


if __name__ == '__main__':
    train()
