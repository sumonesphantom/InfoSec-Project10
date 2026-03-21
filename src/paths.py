"""Shared project paths and trained-model location (MODEL_ARCH / MODEL_PATH)."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')


def model_arch():
    return os.environ.get('MODEL_ARCH', 'bilstm').strip().lower()


def trained_model_path():
    override = os.environ.get('MODEL_PATH', '').strip()
    if override:
        return override if os.path.isabs(override) else os.path.join(PROJECT_ROOT, override)
    name = 'best_model_conv.keras' if model_arch() == 'conv' else 'best_model.keras'
    return os.path.join(MODEL_DIR, name)


def milestone_checkpoint_pattern():
    if model_arch() == 'conv':
        return os.path.join(MODEL_DIR, 'model_conv_acc_{val_accuracy:.4f}_epoch_{epoch:02d}.keras')
    return os.path.join(MODEL_DIR, 'model_acc_{val_accuracy:.4f}_epoch_{epoch:02d}.keras')
