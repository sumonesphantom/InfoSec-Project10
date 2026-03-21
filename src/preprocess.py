"""
Text Preprocessing Pipeline for Phishing Email Detection
Handles cleaning, tokenization, and sequence preparation for BiLSTM model.
"""

import re
import string
import sys
import os
import numpy as np
import pandas as pd
import nltk
from concurrent.futures import ProcessPoolExecutor
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle


# Loaded lazily via `_ensure_nltk_resources()` so imports don't crash
# when NLTK corpora haven't been downloaded yet.
STOP_WORDS = None
LEMMATIZER = WordNetLemmatizer()

# Maximum sequence length and vocabulary size
MAX_SEQUENCE_LENGTH = 200
MAX_VOCAB_SIZE = 50000


def _ensure_nltk_resources():
    """
    Ensure required NLTK corpora/tokenizers are present.
    We download them on-demand so training doesn't fail with LookupError.
    If downloads are blocked/unavailable, we fall back to safer defaults.
    """
    global STOP_WORDS

    required = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/wordnet', 'wordnet'),
    ]
    optional = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
    ]

    for check_path, resource_name in required:
        try:
            nltk.data.find(check_path)
            continue
        except LookupError:
            # If NLTK download is blocked, we don't want training to crash.
            try:
                print(f"Downloading NLTK resource: {resource_name} ...")
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass

    for check_path, resource_name in optional:
        try:
            nltk.data.find(check_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass

    if STOP_WORDS is None:
        try:
            STOP_WORDS = set(stopwords.words('english'))
        except LookupError:
            # Fallback: no stopword removal
            STOP_WORDS = set()


def clean_text(text):
    """Clean and normalize email text."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' email ', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_lemmatize(text):
    """Tokenize, remove stopwords, and lemmatize."""
    if STOP_WORDS is None:
        _ensure_nltk_resources()
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # punkt not available
        tokens = text.split()

    cleaned = []
    for t in tokens:
        if t in STOP_WORDS or len(t) <= 2:
            continue
        try:
            cleaned.append(LEMMATIZER.lemmatize(t))
        except LookupError:
            # wordnet not available
            cleaned.append(t)

    return ' '.join(cleaned)


def preprocess_text(text):
    """Full preprocessing pipeline for a single text."""
    text = clean_text(text)
    text = tokenize_and_lemmatize(text)
    return text


def _ensure_subprocess_pythonpath():
    """So ProcessPoolExecutor workers can `import src.*` when using spawn (macOS)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = root if not prev else f"{root}{os.pathsep}{prev}"


def _preprocess_worker_count():
    raw = os.environ.get('PREPROCESS_WORKERS', '').strip().lower()
    if raw in ('', 'auto'):
        return min(os.cpu_count() or 4, 8)
    try:
        n = int(raw)
    except ValueError:
        return min(os.cpu_count() or 4, 8)
    return max(0, n)


def _parallel_preprocess_texts(texts):
    """CPU-bound NLTK path; parallelize with processes when worthwhile."""
    workers = _preprocess_worker_count()
    n = len(texts)
    if workers <= 1 or n < 3000:
        return [preprocess_text(t) for t in texts]

    _ensure_subprocess_pythonpath()
    chunksize = max(64, n // (workers * 16))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(preprocess_text, texts, chunksize=chunksize))


def load_and_preprocess_data(data_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load dataset, preprocess text, and create train/val/test splits.

    Returns:
        X_train, X_val, X_test: Padded sequences
        y_train, y_val, y_test: Labels
        tokenizer: Fitted Keras Tokenizer
    """
    _ensure_nltk_resources()
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Drop rows with missing text
    df = df.dropna(subset=['text_combined'])
    df = df[df['text_combined'].str.strip().astype(bool)]

    print("\nPreprocessing text...")
    texts = df['text_combined'].astype(str).tolist()
    workers = _preprocess_worker_count()
    if workers > 1 and len(texts) >= 3000:
        print(f"  Using {workers} parallel workers (set PREPROCESS_WORKERS=1 to disable).")
    df['cleaned_text'] = _parallel_preprocess_texts(texts)

    # Remove empty texts after cleaning
    df = df[df['cleaned_text'].str.strip().astype(bool)]
    print(f"Dataset shape after cleaning: {df.shape}")

    texts = df['cleaned_text'].values
    labels = df['label'].values

    # Split: train / temp
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        texts, labels, test_size=(test_size + val_size),
        random_state=random_state, stratify=labels
    )
    # Split temp into val / test
    relative_test = test_size / (test_size + val_size)
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=relative_test,
        random_state=random_state, stratify=y_temp
    )

    print(f"\nSplit sizes - Train: {len(X_train_text)}, Val: {len(X_val_text)}, Test: {len(X_test_text)}")

    # Tokenization
    print("Tokenizing...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_val_seq = tokenizer.texts_to_sequences(X_val_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    # Padding
    X_train = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_val = pad_sequences(X_val_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_test = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
    print(f"Vocabulary size: {vocab_size}")

    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer


def save_tokenizer(tokenizer, path):
    """Save tokenizer to disk."""
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {path}")


def load_tokenizer(path):
    """Load tokenizer from disk."""
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer
