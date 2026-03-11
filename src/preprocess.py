"""
Text Preprocessing Pipeline for Phishing Email Detection
Handles cleaning, tokenization, and sequence preparation for BiLSTM model.
"""

import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import os


STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Maximum sequence length and vocabulary size
MAX_SEQUENCE_LENGTH = 200
MAX_VOCAB_SIZE = 50000


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
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


def preprocess_text(text):
    """Full preprocessing pipeline for a single text."""
    text = clean_text(text)
    text = tokenize_and_lemmatize(text)
    return text


def load_and_preprocess_data(data_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load dataset, preprocess text, and create train/val/test splits.

    Returns:
        X_train, X_val, X_test: Padded sequences
        y_train, y_val, y_test: Labels
        tokenizer: Fitted Keras Tokenizer
    """
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Drop rows with missing text
    df = df.dropna(subset=['text_combined'])
    df = df[df['text_combined'].str.strip().astype(bool)]

    print("\nPreprocessing text...")
    df['cleaned_text'] = df['text_combined'].apply(preprocess_text)

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
