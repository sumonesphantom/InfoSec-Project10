"""
Bidirectional LSTM with Attention Mechanism for Phishing Email Detection.

Architecture:
    Input -> Embedding -> BiLSTM -> Attention -> Dense -> Sigmoid

References:
    - Li et al. (2022): LSTM Based Phishing Detection for Big Email Data
    - Peng et al. (2021): Phishing email detection based on attention mechanism
    - Adebowale et al. (2023): Intelligent phishing detection using deep learning
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class AttentionLayer(layers.Layer):
    """
    Attention mechanism that learns to weight different time steps
    of the BiLSTM output based on their relevance for classification.

    Computes: attention_weights = softmax(tanh(H * W + b) * u)
    Context vector = sum(attention_weights * H)
    """

    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='context_vector',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        # Score computation
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)  # (batch, time, units)
        attention_weights = tf.nn.softmax(
            tf.reduce_sum(score * self.u, axis=-1, keepdims=True), axis=1
        )  # (batch, time, 1)

        # Weighted sum (context vector)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)  # (batch, features)

        return context_vector, attention_weights

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


def build_bilstm_attention_model(
    vocab_size,
    embedding_dim=128,
    lstm_units=128,
    attention_units=128,
    dense_units=64,
    dropout_rate=0.3,
    recurrent_dropout=0.1,
    max_sequence_length=200,
    learning_rate=0.001,
    jit_compile=False,
):
    """
    Build a Bidirectional LSTM model with Attention mechanism.

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        lstm_units: Number of LSTM units per direction
        attention_units: Number of attention units
        dense_units: Number of units in the dense layer
        dropout_rate: Dropout rate for regularization
        max_sequence_length: Maximum input sequence length
        learning_rate: Learning rate for Adam optimizer
        recurrent_dropout: LSTM recurrent dropout; use 0 for faster training (FAST_RNN=1).
        jit_compile: If True, compile step with XLA (set TF_JIT=1); may help on some setups.

    Returns:
        Compiled Keras Model
    """
    # Input layer
    inputs = layers.Input(shape=(max_sequence_length,), name='input')

    # Embedding layer
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_sequence_length,
        name='embedding'
    )(inputs)

    # Spatial Dropout for embedding regularization
    x = layers.SpatialDropout1D(dropout_rate, name='spatial_dropout')(x)

    # Bidirectional LSTM - returns full sequence for attention
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(1e-5),
            name='lstm'
        ),
        name='bidirectional_lstm'
    )(x)

    # Attention layer
    context_vector, attention_weights = AttentionLayer(
        units=attention_units, name='attention'
    )(x)

    # Dense classification layers
    x = layers.BatchNormalization(name='batch_norm')(context_vector)
    x = layers.Dense(dense_units, activation='relu', name='dense_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    x = layers.Dense(32, activation='relu', name='dense_2')(x)

    # Output layer (binary classification); float32 head when using mixed_float16 policy
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='output')(x)

    # Build model
    model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_Attention_Phishing')

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')],
        jit_compile=jit_compile,
    )

    return model


def build_conv_pool_model(
    vocab_size,
    embedding_dim=128,
    conv_filters=256,
    kernel_size=5,
    dense_units=64,
    dropout_rate=0.3,
    max_sequence_length=200,
    learning_rate=0.001,
    jit_compile=False,
):
    """
    Embedding + Conv1D + global max pool — much faster on GPU/Metal than BiLSTM+attention.
    Same padded integer input as the BiLSTM model.
    """
    inputs = layers.Input(shape=(max_sequence_length,), name='input')
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_sequence_length,
        name='embedding',
    )(inputs)
    x = layers.SpatialDropout1D(dropout_rate, name='spatial_dropout')(x)
    x = layers.Conv1D(
        conv_filters,
        kernel_size,
        activation='relu',
        padding='same',
        name='conv1d',
    )(x)
    x = layers.GlobalMaxPooling1D(name='pool')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    x = layers.Dense(dense_units, activation='relu', name='dense_1')(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='Conv1D_Phishing')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')],
        jit_compile=jit_compile,
    )
    return model


def build_attention_extraction_model(trained_model):
    """
    Build a model that also outputs attention weights for interpretability.
    """
    attention_layer = trained_model.get_layer('attention')
    bilstm_output = trained_model.get_layer('bidirectional_lstm').output
    _, attention_weights = attention_layer(bilstm_output)

    attention_model = Model(
        inputs=trained_model.input,
        outputs=[trained_model.output, attention_weights]
    )
    return attention_model
