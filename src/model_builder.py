"""
Model builder module for Next-Word Prediction LSTM project.
Defines a causal multi-layer stacked LSTM architecture with Embedding,
regularized LSTM layers, dense projection, and a Softmax vocabulary head.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


def build_lstm_model(
    vocab_size,
    max_sequence_len=12,
    embedding_dim=128,
    lstm_units_1=128,
    lstm_units_2=128,
    dense_units=128,
    dropout_rate=0.15,
    learning_rate=0.002,
    **kwargs,
):
    """
    Build and compile a causal prefix-conditioned stacked LSTM for next-word prediction.
    Architecture:
      Embedding(128) -> LSTM(128, return_sequences=True) -> Dropout(0.15) ->
      LSTM(128, return_sequences=False) -> Dropout(0.15) -> Dense(128, ReLU) -> Dense(vocab_size, Softmax)
    Strictly causal (left-to-right) so hidden states faithfully accumulate sequential context.
    """
    input_len = max_sequence_len - 1

    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="embedding_layer",
            ),
            LSTM(
                lstm_units_1,
                return_sequences=True,
                name="lstm_layer_1",
            ),
            Dropout(dropout_rate, name="dropout_1"),
            LSTM(
                lstm_units_2,
                return_sequences=False,
                name="lstm_layer_2",
            ),
            Dropout(dropout_rate, name="dropout_2"),
            Dense(dense_units, activation="relu", name="dense_projection"),
            Dense(vocab_size, activation="softmax", name="softmax_output"),
        ],
        name="NextWordPrediction_StackedLSTM",
    )

    model.build(input_shape=(None, input_len))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    return model
