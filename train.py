"""
Training pipeline for Next-Word Prediction LSTM project.
Loads diverse English corpus, splits at sentence level to prevent n-gram leakage,
fits Tokenizer with OOV handling, trains causal stacked BiLSTM network with
smooth learning rate decay and convergence monitoring, and evaluates on held-out test set and benchmark prompts.
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.preprocessing import (
    load_corpus,
    split_corpus,
    fit_tokenizer,
    generate_ngram_sequences,
)
from src.model_builder import build_lstm_model
from evaluate import evaluate_test_set, evaluate_benchmark_prompts
from predict import NextWordPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")
TEST_CACHE_PATH = os.path.join(DATA_DIR, "test_sentences.json")

MODEL_PATH = os.path.join(MODELS_DIR, "next_word_lstm.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pickle")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
EVAL_PATH = os.path.join(MODELS_DIR, "eval_report.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def run_training_pipeline(
    corpus_path=CORPUS_PATH,
    max_sequence_len=12,
    num_words=2500,
    epochs=25,
    batch_size=32,
):
    print("==================================================")
    print("STARTING CAUSAL STACKED LSTM NEXT-WORD PREDICTION TRAINING")
    print("==================================================")

    # 1. Load Clean Corpus
    sentences = load_corpus(corpus_path)
    if len(sentences) < 20:
        raise ValueError(f"Corpus at {corpus_path} contains insufficient sentences.")

    # 2. Sentence-level Train/Val/Test Split (prevents n-gram data leakage)
    train_sents, val_sents, test_sents = split_corpus(
        sentences, test_size=0.10, val_size=0.10, random_state=42
    )

    # Cache test sentences for repeatable evaluation
    with open(TEST_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(test_sents, f, indent=2)

    # 3. Fit Tokenizer on Corpus (with OOV token at index 1)
    tokenizer, vocab_size = fit_tokenizer(sentences, oov_token="<oov>", num_words=num_words)

    # 4. Generate n-gram sequences for each split
    X_train, y_train = generate_ngram_sequences(
        train_sents,
        tokenizer,
        max_sequence_len=max_sequence_len,
        is_training=True,
        oov_drop_rate=0.01,
        vocab_size=vocab_size,
    )
    X_val, y_val = generate_ngram_sequences(
        val_sents,
        tokenizer,
        max_sequence_len=max_sequence_len,
        is_training=False,
        vocab_size=vocab_size,
    )
    X_test, y_test = generate_ngram_sequences(
        test_sents,
        tokenizer,
        max_sequence_len=max_sequence_len,
        is_training=False,
        vocab_size=vocab_size,
    )

    print(
        f"Sequences generated: Train={len(X_train)}, "
        f"Validation={len(X_val)}, Test={len(X_test)}"
    )

    # 5. Build Causal Stacked Unidirectional LSTM Model
    model = build_lstm_model(
        vocab_size=vocab_size,
        max_sequence_len=max_sequence_len,
        embedding_dim=128,
        lstm_units_1=128,
        lstm_units_2=128,
        dense_units=128,
        dropout_rate=0.15,
        learning_rate=0.002,
    )
    model.summary()

    # 6. Training Callbacks: monitor validation accuracy for optimal generalization
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            min_delta=0.005,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=3,
            min_lr=0.0001,
            verbose=1,
        ),
    ]

    # 7. Train Model
    print(f"\nTraining model for up to {epochs} epochs (batch size = {batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # 8. Save Model, Tokenizer, and Metadata
    print(f"\nSaving trained model to {MODEL_PATH}...")
    model.save(MODEL_PATH)

    print(f"Saving tokenizer to {TOKENIZER_PATH}...")
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    final_train_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])

    best_val_loss = float(min(history.history["val_loss"]))
    best_val_acc = float(max(history.history["val_accuracy"]))

    meta = {
        "max_sequence_len": int(max_sequence_len),
        "total_words": int(vocab_size),
        "architecture": "Embedding(128) -> LSTM(128) -> LSTM(128) -> Dense(128) -> Softmax",
        "final_train_loss": round(final_train_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "final_train_accuracy": round(final_train_acc, 4),
        "final_val_accuracy": round(final_val_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "sample_prompts": [
            "hello how are",
            "the sun rises in the",
            "i am a",
            "i am going to",
            "machine learning is",
            "i want to learn",
            "the capital of france is",
            "once upon a",
            "roses are red and",
            "how are you",
            "the weather is",
            "python is a",
            "the future of technology",
            "one of the",
        ],
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {META_PATH}.")

    # 9. Evaluate on Held-Out Test Set
    test_metrics = evaluate_test_set(
        model, tokenizer, test_sents, max_sequence_len=max_sequence_len, vocab_size=vocab_size
    )

    # 10. Evaluate Benchmark Prompts
    predictor = NextWordPredictor(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        meta_path=META_PATH,
    )
    benchmark_results = evaluate_benchmark_prompts(predictor)

    eval_report = {
        "model_architecture": meta["architecture"],
        "vocab_size": vocab_size,
        "max_sequence_len": max_sequence_len,
        "train_metrics": {
            "loss": meta["final_train_loss"],
            "val_loss": meta["final_val_loss"],
            "best_val_loss": meta["best_val_loss"],
            "accuracy": meta["final_train_accuracy"],
            "val_accuracy": meta["final_val_accuracy"],
            "best_val_accuracy": meta["best_val_accuracy"],
        },
        "test_metrics": test_metrics,
        "benchmark_prompts": benchmark_results,
    }

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    print(f"Saved complete evaluation report to {EVAL_PATH}.")

    print("\n==================================================")
    print("TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================")
    return model, tokenizer, eval_report


if __name__ == "__main__":
    run_training_pipeline()
