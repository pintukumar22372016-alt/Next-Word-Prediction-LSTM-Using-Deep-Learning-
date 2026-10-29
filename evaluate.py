"""
Evaluation module for Next-Word Prediction LSTM project.
Measures Top-1 and Top-5 accuracy and loss on held-out test set,
evaluates core benchmark prompts across multiple domains, and outputs structured evaluation reports.
"""

import os
import json
import numpy as np
import tensorflow as tf
from src.preprocessing import generate_ngram_sequences, load_corpus

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
EVAL_REPORT_PATH = os.path.join(MODELS_DIR, "eval_report.json")
TEST_CACHE_PATH = os.path.join(DATA_DIR, "test_sentences.json")

# Multi-domain benchmark evaluation suite covering conversational, educational,
# technical, descriptive, idiomatic, and unfamiliar inputs
BENCHMARK_PROMPTS = [
    # Conversational & Daily Life
    "hello how are",
    "how are you",
    "could you please",
    "it is always a",
    "thank you for",
    "let us work",
    "i am going to",
    # Technical & Programming
    "python is a",
    "machine learning is",
    "deep neural networks",
    "software engineers write",
    "the future of",
    "the future of technology",
    # Educational & Academic
    "i want to learn",
    "reading books regularly",
    "the university provides",
    "in order to succeed you",
    # Descriptive & Common Expressions
    "the sun rises in the",
    "the weather is",
    "once upon a",
    "roses are red and",
    "the capital of france is",
    "one of the",
    # Unfamiliar / Out-of-vocabulary inputs
    "astronomers observed an unfamiliar distant celestial galaxy",
    "quantum cryptography utilizes photonic entanglement",
]


def evaluate_test_set(model, tokenizer, test_sentences, max_sequence_len=12, batch_size=64, vocab_size=None):
    """
    Measure next-word Top-1 and Top-5 categorical accuracy on held-out test sequences.
    """
    effective_vocab = vocab_size or (len(tokenizer.word_index) + 1)
    X_test, y_test = generate_ngram_sequences(
        test_sentences,
        tokenizer,
        max_sequence_len=max_sequence_len,
        is_training=False,
        vocab_size=effective_vocab,
    )

    if len(X_test) == 0:
        print("Warning: No test sequences found!")
        return {"top_1_accuracy": 0.0, "top_5_accuracy": 0.0, "test_sequences": 0, "test_loss": None}

    print(f"Evaluating model on {len(X_test)} held-out test sequences...")
    predictions = model.predict(X_test, batch_size=batch_size, verbose=0)

    # Compute loss if valid targets
    try:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        eval_loss = float(loss_fn(y_test, predictions).numpy())
    except Exception:
        eval_loss = None

    top_1_correct = 0
    top_5_correct = 0

    for i in range(len(y_test)):
        target = y_test[i]
        top_indices = np.argsort(predictions[i])[::-1]

        if top_indices[0] == target:
            top_1_correct += 1
        if target in top_indices[:5]:
            top_5_correct += 1

    top_1_acc = round(top_1_correct / len(y_test), 4)
    top_5_acc = round(top_5_correct / len(y_test), 4)

    print("==================================================")
    print("HELD-OUT TEST SET EVALUATION RESULTS")
    print("==================================================")
    print(f"Total Test Sequences : {len(X_test)}")
    if eval_loss is not None:
        print(f"Test Loss (CrossEnt) : {eval_loss:.4f}")
    print(f"Top-1 Accuracy       : {top_1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy       : {top_5_acc * 100:.2f}%")
    print("==================================================")

    return {
        "test_sequences": int(len(X_test)),
        "test_loss": round(eval_loss, 4) if eval_loss is not None else None,
        "top_1_accuracy": top_1_acc,
        "top_5_accuracy": top_5_acc,
    }


def evaluate_benchmark_prompts(predictor, prompts=BENCHMARK_PROMPTS):
    """
    Evaluate candidate predictions for core benchmark inputs.
    """
    results = {}
    print("\n==================================================")
    print(f"EVALUATING {len(prompts)} BENCHMARK PROMPTS")
    print("==================================================")

    for prompt in prompts:
        res = predictor.predict_next(prompt, top_k=5)
        preds = res.get("predictions", [])
        results[prompt] = {
            "source": res.get("source"),
            "source_label": res.get("source_label"),
            "top_candidates": preds,
        }

        cands_str = ", ".join(
            [f"'{p['word']}' ({p.get('percentage', 0)}%)" for p in preds]
        )
        print(f"Input: \"{prompt}\" [{res.get('source_label')}]")
        print(f"  Top Candidates: {cands_str if preds else '[No prediction]'}")
        print("--------------------------------------------------")

    return results


def run_full_evaluation(test_sentences=None):
    """Run full evaluation suite and write models/eval_report.json."""
    from predict import NextWordPredictor

    predictor = NextWordPredictor()
    if not predictor.is_ready:
        print("Model artifacts not available. Please train the model first.")
        return None

    # Load cached test sentences if none provided
    if (test_sentences is None or len(test_sentences) == 0) and os.path.exists(TEST_CACHE_PATH):
        try:
            with open(TEST_CACHE_PATH, "r", encoding="utf-8") as f:
                test_sentences = json.load(f)
        except Exception as e:
            print(f"Could not load cached test sentences: {e}")

    report = {
        "model_architecture": predictor.meta.get(
            "architecture", "BiLSTM(96) + LSTM(128) + Dense(128) + Softmax"
        ),
        "vocab_size": predictor.vocab_size or (len(predictor.tokenizer.word_index) + 1 if predictor.tokenizer else 0),
        "max_sequence_len": predictor.max_sequence_len,
    }

    if test_sentences is not None and len(test_sentences) > 0:
        test_metrics = evaluate_test_set(
            predictor.model,
            predictor.tokenizer,
            test_sentences,
            max_sequence_len=predictor.max_sequence_len,
            vocab_size=predictor.vocab_size,
        )
        report["test_metrics"] = test_metrics

    prompt_results = evaluate_benchmark_prompts(predictor)
    report["benchmark_prompts"] = prompt_results

    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved evaluation report to {EVAL_REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_full_evaluation()
