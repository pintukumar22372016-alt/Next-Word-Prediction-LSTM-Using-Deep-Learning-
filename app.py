"""
Flask Web Application for Next Word Prediction & Text Completion.
Serves responsive web interface and REST API endpoints (/api/predict, /api/generate, /api/info, /api/eva"""

import os
import json
from flask import Flask, render_template, request, jsonify
from predict import NextWordPredictor

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVAL_PATH = os.path.join(MODELS_DIR, "eval_report.json")

# Instantiate single global predictor
predictor = NextWordPredictor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info", methods=["GET"])
def get_info():
    """Return model architecture, vocabulary size, and training metadata."""
    if not predictor.is_ready:
        predictor.load_artifacts()

    # Read evaluation report if available
    eval_report = {}
    if os.path.exists(EVAL_PATH):
        try:
            with open(EVAL_PATH, "r", encoding="utf-8") as f:
                eval_report = json.load(f)
        except Exception:
            pass

    test_metrics = eval_report.get("test_metrics", {})
    train_metrics = eval_report.get("train_metrics", {})

    return jsonify({
        "ready": predictor.is_ready,
        "model_architecture": predictor.meta.get("architecture", "Embedding(128) -> LSTM(128) -> LSTM(128) -> Dense(128) -> Softmax"),
        "total_words": predictor.vocab_size or (
            len(predictor.tokenizer.word_index) + 1 if predictor.tokenizer else (
                len(predictor.ngram_model.vocabulary) if predictor.ngram_model else 0
            )
        ),
        "max_sequence_len": predictor.max_sequence_len,
        "final_accuracy": train_metrics.get("accuracy", predictor.meta.get("final_train_accuracy", 0.0)),
        "final_val_accuracy": train_metrics.get("val_accuracy", predictor.meta.get("final_val_accuracy", 0.0)),
        "test_top1_accuracy": test_metrics.get("top_1_accuracy", None),
        "test_top5_accuracy": test_metrics.get("top_5_accuracy", None),
        "sample_prompts": predictor.meta.get("sample_prompts", [
            "hello how are",
            "the sun rises in the",
            "i am a",
            "i am going to",
            "machine learning is",
            "i want to learn",
            "the capital of france is",
            "once upon a",
            "roses are red and",
            "python is a",
            "the future of technology",
        ]),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict next word candidates.
    Uses trained LSTM as primary source with automatic N-Gram backoff for unfamiliar contexts.
    Returns consistent JSON with prediction source, candidate ranking, and message.
    """
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        top_k = min(max(int(data.get("top_k", 5)), 1), 10)

        # Predict using resilient NextWordPredictor
        result = predictor.predict_next(text, top_k=top_k)
        return jsonify(result), 200
    except Exception as e:
        print(f"Error in /api/predict: {e}")
        return jsonify({
            "seed": text if "text" in locals() else "",
            "corrected_text": None,
            "has_spelling_correction": False,
            "source": "error",
            "source_label": "Error",
            "message": f"Prediction error: {str(e)}",
            "predictions": [],
            "top_word": "",
            "ready": False,
        }), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Roll out multi-word text generation by iteratively querying the prediction engine.
    """
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        num_words = data.get("num_words", 8)
        temperature = data.get("temperature", 0.7)

        result = predictor.generate_continuation(text, num_words=num_words, temperature=temperature)
        return jsonify(result), 200
    except Exception as e:
        print(f"Error in /api/generate: {e}")
        return jsonify({"error": str(e), "completed_text": "", "generated_words": []}), 500


@app.route("/api/eval", methods=["GET"])
def get_eval():
    """Return latest evaluation report."""
    if os.path.exists(EVAL_PATH):
        try:
            with open(EVAL_PATH, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"message": "No evaluation report found yet. Run evaluate.py or train.py."}), 404


if __name__ == "__main__":
    predictor.load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)
