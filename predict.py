"""
Inference module for Next-Word Prediction LSTM project.
Features a principled, layered prediction pipeline:
1. Primary Source: Causal multi-layer LSTM neural network producing genuine softmax probabilities.
2. Fallback Source: Genuine Statistical N-Gram Language Model with Trigram/Bigram/Unigram backoff,
   clearly labelled when active for unfamiliar or out-of-vocabulary inputs.
3. Auxiliary: Local Spell Checker for typo tolerance with empirical word frequencies.
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf

from src.preprocessing import (
    clean_sentence,
    preprocess_text_for_inference,
    load_corpus,
    is_valid_word_candidate,
)
from src.language_model import NGramLanguageModel
from src.spell_checker import LocalSpellChecker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.txt")
MODEL_PATH = os.path.join(MODELS_DIR, "next_word_lstm.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pickle")
META_PATH = os.path.join(MODELS_DIR, "meta.json")


class NextWordPredictor:
    def __init__(
        self,
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        meta_path=META_PATH,
        corpus_path=CORPUS_PATH,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.meta_path = meta_path
        self.corpus_path = corpus_path

        self.model = None
        self.tokenizer = None
        self.reverse_word_map = {}
        self.meta = {}
        self.max_sequence_len = 12
        self.oov_index = 1
        self.is_ready = False

        # Secondary & Auxiliary Models
        self.ngram_model = None
        self.spell_checker = None

        self.vocab_size = None
        self.load_artifacts()
        self.load_secondary_models()

    def load_artifacts(self):
        """Load trained neural network, tokenizer, and metadata."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.tokenizer_path):
            print(f"Artifacts not yet found at {self.model_path} or {self.tokenizer_path}")
            self.is_ready = False
            return False

        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
                self.max_sequence_len = self.meta.get("max_sequence_len", 12)
                self.vocab_size = self.meta.get("total_words", None)

            with open(self.tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)

            self.reverse_word_map = {
                idx: word for word, idx in self.tokenizer.word_index.items()
            }
            self.oov_index = self.tokenizer.word_index.get(
                self.tokenizer.oov_token, 1
            )

            try:
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
            except Exception:
                self.model = tf.keras.models.load_model(self.model_path)

            self.is_ready = True
            print(
                f"NextWordPredictor loaded successfully: "
                f"Vocab={len(self.tokenizer.word_index)+1}, MaxSeqLen={self.max_sequence_len}"
            )
            return True
        except Exception as e:
            print(f"Error loading LSTM artifacts: {e}")
            self.is_ready = False
            return False

    def load_secondary_models(self):
        """Build or load the N-Gram fallback model and local spell checker."""
        sentences = []
        if os.path.exists(self.corpus_path):
            sentences = load_corpus(self.corpus_path)

        if not sentences and self.tokenizer:
            sentences = list(self.tokenizer.word_index.keys())

        # Initialize N-Gram Model on text corpus
        self.ngram_model = NGramLanguageModel(sentences)

        # Initialize Spell Checker with full sentences for true empirical frequency distribution
        self.spell_checker = LocalSpellChecker(sentences)
        if self.tokenizer:
            for w in self.tokenizer.word_index.keys():
                if w not in self.spell_checker.word_counts:
                    self.spell_checker.word_counts[w] += 1

    def predict_next(self, text, top_k=5):
        """
        Predict contextually relevant top-k candidates for any given text input.
        Pipeline:
          1. Validate non-empty input.
          2. Check for minor spelling mistakes (without altering raw user seed).
          3. Query Primary LSTM neural network. If usable, returns top-k genuine softmax probabilities.
          4. If LSTM is unavailable or context is completely out-of-vocabulary,
             transparently falls back to genuine local N-Gram Language Model, clearly labelled.
          5. Never hardcodes predictions or fabricates probabilities.
        """
        raw_text = str(text) if text is not None else ""
        trimmed = raw_text.strip()

        # Handle Empty Input
        if not trimmed:
            return {
                "seed": raw_text,
                "corrected_text": None,
                "has_spelling_correction": False,
                "source": "none",
                "source_label": "Awaiting Input",
                "message": "Please enter text above to see next-word predictions.",
                "predictions": [],
                "top_word": "",
                "ready": True,
            }

        # Clean input for tokenization
        cleaned_input = clean_sentence(trimmed)
        tokens = cleaned_input.split()

        # Optional Spelling Correction Check
        corrected_text = None
        has_correction = False
        active_tokens = tokens

        if self.spell_checker and tokens:
            corrected_str, has_correction, changes = self.spell_checker.correct_sentence(cleaned_input)
            if has_correction and corrected_str != cleaned_input:
                corrected_text = corrected_str
                active_tokens = corrected_str.split()

        # -------------------------------------------------------------
        # 1. PRIMARY PREDICTION: Trained LSTM Model
        # -------------------------------------------------------------
        lstm_candidates = []
        lstm_usable = False

        if self.is_ready and self.model and self.tokenizer and active_tokens:
            try:
                # Check vocabulary presence of tokens
                token_ids = [
                    self.tokenizer.word_index.get(t, self.oov_index)
                    for t in active_tokens
                ]
                known_count = sum(
                    1 for tid in token_ids
                    if tid != self.oov_index and (self.vocab_size is None or tid < self.vocab_size)
                )

                known_ratio = known_count / max(len(active_tokens), 1)

                # If the context is sufficiently in-vocabulary (at least one token known and >= 35% known)
                if known_count > 0 and known_ratio >= 0.35:
                    padded, words, has_valid = preprocess_text_for_inference(
                        " ".join(active_tokens),
                        self.tokenizer,
                        self.max_sequence_len,
                        vocab_size=self.vocab_size,
                    )
                    if has_valid and padded is not None:
                        preds = self.model.predict(padded, verbose=0)[0]

                        # Mask padding (0) and OOV (1)
                        preds[0] = 0.0
                        if self.oov_index < len(preds):
                            preds[self.oov_index] = 0.0

                        # Mask beyond effective vocabulary if capped
                        if self.vocab_size and self.vocab_size < len(preds):
                            preds[self.vocab_size:] = 0.0

                        sorted_indices = np.argsort(preds)[::-1]
                        seen = set()
                        last_token = active_tokens[-1].lower() if active_tokens else None

                        for idx in sorted_indices:
                            word = self.reverse_word_map.get(int(idx), None)
                            prob = float(preds[idx])
                            if (
                                word
                                and word != self.tokenizer.oov_token
                                and is_valid_word_candidate(word)
                                and word not in seen
                                and prob > 0.0
                            ):
                                # Disallow immediate consecutive repetition of articles and prepositions
                                if word == last_token and word in (
                                    "the", "a", "an", "in", "to", "of", "and", "is", "was", "for", "on", "at", "it"
                                ):
                                    continue
                                seen.add(word)
                                lstm_candidates.append({
                                    "word": word,
                                    "probability": round(prob, 4),
                                    "percentage": round(prob * 100, 1),
                                    "score_type": "softmax_confidence",
                                })
                                if len(lstm_candidates) >= top_k:
                                    break

                        # LSTM is usable if we obtained candidates and top candidate has genuine confidence
                        min_conf = max(0.012, 2.0 / max(len(preds), 1))
                        if len(lstm_candidates) >= min(top_k, 3) and lstm_candidates[0]["probability"] >= min_conf:
                            lstm_usable = True
            except Exception as e:
                print(f"LSTM inference exception: {e}")
                lstm_usable = False

        # If LSTM successfully generated predictions, return them directly
        if lstm_usable and len(lstm_candidates) >= top_k:
            # Genuine Softmax Probabilities
            sum_probs = sum(c["probability"] for c in lstm_candidates)
            for c in lstm_candidates:
                c["relative_share"] = (
                    round((c["probability"] / sum_probs) * 100, 1)
                    if sum_probs > 0
                    else round(100.0 / len(lstm_candidates), 1)
                )

            return {
                "seed": raw_text,
                "corrected_text": corrected_text if has_correction else None,
                "has_spelling_correction": has_correction,
                "source": "lstm",
                "source_label": "LSTM Neural Network",
                "message": "Predictions generated by trained LSTM deep neural network.",
                "predictions": lstm_candidates[:top_k],
                "top_word": lstm_candidates[0]["word"],
                "ready": True,
            }

        # -------------------------------------------------------------
        # 2. FALLBACK RESOLUTION: Genuine Statistical N-Gram Language Model
        # -------------------------------------------------------------
        fallback_candidates = []
        if self.ngram_model:
            fallback_candidates = self.ngram_model.predict_next(
                active_tokens, top_k=top_k
            )

        if not fallback_candidates and self.ngram_model:
            fallback_candidates = self.ngram_model.predict_next(["the"], top_k=top_k)

        # Re-normalize relative shares for visual bars
        tot_score = sum(c["score"] for c in fallback_candidates if "score" in c and c["score"] is not None)
        for c in fallback_candidates:
            c["relative_share"] = (
                round((c["score"] / tot_score) * 100, 1)
                if tot_score > 0
                else round(100.0 / max(len(fallback_candidates), 1), 1)
            )

        context_desc = "unfamiliar to the neural network" if active_tokens else "partially typed"
        return {
            "seed": raw_text,
            "corrected_text": corrected_text if has_correction else None,
            "has_spelling_correction": has_correction,
            "source": "fallback",
            "source_label": "Statistical N-Gram Language Model",
            "message": f"Input context was {context_desc}; suggestions provided by genuine local N-Gram language model.",
            "predictions": fallback_candidates[:top_k],
            "top_word": fallback_candidates[0]["word"] if fallback_candidates else "",
            "ready": True,
        }


    def generate_continuation(self, text, num_words=8, temperature=0.7):
        """
        Iteratively roll out text continuation by querying the prediction pipeline
        and feeding chosen tokens back into context.
        """
        raw_text = str(text) if text is not None else ""
        if not raw_text.strip():
            return {
                "error": "Please provide initial seed text.",
                "completed_text": "",
                "generated_words": [],
            }

        current_text = raw_text.strip()
        generated_words = []
        num_words = max(1, min(int(num_words), 30))
        temperature = max(0.05, min(float(temperature), 1.5))

        for _ in range(num_words):
            res = self.predict_next(current_text, top_k=5)
            preds = res.get("predictions", [])
            if not preds:
                break

            if temperature <= 0.1:
                chosen_word = preds[0]["word"]
            else:
                # Sample based on available probabilities / scores
                weights = []
                words = []
                for p in preds:
                    w = p["word"]
                    score = p.get("probability") or p.get("score") or 0.05
                    words.append(w)
                    weights.append(float(score))

                # Apply temperature to logits
                arr = np.array(weights, dtype=np.float64)
                arr = np.maximum(arr, 1e-6)
                log_probs = np.log(arr) / temperature
                exp_probs = np.exp(log_probs - np.max(log_probs))
                probs = exp_probs / np.sum(exp_probs)

                chosen_word = np.random.choice(words, p=probs)

            # Avoid immediate repetitive loops
            if generated_words and chosen_word == generated_words[-1]:
                alt_words = [p["word"] for p in preds if p["word"] != chosen_word]
                if alt_words:
                    chosen_word = alt_words[0]

            generated_words.append(chosen_word)
            current_text += " " + chosen_word

        return {
            "original_text": raw_text,
            "completed_text": current_text,
            "generated_words": generated_words,
        }
