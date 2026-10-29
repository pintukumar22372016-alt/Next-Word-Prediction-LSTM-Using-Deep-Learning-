"""
Comprehensive automated unit and integration tests for
Next-Word Prediction LSTM machine learning pipeline, N-Gram fallback engine, and Flask API.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import (
    clean_sentence,
    split_corpus,
    fit_tokenizer,
    generate_ngram_sequences,
    preprocess_text_for_inference,
)
from src.model_builder import build_lstm_model
from src.spell_checker import LocalSpellChecker
from src.language_model import NGramLanguageModel
from predict import NextWordPredictor
from app import app


class TestPreprocessing(unittest.TestCase):
    def test_clean_sentence(self):
        """Test text normalization, casing, extra whitespace, and punctuation handling."""
        self.assertEqual(clean_sentence("Hello World!"), "hello world")
        self.assertEqual(clean_sentence("  I   am   a   STUDENT... "), "i am a student")
        self.assertEqual(clean_sentence("# This is a comment"), "")
        self.assertEqual(clean_sentence(""), "")
        self.assertEqual(clean_sentence("It's a sunny day!"), "it's a sunny day")

    def test_split_corpus_no_leakage(self):
        """Test sentence-level split with distinct subsets and zero leakage."""
        sentences = [f"sample sentence number {i} for testing splits" for i in range(50)]
        train, val, test = split_corpus(sentences, test_size=0.2, val_size=0.2, random_state=42)

        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

        set_train = set(train)
        set_val = set(val)
        set_test = set(test)
        self.assertEqual(len(set_train.intersection(set_val)), 0)
        self.assertEqual(len(set_train.intersection(set_test)), 0)
        self.assertEqual(len(set_val.intersection(set_test)), 0)

    def test_tokenizer_and_oov(self):
        """Test tokenizer fits correctly and handles OOV token at index 1."""
        sentences = ["i am a student", "machine learning is exciting"]
        tokenizer, vocab_size = fit_tokenizer(sentences, oov_token="<oov>")

        self.assertIn("<oov>", tokenizer.word_index)
        self.assertEqual(tokenizer.word_index["<oov>"], 1)
        self.assertGreater(vocab_size, 5)

        # Out-of-vocabulary word mapping
        seq = tokenizer.texts_to_sequences(["unknown extraterrestrial vocabulary"])[0]
        self.assertEqual(seq, [1, 1, 1])


class TestSecondaryEngines(unittest.TestCase):
    def test_spell_checker_typos(self):
        """Test local spell checker corrects common typos while preserving known words."""
        vocab = ["python", "machine", "learning", "student", "today", "hello"]
        checker = LocalSpellChecker(vocab)

        self.assertEqual(checker.correct_word("pythn"), "python")
        self.assertEqual(checker.correct_word("studnt"), "student")
        self.assertEqual(checker.correct_word("today"), "today")

        corrected, has_changed, _ = checker.correct_sentence("i am a studnt")
        self.assertTrue(has_changed)
        self.assertIn("student", corrected)

    def test_ngram_fallback_language_model(self):
        """Test N-Gram model predicts contextually without crashes or duplicates."""
        sentences = [
            "the future of technology depends on innovation",
            "the future of artificial intelligence is bright",
            "today is a wonderful day",
            "hello world is a traditional program",
        ]
        ngram = NGramLanguageModel(sentences)

        cands = ngram.predict_next(["the", "future", "of"], top_k=3)
        self.assertGreaterEqual(len(cands), 2)
        words = [c["word"] for c in cands]
        self.assertIn("technology", words)
        self.assertNotIn("", words)
        self.assertEqual(len(words), len(set(words)))


class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = NextWordPredictor()

    def test_predictor_loaded_and_ready(self):
        """Verify predictor loaded weights, tokenizer, and fallback engines."""
        self.assertTrue(self.predictor.is_ready)
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.tokenizer)
        self.assertIsNotNone(self.predictor.ngram_model)
        self.assertIsNotNone(self.predictor.spell_checker)

    def test_empty_and_whitespace_inputs(self):
        """Test predictor gracefully handles empty or whitespace input."""
        res_empty = self.predictor.predict_next("")
        self.assertEqual(res_empty.get("predictions"), [])
        self.assertEqual(res_empty.get("source"), "none")

        res_spaces = self.predictor.predict_next("    ")
        self.assertEqual(res_spaces.get("predictions"), [])
        self.assertEqual(res_spaces.get("source"), "none")

    def test_one_word_and_short_inputs(self):
        """Test one-word inputs return usable suggestions."""
        for word in ["hello", "today", "python"]:
            res = self.predictor.predict_next(word, top_k=5)
            self.assertIn(res.get("source"), ["lstm", "fallback"])
            self.assertGreater(len(res.get("predictions", [])), 0)

    def test_unfamiliar_input_triggers_fallback(self):
        """Test completely unfamiliar sentence routes to fallback model."""
        res = self.predictor.predict_next("quantum cryptography utilizes photonic entanglement", top_k=5)
        self.assertGreater(len(res.get("predictions", [])), 0)
        self.assertEqual(res.get("source"), "fallback")
        self.assertIn("N-Gram", res.get("source_label"))

    def test_candidate_ranking_and_no_oov_pad(self):
        """Verify output candidates exclude index 0 (pad) and index 1 (oov)."""
        res = self.predictor.predict_next("i am a", top_k=5)
        preds = res.get("predictions", [])
        words = [p["word"] for p in preds]

        self.assertNotIn("<oov>", words)
        self.assertNotIn("", words)
        self.assertEqual(len(words), len(set(words)))  # no duplicates

    def test_generation_continuation(self):
        """Verify multi-word generation iterative rollout."""
        gen_res = self.predictor.generate_continuation("the future of", num_words=4, temperature=0.7)
        self.assertIn("completed_text", gen_res)
        self.assertGreaterEqual(len(gen_res["generated_words"]), 1)


class TestFlaskAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    def test_info_endpoint(self):
        """Test GET /api/info returns model status and architecture metadata."""
        res = self.client.get("/api/info")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("ready", data)
        self.assertIn("model_architecture", data)

    def test_predict_endpoint_response_structure(self):
        """Test POST /api/predict response structure."""
        res = self.client.post("/api/predict", json={"text": "machine learning", "top_k": 5})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("predictions", data)
        self.assertIn("seed", data)
        self.assertIn("source", data)
        self.assertIn("source_label", data)
        self.assertIn("message", data)


if __name__ == "__main__":
    unittest.main()
