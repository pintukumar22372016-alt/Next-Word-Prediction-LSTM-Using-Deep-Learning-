"""
Verification script for the new feature: Always Provide Next-Word Suggestions.
Tests all required inputs:
1. i am a
2. i am going to
3. hello
4. python is
5. machine learning
6. today
7. the future of
8. An unfamiliar sentence
9. Empty input
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from predict import NextWordPredictor

TEST_INPUTS = [
    "hello how are",
    "the sun rises in the",
    "i am a",
    "i am going to",
    "hello",
    "python is a",
    "python is",
    "machine learning is",
    "machine learning",
    "today",
    "once upon a",
    "the future of technology",
    "the future of",
    "astronomers observed an unfamiliar distant celestial galaxy",
    "",
]


def test_inputs():
    predictor = NextWordPredictor()
    print("=" * 60)
    print("TESTING USER-REQUESTED FEATURE INPUTS")
    print("=" * 60)

    for text in TEST_INPUTS:
        res = predictor.predict_next(text, top_k=5)
        source = res.get("source")
        source_label = res.get("source_label")
        top_word = res.get("top_word")
        cands = res.get("predictions", [])
        msg = res.get("message")
        has_spelling = res.get("has_spelling_correction")
        corrected = res.get("corrected_text")

        print(f"\n[Input]: \"{text}\"")
        print(f"  Source      : {source} ({source_label})")
        print(f"  Top Word    : '{top_word}'")
        if has_spelling:
            print(f"  Corrected   : \"{corrected}\"")
        cand_str = ", ".join([f"'{c['word']}' ({c['percentage']}%)" for c in cands])
        print(f"  Candidates  : [{cand_str}]")
        print(f"  Message     : {msg}")

        # Assertions
        if text.strip():
            assert len(cands) > 0, f"Expected suggestions for non-empty input '{text}'"
            words = [c["word"] for c in cands]
            assert len(words) == len(set(words)), f"Duplicate words found in {words}"
            assert "<oov>" not in words, "OOV marker leaked into suggestions"
            assert "" not in words, "Empty string found in suggestions"
        else:
            assert len(cands) == 0, "Expected empty candidates for empty input"
            assert source == "none", "Expected source 'none' for empty input"

    print("\n" + "=" * 60)
    print("ALL INPUT TESTS PASSED PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_inputs()
