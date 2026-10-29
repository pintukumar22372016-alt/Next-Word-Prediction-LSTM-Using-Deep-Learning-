"""
Preprocessing module for Next-Word Prediction LSTM project.
Handles text cleaning, corpus loading, train/val/test splitting,
tokenization with explicit OOV handling, and causal n-gram sequence generation.
"""

import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_sentence(text):
    """Clean a single sentence string: lowercases, normalizes quotes, strips punctuation."""
    if not text:
        return ""
    text = str(text).lower().strip()
    # Remove markdown / comments
    if text.startswith("#"):
        return ""
    # Normalize curly apostrophes / quotes
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace('"', " ").replace("“", " ").replace("”", " ")
    # Strip non-alphanumeric characters (keep words, spaces, and apostrophes for contractions)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_corpus(file_path):
    """
    Load raw text corpus, clean sentences, and deduplicate near-duplicates.
    Returns list of clean sentences.
    """
    sentences = []
    seen = set()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            cleaned = clean_sentence(line)
            # Minimum sentence length to ensure meaningful n-grams
            if cleaned and len(cleaned.split()) >= 3:
                if cleaned not in seen:
                    seen.add(cleaned)
                    sentences.append(cleaned)

    print(f"Loaded {len(sentences)} unique, clean sentences from {file_path}.")
    return sentences


def split_corpus(sentences, test_size=0.10, val_size=0.10, random_state=42):
    """
    Split at sentence-level to prevent n-gram data leakage between
    training, validation, and test datasets.
    """
    train_val, test_sentences = train_test_split(
        sentences, test_size=test_size, random_state=random_state
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    train_sentences, val_sentences = train_test_split(
        train_val, test_size=adjusted_val_size, random_state=random_state
    )

    print(
        f"Corpus split: Train={len(train_sentences)}, "
        f"Validation={len(val_sentences)}, Test={len(test_sentences)}"
    )
    return train_sentences, val_sentences, test_sentences


def fit_tokenizer(sentences, oov_token="<oov>", num_words=3500):
    """
    Fit Keras Tokenizer on sentences with an explicit OOV token.
    Index 0 is reserved for padding, index 1 for OOV token.
    num_words caps the effective vocabulary to top frequent words.
    """
    tokenizer = Tokenizer(
        num_words=num_words,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        oov_token=oov_token,
        lower=True,
    )
    tokenizer.fit_on_texts(sentences)
    if num_words:
        vocab_size = min(len(tokenizer.word_index) + 1, num_words)
    else:
        vocab_size = len(tokenizer.word_index) + 1
    print(f"Tokenizer fitted. Vocabulary size: {vocab_size} (including pad & oov)")
    return tokenizer, vocab_size


def generate_ngram_sequences(
    sentences,
    tokenizer,
    max_sequence_len=12,
    is_training=False,
    oov_drop_rate=0.015,
    vocab_size=None,
):
    """
    Generate n-gram sequences from sentences using fitted tokenizer.
    Each prefix [w1, ..., wt] is paired with target wt+1.
    Pre-pads X to (max_sequence_len - 1).
    Skips target tokens that are OOV to prevent the model from learning to predict OOV markers.
    Optionally applies small word dropout to input tokens during training to teach the OOV embedding.
    Ensures all target and prefix tokens respect vocab_size bounds.
    """
    oov_index = tokenizer.word_index.get(tokenizer.oov_token, 1)
    effective_vocab = vocab_size or (len(tokenizer.word_index) + 1)
    input_sequences = []
    targets = []

    for sentence in sentences:
        token_list = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(token_list)):
            prefix = token_list[:i]
            target = token_list[i]

            # Targets should always be valid in-vocab words (not padding 0 or OOV 1 or beyond vocab)
            if target <= 1 or target >= effective_vocab:
                continue

            # Bound prefix tokens within [0, effective_vocab - 1]
            bounded_prefix = [
                oov_index if tid >= effective_vocab else tid
                for tid in prefix
            ]

            # Word dropout on training inputs to train the OOV embedding representation
            if is_training and oov_drop_rate > 0.0:
                bounded_prefix = [
                    oov_index if (np.random.rand() < oov_drop_rate and tid != 0) else tid
                    for tid in bounded_prefix
                ]

            input_sequences.append(bounded_prefix)
            targets.append(target)

    if not input_sequences:
        return (
            np.empty((0, max_sequence_len - 1), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    # Pad sequences to max_sequence_len - 1
    X = pad_sequences(
        input_sequences,
        maxlen=max_sequence_len - 1,
        padding="pre",
        truncating="pre",
    )
    y = np.array(targets, dtype=np.int32)

    return X, y


def preprocess_text_for_inference(text, tokenizer, max_sequence_len=12, vocab_size=None):
    """
    Preprocess an input text string for model prediction.
    Cleans text, tokenizes, bounds token IDs to vocab_size, and pre-pads to (max_sequence_len - 1).
    Returns (padded_array, cleaned_tokens, has_valid_tokens).
    """
    cleaned = clean_sentence(text)
    if not cleaned:
        return None, [], False

    token_list = tokenizer.texts_to_sequences([cleaned])[0]
    if not token_list:
        return None, [], False

    oov_index = tokenizer.word_index.get(tokenizer.oov_token, 1)
    effective_vocab = vocab_size or (len(tokenizer.word_index) + 1)

    # Bound tokens within vocabulary range
    bounded_tokens = [
        oov_index if tid >= effective_vocab else tid
        for tid in token_list
    ]

    # Pad pre to max_sequence_len - 1, truncating from the left (keeping the most recent tokens)
    padded = pad_sequences(
        [bounded_tokens],
        maxlen=max_sequence_len - 1,
        padding="pre",
        truncating="pre",
    )
    return padded, cleaned.split(), True


def is_valid_word_candidate(word):
    """
    Check if a candidate word is a valid next-word prediction.
    Excludes special tokens, isolated punctuation, digits, contractions fragments,
    and single-character tokens except 'a' and 'i'.
    """
    if not word or not isinstance(word, str):
        return False
    w = word.strip().lower()
    if w in ("<oov>", "<pad>", "'", '"', "", "s", "t", "d", "ll", "ve", "re", "m"):
        return False
    if w.isdigit():
        return False
    if len(w) == 1 and w not in ("a", "i"):
        return False
    if not any(c.isalpha() for c in w):
        return False
    return True


