"""
Local Spell Checker module.
Uses Damerau-Levenshtein edit distance against a local vocabulary
to provide graceful handling for minor typos without external dependencies.
"""

import re
from collections import Counter


class LocalSpellChecker:
    def __init__(self, vocabulary=None):
        self.word_counts = Counter()
        if vocabulary:
            self.build_dictionary(vocabulary)

    def build_dictionary(self, words_or_sentences):
        """Build word frequency table from word list or list of sentences."""
        for item in words_or_sentences:
            if isinstance(item, str):
                tokens = re.findall(r"[a-z0-9']+", item.lower())
                self.word_counts.update(tokens)
            else:
                self.word_counts[str(item).lower()] += 1

    def is_known(self, word):
        """Check if a word exists in the dictionary."""
        return word.lower() in self.word_counts

    def edits1(self, word):
        """Generate all edits that are one edit away from `word`."""
        letters = "abcdefghijklmnopqrstuvwxyz0123456789'"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """Generate all edits that are two edits away from `word`."""
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def known(self, words):
        """Filter the subset of `words` that exist in dictionary."""
        return set(w for w in words if w in self.word_counts)

    def correct_word(self, word):
        """
        Suggest correction for a single word.
        Returns original word if already known or if no close candidate found.
        """
        if not word:
            return word

        w = word.lower()
        if self.is_known(w):
            return w

        # Candidates at edit distance 1
        candidates = self.known(self.edits1(w))
        if candidates:
            # Filter candidates so we don't convert words into wildly different lengths
            valid_cands = [c for c in candidates if abs(len(c) - len(w)) <= 1]
            if valid_cands:
                return max(valid_cands, key=lambda c: self.word_counts[c])

        # Candidates at edit distance 2 (only for words of length >= 4)
        if len(w) >= 4:
            candidates = self.known(self.edits2(w))
            valid_cands = [c for c in candidates if abs(len(c) - len(w)) <= 1 and len(c) >= 3]
            if valid_cands:
                return max(valid_cands, key=lambda c: self.word_counts[c])

        return w

    def correct_sentence(self, sentence):
        """
        Check and correct tokens in sentence.
        Returns (corrected_sentence, has_changed, changes_dict).
        """
        if not sentence:
            return "", False, {}

        tokens = sentence.split()
        corrected_tokens = []
        changes = {}

        for token in tokens:
            # Strip punctuation for check
            clean_token = re.sub(r"[^a-z0-9']", "", token.lower())
            if not clean_token:
                corrected_tokens.append(token)
                continue

            corrected = self.correct_word(clean_token)
            if corrected != clean_token:
                changes[clean_token] = corrected
                corrected_tokens.append(corrected)
            else:
                corrected_tokens.append(clean_token)

        corrected_str = " ".join(corrected_tokens)
        has_changed = len(changes) > 0
        return corrected_str, has_changed, changes
