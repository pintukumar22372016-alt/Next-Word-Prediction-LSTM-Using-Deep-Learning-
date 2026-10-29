"""
Statistical N-Gram Language Model module.
Provides a reliable fallback prediction engine using genuine unigram, bigram,
trigram, 4-gram, and 5-gram frequencies derived from the local text corpus.
Ensures context-aware suggestions for inputs outside or poorly handled by the neural network.
"""

import re
from collections import Counter, defaultdict
from src.preprocessing import is_valid_word_candidate


class NGramLanguageModel:
    def __init__(self, sentences=None):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.fourgram_counts = defaultdict(Counter)
        self.fivegram_counts = defaultdict(Counter)
        self.total_tokens = 0
        self.vocabulary = set()

        if sentences:
            self.train(sentences)

    def train(self, sentences):
        """Build unigram, bigram, trigram, 4-gram, and 5-gram frequency tables from sentences."""
        for sentence in sentences:
            tokens = re.findall(r"[a-z0-9']+", sentence.lower())
            if not tokens:
                continue

            self.total_tokens += len(tokens)
            for i, token in enumerate(tokens):
                self.unigram_counts[token] += 1
                self.vocabulary.add(token)

                # Bigrams: (w1) -> w2
                if i > 0:
                    prev = tokens[i - 1]
                    self.bigram_counts[prev][token] += 1

                # Trigrams: (w1, w2) -> w3
                if i > 1:
                    prev2 = (tokens[i - 2], tokens[i - 1])
                    self.trigram_counts[prev2][token] += 1

                # 4-Grams: (w1, w2, w3) -> w4
                if i > 2:
                    prev3 = (tokens[i - 3], tokens[i - 2], tokens[i - 1])
                    self.fourgram_counts[prev3][token] += 1

                # 5-Grams: (w1, w2, w3, w4) -> w5
                if i > 3:
                    prev4 = (tokens[i - 4], tokens[i - 3], tokens[i - 2], tokens[i - 1])
                    self.fivegram_counts[prev4][token] += 1

        print(
            f"NGramLanguageModel trained: {len(self.vocabulary)} unique words, "
            f"{len(self.bigram_counts)} bigrams, {len(self.trigram_counts)} trigrams, "
            f"{len(self.fourgram_counts)} 4-grams, {len(self.fivegram_counts)} 5-grams."
        )

    def predict_next(self, tokens, top_k=5, exclude_words=None):
        """
        Predict next words using 5-Gram -> 4-Gram -> Trigram -> Bigram -> Unigram backoff.
        Guarantees distinct candidates, strictly sorted by genuine frequency score,
        with no padding/oov markers and no duplicates.
        """
        if exclude_words is None:
            seen_words = set()
        else:
            seen_words = set(exclude_words)
        seen_words.add("<oov>")
        seen_words.add("")

        # Clean input tokens
        clean_tokens = [re.sub(r"[^a-z0-9']", "", t.lower()) for t in tokens if t]
        clean_tokens = [t for t in clean_tokens if t and t != "<oov>"]

        candidates = []

        # 1. 5-Gram Context: (tokens[-4], tokens[-3], tokens[-2], tokens[-1])
        if len(candidates) < top_k and len(clean_tokens) >= 4:
            ctx5 = (clean_tokens[-4], clean_tokens[-3], clean_tokens[-2], clean_tokens[-1])
            if ctx5 in self.fivegram_counts:
                counter = self.fivegram_counts[ctx5]
                total_ctx = max(sum(counter.values()), 1)
                for word, count in counter.most_common():
                    if word not in seen_words and is_valid_word_candidate(word):
                        seen_words.add(word)
                        score = count / total_ctx
                        candidates.append({
                            "word": word,
                            "probability": None,
                            "score": round(score, 4),
                            "percentage": round(score * 100, 1),
                            "score_type": "5gram_frequency",
                            "context_used": f"5gram: {ctx5}",
                        })
                        if len(candidates) >= top_k:
                            break

        # 2. 4-Gram Context: (tokens[-3], tokens[-2], tokens[-1])
        if len(candidates) < top_k and len(clean_tokens) >= 3:
            ctx4 = (clean_tokens[-3], clean_tokens[-2], clean_tokens[-1])
            if ctx4 in self.fourgram_counts:
                counter = self.fourgram_counts[ctx4]
                total_ctx = max(sum(counter.values()), 1)
                for word, count in counter.most_common():
                    if word not in seen_words and is_valid_word_candidate(word):
                        seen_words.add(word)
                        score = count / total_ctx
                        candidates.append({
                            "word": word,
                            "probability": None,
                            "score": round(score, 4),
                            "percentage": round(score * 100, 1),
                            "score_type": "4gram_frequency",
                            "context_used": f"4gram: {ctx4}",
                        })
                        if len(candidates) >= top_k:
                            break

        # 3. Trigram Context: (tokens[-2], tokens[-1])
        if len(candidates) < top_k and len(clean_tokens) >= 2:
            trigram_ctx = (clean_tokens[-2], clean_tokens[-1])
            if trigram_ctx in self.trigram_counts:
                counter = self.trigram_counts[trigram_ctx]
                total_ctx = max(sum(counter.values()), 1)
                for word, count in counter.most_common():
                    if word not in seen_words and is_valid_word_candidate(word):
                        seen_words.add(word)
                        score = count / total_ctx
                        candidates.append({
                            "word": word,
                            "probability": None,
                            "score": round(score, 4),
                            "percentage": round(score * 100, 1),
                            "score_type": "trigram_frequency",
                            "context_used": f"trigram: ('{clean_tokens[-2]}', '{clean_tokens[-1]}')",
                        })
                        if len(candidates) >= top_k:
                            break

        # 4. Bigram Context Backoff: (tokens[-1])
        if len(candidates) < top_k and len(clean_tokens) >= 1:
            bigram_ctx = clean_tokens[-1]
            if bigram_ctx in self.bigram_counts:
                counter = self.bigram_counts[bigram_ctx]
                total_ctx = max(sum(counter.values()), 1)
                for word, count in counter.most_common():
                    if word not in seen_words and is_valid_word_candidate(word):
                        seen_words.add(word)
                        score = count / total_ctx
                        candidates.append({
                            "word": word,
                            "probability": None,
                            "score": round(score, 4),
                            "percentage": round(score * 100, 1),
                            "score_type": "bigram_frequency",
                            "context_used": f"bigram: ('{clean_tokens[-1]}')",
                        })
                        if len(candidates) >= top_k:
                            break

        # 5. Unigram Frequency Backoff (sorted by genuine corpus frequency)
        if len(candidates) < top_k and self.unigram_counts:
            unigram_candidates = []
            for word, count in self.unigram_counts.most_common():
                if word not in seen_words and is_valid_word_candidate(word):
                    score = count / max(self.total_tokens, 1)
                    unigram_candidates.append((word, score))
                    if len(unigram_candidates) >= (top_k - len(candidates)):
                        break

            unigram_candidates.sort(key=lambda x: x[1], reverse=True)

            for word, score in unigram_candidates:
                seen_words.add(word)
                candidates.append({
                    "word": word,
                    "probability": None,
                    "score": round(score, 4),
                    "percentage": round(score * 100, 1),
                    "score_type": "unigram_frequency",
                    "context_used": "unigram_corpus_frequency",
                })
                if len(candidates) >= top_k:
                    break

        # Normalize relative share among the selected candidates for UI visual bars
        total_score = sum(c["score"] for c in candidates if c.get("score") is not None)
        for c in candidates:
            if total_score > 0:
                c["relative_share"] = round((c["score"] / total_score) * 100, 1)
            else:
                c["relative_share"] = round(100.0 / max(len(candidates), 1), 1)

        return candidates[:top_k]
