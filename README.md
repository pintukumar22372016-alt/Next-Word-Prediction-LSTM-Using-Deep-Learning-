# 🔮 NeuroText: Next Word Prediction & Text Completion with LSTM

An end-to-end Deep Learning Natural Language Processing (NLP) web application that predicts the next most probable words in real-time and generates multi-word continuations using a causal **Long Short-Term Memory (LSTM)** neural network trained on a clean, balanced multi-domain English corpus.

---

## 🚀 Key Features

- **⚡ Real-Time Next Word Prediction:** Keystroke-debounced inference computing the top 5 next word candidates with genuine Softmax confidence scores.
- **✨ Press TAB / 1-Click Autocomplete:** Instantly complete sentences with the highest probability token.
- **📜 Multi-Word Text Generation:** Recurrent roll-out generation with adjustable **token length** and **temperature (creativity)** controls.
- **📚 Diverse Balanced Corpus:** High-quality English sentences spanning Education, Technology & Programming, Work & Careers, Everyday Conversation, Idioms, and Knowledge.
- **🛡️ OOV Token Handling:** Out-of-vocabulary words are mapped to `<oov>` token and excluded from candidate word predictions.
- **📊 Repeatable Evaluation:** Automated evaluation script measuring Top-1 and Top-5 accuracy on a held-out test set with zero n-gram leakage.
- **💎 Modern Glassmorphism UI:** Built with dark mode aesthetics, animated probability meters, responsive grid, and pure vanilla CSS.

---

## 🧠 Deep Learning Architecture

## 🧠 Deep Learning Architecture & Pipeline

```
Input Prompt (Raw user text: e.g. "i am going to")
      │
      ▼
┌──────────────────────────────────────────────┐
│  Text Normalization & Local Spell Checker    │
│  - Lowercasing, contraction preservation     │
│  - Typo correction (e.g., "thiss" -> "this") │
└──────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────┐
│  Tokenizer (Bounded Top 3,500 Vocabulary)    │
│  - Out-of-vocabulary mapping: <oov> (idx 1)  │
│  - Padding token: <pad> (idx 0)              │
└──────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────┐
│  Pre-Padding (maxlen = 12, left-truncating)  │
└──────────────────────────────────────────────┘
      │
      ├──> Known Token Ratio Check (>= 35% known)
      │         │
      │         ▼
      │   ┌──────────────────────────────────────────────┐
      │   │  Embedding Layer (dim = 128)                 │
      │   └──────────────────────────────────────────────┘
      │         │
      │         ▼
      │   ┌──────────────────────────────────────────────┐
      │   │  Causal LSTM Layer 1 (128 units, seq=True)   │
      │   └──────────────────────────────────────────────┘
      │         │  Dropout (0.15)
      │         ▼
      │   ┌──────────────────────────────────────────────┐
      │   │  Causal LSTM Layer 2 (128 units)             │
      │   └──────────────────────────────────────────────┘
      │         │  Dropout (0.15)
      │         ▼
      │   ┌──────────────────────────────────────────────┐
      │   │  Dense Projection (128 units, ReLU)          │
      │   └──────────────────────────────────────────────┘
      │         │
      │         ▼
      │   ┌──────────────────────────────────────────────┐
      │   │  Softmax Output Layer (over 3,500 classes)   │
      │   └──────────────────────────────────────────────┘
      │         │
      │         ▼
      │   Filter <pad> (0), <oov> (1) & invalid tokens
      │   ──> Top-K Candidate Predictions (Source: LSTM)
      │
      └──> (If unfamiliar / OOV ratio < 35%)
                │
                ▼
          ┌──────────────────────────────────────────────┐
          │  Statistical N-Gram Language Model           │
          │  - Trigram Context: (token[-2], token[-1])   │
          │  - Bigram Context Backoff: (token[-1])       │
          │  - Corpus Unigram Frequency Backoff          │
          │  ──> Top-K Predictions (Source: Fallback N-Gram)
          └──────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
NextWordPrediction-LSTM_Deep_Learning_Project/
├── data/
│   ├── corpus.txt                   # Curated 3,056-sentence corpus (NLTK Brown + Webtext + modern)
│   ├── curate_corpus.py             # Corpus compilation and deduplication script
│   └── test_sentences.json          # Cached held-out test sentences (zero leakage)
├── models/
│   ├── next_word_lstm.keras         # Trained BiLSTM+LSTM model weights
│   ├── tokenizer.pickle             # Fitted Keras Tokenizer
│   ├── meta.json                    # Model hyperparameters & training stats
│   └── eval_report.json             # Test set metrics & 23-domain benchmark results
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # Cleaning, sentence-level split, bounded tokens, padding
│   ├── model_builder.py             # Causal BiLSTM + LSTM neural architecture
│   ├── language_model.py            # Local Statistical N-Gram Language Model fallback
│   └── spell_checker.py             # Frequency-based local spelling corrector
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py             # 13 automated unit & integration tests
│   └── test_feature_inputs.py       # 9 feature verification input cases
├── static/
│   ├── css/
│   │   └── style.css                # Glassmorphic responsive dark mode UI
│   └── js/
│       └── app.js                   # Debounced autocomplete, ghost preview & generation
├── templates/
│   └── index.html                   # Accessible HTML5 web app interface
├── predict.py                       # Layered NextWordPredictor inference engine
├── evaluate.py                      # Held-out test evaluation & 23-prompt benchmark
├── train.py                         # Complete model training & validation pipeline
├── app.py                           # Flask backend REST API server
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

---

## 🛠️ Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🏋️ Retraining the Model

To retrain the neural network on `data/corpus.txt`:
```bash
python train.py
```
This script will:
1. Load `data/corpus.txt` and perform leak-free sentence-level train/val/test splits (80/10/10).
2. Fit the tokenizer on training sentences only with vocabulary capped at 3,500 tokens.
3. Train the causal BiLSTM+LSTM architecture with `EarlyStopping` monitoring validation accuracy (`patience=5`, `restore_best_weights=True`) and `ReduceLROnPlateau`.
4. Measure Top-1 and Top-5 accuracy on the held-out test set.
5. Export model weights, tokenizer, metadata, and evaluation report.

---

## 🧪 Running Automated Tests

Run the full test suites with:
```bash
# 1. Complete unit & integration test suite (13 tests)
python -m unittest discover -s tests -v

# 2. Broad feature input verification suite (9 test cases)
python tests/test_feature_inputs.py

# 3. Model evaluation & benchmark suite
python evaluate.py
```

---

## 📊 Evaluation & Benchmark Results

### Held-Out Test Set Metrics (Zero N-Gram Leakage):
- **Total Held-Out Test Sequences:** 2,627
- **Test Cross-Entropy Loss:** 6.7094
- **Top-1 Categorical Accuracy:** 9.48%
- **Top-5 Categorical Accuracy:** 20.94%

### Sample Benchmark Prompts Across Domains:
| Input Prompt | Engine Source | Top-5 Model Predictions |
| :--- | :--- | :--- |
| `i am going to` | LSTM Neural Network | `the` (16.1%), `a` (3.8%), `be` (3.8%), `this` (1.7%), `his` (1.7%) |
| `thank you for` | LSTM Neural Network | `the` (15.5%), `a` (6.2%), `be` (3.9%), `this` (1.8%), `have` (1.7%) |
| `python is a` | LSTM Neural Network | `time` (1.2%), `new` (0.9%), `other` (0.7%), `bit` (0.7%), `most` (0.6%) |
| `deep neural networks` | LSTM Neural Network | `of` (14.7%), `and` (10.6%), `is` (5.3%), `for` (4.6%), `in` (4.4%) |
| `the capital of france is` | N-Gram Fallback | `paris` (100.0%), `a` (10.8%), `the` (6.7%), `not` (3.6%), `that` (3.4%) |
| `reading books regularly` | N-Gram Fallback | `expands` (50.0%), `sharpens` (50.0%), `the` (5.0%), `to` (2.4%), `a` (2.2%) |
| `once upon a` | LSTM Neural Network | `time` (1.1%), `new` (0.9%), `other` (0.7%), `bit` (0.7%), `most` (0.6%) |
| `astronomers observed...` | N-Gram Fallback | `the` (5.0%), `to` (2.4%), `a` (2.2%), `of` (2.1%), `and` (2.0%) |
