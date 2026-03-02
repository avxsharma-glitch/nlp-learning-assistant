# 🎓 NLP Learning Assistant

A modular, upgrade-ready NLP chatbot that acts as a teaching assistant. It classifies user intent using TF-IDF + Logistic Regression, maintains short-term conversation memory, adapts response depth to the learner's level, and returns structured teaching responses via a Flask REST API.

---

## 📁 Project Structure

```
nlp-learning-assistant/
├── config.py                 ← Shared paths, thresholds, flags
├── preprocessing.py          ← NLTK normalize() (tokenise → stopwords → lemmatise)
├── model.py                  ← LanguageModelInterface + LocalSklearnModel
├── memory.py                 ← SQLite-backed ConversationMemory
├── response_engine.py        ← detect_level, build_structured_response, fallback
├── train.py                  ← Train TF-IDF + LogReg; save to artifacts/
├── app.py                    ← Flask API (/health, /chat)
├── chat.py                   ← Original CLI interface (unchanged)
├── intents.json              ← Structured intents (greeting, tfidf, logreg_nlp, fallback)
├── response_engine_tests.py  ← Unit tests for response_engine
├── requirements.txt          ← Pinned dependencies
├── .gitignore                ← Excludes artifacts/ and *.db
└── artifacts/                ← Generated at runtime (not committed)
    ├── model.pkl
    ├── vectorizer.pkl
    └── memory.db
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| NLTK | Tokenisation, stopword removal, lemmatisation |
| Scikit-learn | TF-IDF vectorisation + Logistic Regression |
| Flask | REST API |
| SQLite | Short-term conversation memory |
| JSON | Intent storage and training data |

---

## 🚀 Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This reads `intents.json`, normalises all patterns, trains TF-IDF + Logistic Regression, prints a validation classification report, and saves `artifacts/model.pkl` and `artifacts/vectorizer.pkl`.

### 3. Run the Flask API

```bash
python app.py
```

The server starts at **http://127.0.0.1:5000**.

### 4. Health Check

```bash
curl http://127.0.0.1:5000/health
# {"status": "ok"}
```

### 5. Chat Endpoint

```bash
curl -X POST http://127.0.0.1:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "explain TF-IDF", "level": "beginner"}'
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | ✅ | User's question |
| `level` | string | ❌ | `beginner` / `intermediate` / `advanced` |

**Response body**

```json
{
  "intent":     "tfidf",
  "confidence": 0.9123,
  "level":      "beginner",
  "response":   "**Answer:** TF-IDF ...\n\n**Steps:** ..."
}
```

---

## 🧠 Supported Intents

| Intent | Example Patterns |
|--------|-----------------|
| `greeting` | hello, hi, hey |
| `tfidf` | what is TF-IDF, explain TF-IDF, TF-IDF formula |
| `logreg_nlp` | logistic regression, softmax, logreg |
| `fallback` | (low-confidence catch-all) |

---

## 💡 Response Format

Every `/chat` response contains these sections (Markdown-formatted):

- **Answer** — One-sentence direct answer
- **Steps** — Numbered step-by-step explanation
- **Analogy** — Intuitive real-world comparison
- **Deeper Insight** — Only shown for `advanced` level
- **Summary** — One-line takeaway
- **Practice** — A follow-up question to test understanding

A *context nod* is prepended when the assistant has recent conversation history.

---

## 🔧 Behaviour Details

| Feature | Detail |
|---------|--------|
| Confidence threshold | `0.30` (configurable in `config.py`) — below this, fallback response is returned |
| Memory window | Last 5 interactions stored in SQLite (`artifacts/memory.db`) |
| Adaptive depth | Level auto-detected from keywords (e.g. "simple", "advanced") or explicit `level` field |
| Debug logging | Set env var `NLP_DEBUG=true` to enable DEBUG-level logs |

---

## 🧪 Running Tests

```bash
python -m unittest response_engine_tests -v
```

---

## 🔮 Upgrade Path

The `LanguageModelInterface` in `model.py` defines a clean abstraction. To swap in a transformer-based model (e.g. BERT via HuggingFace), implement a new class that inherits from `LanguageModelInterface` and override `predict_intent()`. No changes to `app.py`, `memory.py`, or `response_engine.py` are required.

---

## 📄 License

MIT License
