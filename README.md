# 🎓 NLP Learning Assistant

A beginner-friendly intent-based NLP chatbot that acts as a learning assistant. It explains concepts clearly, provides simple examples, and responds in a structured learning format.

---

## 📁 Project Structure

```
nlp-learning-assistant/
├── intents.json          ← Training data (tags, patterns, responses)
├── train.py              ← Train and save the ML model
├── chat.py               ← CLI chatbot interface
├── app.py                ← Flask web app
├── templates/
│   └── index.html        ← Web UI (dark mode chat)
├── model.pkl             ← Generated after training
├── vectorizer.pkl        ← Generated after training
├── pipeline.pkl          ← Generated after training
└── requirements.txt      ← Python dependencies
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| NLTK | Text preprocessing (tokenization, stopwords) |
| Scikit-learn | TF-IDF vectorization + Logistic Regression |
| Flask | Web interface |
| JSON | Intent storage |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

### 3. Run CLI Chat

```bash
python chat.py
```

### 4. Run Web App

```bash
python app.py
```

Then open: **http://127.0.0.1:5000**

---

## 🧠 Supported Intents

| Intent | Example Patterns |
|--------|-----------------|
| `greeting` | hello, hi, hey |
| `farewell` | bye, goodbye, exit |
| `definition` | what is, define, describe |
| `example` | give me an example, show me how |
| `explain_concept` | explain, break it down, clarify |
| `coding_help` | write code, debug, python code |
| `math_help` | solve, calculate, equation |
| `machine_learning` | ML, neural network, AI |
| `data_structures` | array, stack, queue, dictionary |
| `nlp` | tokenization, TF-IDF, NLP |
| `unknown` | fallback for unrecognized input |

---

## 💡 Response Format

Every response is structured as:

- 📌 **Short Answer** — One-line direct answer
- 📖 **Explanation** — Clear, step-by-step explanation
- 💡 **Example** — Simple, runnable example
- ✅ **Summary** — One-line takeaway

---

## 📄 License

MIT License
