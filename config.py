# ============================================================
# config.py — Shared configuration for the NLP Learning Assistant
# ============================================================

import os

# ----- Paths -----
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR  = os.path.join(BASE_DIR, "artifacts")
INTENTS_FILE   = os.path.join(BASE_DIR, "intents.json")
MODEL_PATH     = os.path.join(ARTIFACTS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.pkl")
MEMORY_DB_PATH = os.path.join(ARTIFACTS_DIR, "memory.db")

# ----- Model behaviour -----
CONFIDENCE_THRESHOLD = 0.30   # Below this score, return fallback response
MEMORY_WINDOW        = 5      # Number of recent interactions to keep in context

# ----- Debug / logging -----
DEBUG = os.environ.get("NLP_DEBUG", "false").lower() == "true"
