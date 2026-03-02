# ============================================================
# train.py — Train TF-IDF + Logistic Regression on intents.json
# ============================================================

import json
import logging
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import classification_report

from config        import INTENTS_FILE, MODEL_PATH, VECTORIZER_PATH, ARTIFACTS_DIR, DEBUG
from preprocessing import normalize

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_training_data(intents_file: str = INTENTS_FILE):
    """
    Load intents.json and return (texts, labels) where every pattern
    has been run through normalize().
    """
    with open(intents_file, "r") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent.get("patterns", []):
            normalised = normalize(pattern)
            if normalised:           # skip empty strings after normalisation
                texts.append(normalised)
                labels.append(tag)

    logger.info("Loaded %d training samples across %d intents.", len(texts), len(set(labels)))
    return texts, labels


def train(intents_file: str = INTENTS_FILE) -> None:
    """Train and persist the TF-IDF vectoriser and Logistic Regression model."""
    texts, labels = load_training_data(intents_file)

    # Split into train / validation sets (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    # Vectorise
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)

    # Train classifier
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="auto")
    model.fit(X_train_vec, y_train)

    # Validation report
    y_pred = model.predict(X_val_vec)
    print("\n=== Validation Classification Report ===")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Persist artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as fv:
        pickle.dump(vectorizer, fv)
    with open(MODEL_PATH, "wb") as fm:
        pickle.dump(model, fm)

    logger.info("Model saved  → %s", MODEL_PATH)
    logger.info("Vectorizer saved → %s", VECTORIZER_PATH)
    print(f"\nArtifacts written to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    train()
