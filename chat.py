# ============================================================
# chat.py — CLI interface for the NLP Learning Assistant
# ============================================================

import json       # For loading intents and responses
import pickle     # For loading saved model artifacts
import random     # For picking a random response from multiple options
import nltk
import sys

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Constants
INTENTS_FILE  = "intents.json"
PIPELINE_FILE = "pipeline.pkl"
CONFIDENCE_THRESHOLD = 0.25   # Below this confidence, fall back to "unknown"

def load_artifacts():
    """Load the trained pipeline and intents data."""
    try:
        with open(PIPELINE_FILE, "rb") as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        print("Error: 'pipeline.pkl' not found.")
        print("Please run 'python train.py' first to train the model.")
        sys.exit(1)

    with open(INTENTS_FILE, "r") as f:
        intents_data = json.load(f)

    return pipeline, intents_data

def preprocess(text):
    """Apply the same cleaning used during training."""
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered)

def predict_intent(pipeline, user_input):
    """
    Return the predicted intent tag and confidence score.
    If confidence is below the threshold, return 'unknown'.
    """
    cleaned = preprocess(user_input)
    proba = pipeline.predict_proba([cleaned])[0]
    max_confidence = max(proba)
    predicted_tag  = pipeline.classes_[proba.argmax()]

    if max_confidence < CONFIDENCE_THRESHOLD:
        return "unknown", max_confidence

    return predicted_tag, max_confidence

def get_response(tag, intents_data):
    """Find the intent matching the tag and return a random response dict."""
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    # Fallback to unknown
    for intent in intents_data["intents"]:
        if intent["tag"] == "unknown":
            return random.choice(intent["responses"])

def format_response(response_dict, tag, confidence):
    """Pretty-print the structured learning response in the terminal."""
    border = "=" * 55
    print(f"\n{border}")
    print(f"  Intent: {tag}  |  Confidence: {confidence:.0%}")
    print(border)
    print(f"\nShort Answer:\n   {response_dict['short']}")
    print(f"\nExplanation:\n   {response_dict['explanation']}")
    print(f"\nExample:\n   {response_dict['example']}")
    print(f"\nSummary:\n   {response_dict['summary']}")
    print(f"\n{border}\n")

def main():
    print("\n" + "=" * 55)
    print("  NLP Learning Assistant - CLI Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 55 + "\n")

    pipeline, intents_data = load_artifacts()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended. Keep learning!")
            break

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            tag, confidence = predict_intent(pipeline, user_input)
            response = get_response(tag, intents_data)
            format_response(response, tag, confidence)
            print("Goodbye! Happy learning!")
            break

        if not user_input:
            print("Please type something!\n")
            continue

        tag, confidence = predict_intent(pipeline, user_input)
        response = get_response(tag, intents_data)
        format_response(response, tag, confidence)

if __name__ == "__main__":
    main()
