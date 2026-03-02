# ============================================================
# app.py — Flask REST API for the NLP Learning Assistant
# ============================================================

import json
import logging

from flask import Flask, jsonify, request

from config          import INTENTS_FILE, CONFIDENCE_THRESHOLD, DEBUG
from memory          import ConversationMemory
from model           import LocalSklearnModel
from preprocessing   import normalize
from response_engine import (
    build_structured_response,
    detect_level,
    fallback_response,
)

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ----- Load shared resources once at startup -----
with open(INTENTS_FILE, "r") as f:
    _intents_data = json.load(f)

_intent_map: dict = {intent["tag"]: intent for intent in _intents_data["intents"]}

_model  = LocalSklearnModel()
_memory = ConversationMemory()


# ------------------------------------------------------------------ #
#  Routes                                                             #
# ------------------------------------------------------------------ #
@app.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Accept JSON body: {"message": str, "level"?: str}

    Returns JSON:
    {
        "intent":      str,
        "confidence":  float,
        "level":       str,
        "response":    str
    }
    """
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()

    if not message:
        return jsonify({"error": "The 'message' field is required and must not be empty."}), 400

    requested_level = body.get("level")

    # 1. Determine teaching level
    level = detect_level(message, requested_level)

    # 2. Normalise and classify
    normalised = normalize(message)
    label, confidence, _proba = _model.predict_intent(normalised)

    logger.info("intent=%s  confidence=%.3f  level=%s", label, confidence, level)

    # 3. Apply confidence threshold
    if confidence < CONFIDENCE_THRESHOLD or label not in _intent_map:
        _memory.add(message, "fallback")
        return jsonify(
            {
                "intent":     "fallback",
                "confidence": round(confidence, 4),
                "level":      level,
                "response":   fallback_response(),
            }
        )

    # 4. Retrieve short-term memory context
    recent = _memory.get_recent()
    recent_intents = [r[1] for r in recent if r[1] not in ("fallback",)]

    # 5. Build structured teaching response
    intent_data = _intent_map[label]
    response_text = build_structured_response(intent_data, level, recent_intents)

    # 6. Persist interaction
    _memory.add(message, label)

    return jsonify(
        {
            "intent":     label,
            "confidence": round(confidence, 4),
            "level":      level,
            "response":   response_text,
        }
    )


if __name__ == "__main__":
    app.run(debug=DEBUG, host="0.0.0.0", port=5000)
