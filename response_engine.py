# ============================================================
# response_engine.py — Response construction helpers
# ============================================================

from typing import Any, Dict, List, Optional

# Keywords that signal the user wants a simpler explanation
_BEGINNER_KEYWORDS   = {"simple", "basic", "beginner", "easy", "intro", "introduction", "eli5"}
# Keywords that signal the user wants more depth
_ADVANCED_KEYWORDS   = {"advanced", "deep", "deeper", "expert", "detail", "detailed", "theory", "math"}

VALID_LEVELS = ("beginner", "intermediate", "advanced")


def detect_level(message: str, requested_level: Optional[str] = None) -> str:
    """
    Determine the teaching depth level.

    Priority:
      1. *requested_level* if it is a valid VALID_LEVELS value.
      2. Keyword scan of *message*.
      3. Default → "intermediate".
    """
    if requested_level and requested_level.lower() in VALID_LEVELS:
        return requested_level.lower()

    lower = message.lower()
    for kw in _BEGINNER_KEYWORDS:
        if kw in lower:
            return "beginner"
    for kw in _ADVANCED_KEYWORDS:
        if kw in lower:
            return "advanced"

    return "intermediate"


def build_structured_response(
    intent_data: Dict[str, Any],
    level: str,
    recent_intents: Optional[List[str]] = None,
) -> str:
    """
    Build a multi-section teaching response from *intent_data*.

    Sections always included: answer, steps, analogy, summary, practice question.
    "deeper" section is appended only when *level* == "advanced".
    A context nod is prepended when *recent_intents* is non-empty.
    """
    parts: List[str] = []

    # Optional context nod
    if recent_intents:
        prev = recent_intents[-1]
        parts.append(f"(We were just discussing **{prev}** — building on that…)\n")

    # Core sections
    parts.append(f"**Answer:** {intent_data.get('answer', '')}")

    steps: List[str] = intent_data.get("steps", [])
    if steps:
        numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        parts.append(f"**Steps:**\n{numbered}")

    analogy = intent_data.get("analogy", "")
    if analogy:
        parts.append(f"**Analogy:** {analogy}")

    # Deeper insight only for advanced learners
    if level == "advanced":
        deeper = intent_data.get("deeper", "")
        if deeper:
            parts.append(f"**Deeper Insight:** {deeper}")

    summary = intent_data.get("summary", "")
    if summary:
        parts.append(f"**Summary:** {summary}")

    # Practice question(s) — pick first one
    practice_questions: List[str] = intent_data.get("practice_questions", [])
    if practice_questions:
        parts.append(f"**Practice:** {practice_questions[0]}")

    return "\n\n".join(parts)


def fallback_response() -> str:
    """Return a generic fallback message when confidence is too low."""
    return (
        "I'm not confident I understood your question. "
        "Could you rephrase it or try asking about a specific NLP topic "
        "(e.g. TF-IDF, logistic regression, tokenisation)?"
    )
