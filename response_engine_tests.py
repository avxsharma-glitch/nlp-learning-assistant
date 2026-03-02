# ============================================================
# response_engine_tests.py — Tests for response_engine module
# ============================================================

import unittest
from response_engine import detect_level, build_structured_response, fallback_response

SAMPLE_INTENT = {
    "tag": "tfidf",
    "answer": "TF-IDF weights word importance.",
    "steps": ["Compute TF.", "Compute IDF.", "Multiply them."],
    "analogy": "Like a search engine scoring words.",
    "deeper": "Sparse representation; no semantic understanding.",
    "summary": "TF-IDF rewards distinctive words.",
    "practice_questions": ["Why does 'the' get a low TF-IDF score?"],
}


class TestDetectLevel(unittest.TestCase):

    # --- keyword-based detection ---
    def test_beginner_keyword_simple(self):
        self.assertEqual(detect_level("explain it in a simple way"), "beginner")

    def test_beginner_keyword_basic(self):
        self.assertEqual(detect_level("give me a basic overview"), "beginner")

    def test_beginner_keyword_eli5(self):
        self.assertEqual(detect_level("eli5 TF-IDF"), "beginner")

    def test_advanced_keyword(self):
        self.assertEqual(detect_level("give me a detailed explanation"), "advanced")

    def test_advanced_keyword_deep(self):
        self.assertEqual(detect_level("I want a deeper understanding"), "advanced")

    def test_default_level(self):
        self.assertEqual(detect_level("what is TF-IDF"), "intermediate")

    # --- explicit requested_level overrides keyword ---
    def test_explicit_beginner(self):
        self.assertEqual(detect_level("advanced query", requested_level="beginner"), "beginner")

    def test_explicit_advanced(self):
        self.assertEqual(detect_level("simple question", requested_level="advanced"), "advanced")

    def test_explicit_intermediate(self):
        self.assertEqual(detect_level("anything", requested_level="intermediate"), "intermediate")

    def test_invalid_requested_level_falls_back_to_keyword(self):
        # "expert" is not a valid level; keyword "simple" should win
        self.assertEqual(detect_level("simple explanation", requested_level="expert"), "beginner")

    def test_case_insensitive_requested_level(self):
        self.assertEqual(detect_level("hi", requested_level="BEGINNER"), "beginner")


class TestBuildStructuredResponse(unittest.TestCase):

    def test_contains_answer(self):
        result = build_structured_response(SAMPLE_INTENT, "intermediate")
        self.assertIn("TF-IDF weights word importance.", result)

    def test_contains_steps(self):
        result = build_structured_response(SAMPLE_INTENT, "intermediate")
        self.assertIn("Compute TF.", result)

    def test_contains_analogy(self):
        result = build_structured_response(SAMPLE_INTENT, "intermediate")
        self.assertIn("search engine", result)

    def test_contains_practice(self):
        result = build_structured_response(SAMPLE_INTENT, "intermediate")
        self.assertIn("Practice", result)

    def test_deeper_included_for_advanced(self):
        result = build_structured_response(SAMPLE_INTENT, "advanced")
        self.assertIn("Sparse representation", result)

    def test_deeper_excluded_for_beginner(self):
        result = build_structured_response(SAMPLE_INTENT, "beginner")
        self.assertNotIn("Sparse representation", result)

    def test_context_nod_when_recent_intents(self):
        result = build_structured_response(SAMPLE_INTENT, "intermediate", ["greeting"])
        self.assertIn("greeting", result)

    def test_no_context_nod_when_empty(self):
        result = build_structured_response(SAMPLE_INTENT, "intermediate", [])
        self.assertNotIn("We were just", result)


class TestFallbackResponse(unittest.TestCase):

    def test_returns_string(self):
        self.assertIsInstance(fallback_response(), str)

    def test_non_empty(self):
        self.assertTrue(len(fallback_response()) > 0)


if __name__ == "__main__":
    unittest.main()
