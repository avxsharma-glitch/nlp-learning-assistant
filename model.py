# ============================================================
# model.py — Model abstraction + scikit-learn implementation
# ============================================================

import abc
import logging
import pickle
from typing import Tuple

import numpy as np

from config import MODEL_PATH, VECTORIZER_PATH, DEBUG

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Abstract interface — swap in a transformer model without changing  #
#  any downstream code.                                               #
# ------------------------------------------------------------------ #
class LanguageModelInterface(abc.ABC):
    """Base interface for intent classification models."""

    @abc.abstractmethod
    def predict_intent(self, text: str) -> Tuple[str, float, np.ndarray]:
        """
        Classify *text* and return a 3-tuple:
          (label: str, confidence: float, prob_vector: np.ndarray)
        """


# ------------------------------------------------------------------ #
#  Concrete scikit-learn implementation                               #
# ------------------------------------------------------------------ #
class LocalSklearnModel(LanguageModelInterface):
    """TF-IDF + Logistic Regression intent classifier."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        vectorizer_path: str = VECTORIZER_PATH,
    ):
        logger.debug("Loading vectorizer from %s", vectorizer_path)
        with open(vectorizer_path, "rb") as fv:
            self._vectorizer = pickle.load(fv)

        logger.debug("Loading model from %s", model_path)
        with open(model_path, "rb") as fm:
            self._model = pickle.load(fm)

        logger.info("LocalSklearnModel loaded successfully.")

    # -------------------------------------------------------------- #
    def predict_intent(self, text: str) -> Tuple[str, float, np.ndarray]:
        """
        Return (label, confidence, prob_vector) for *text*.

        *text* should already be normalised by preprocessing.normalize().
        """
        X           = self._vectorizer.transform([text])
        proba       = self._model.predict_proba(X)[0]
        idx         = int(np.argmax(proba))
        label       = self._model.classes_[idx]
        confidence  = float(proba[idx])

        logger.debug(
            "predict_intent → label=%s  confidence=%.3f", label, confidence
        )
        return label, confidence, proba
