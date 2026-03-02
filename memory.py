# ============================================================
# memory.py — SQLite-backed short-term conversation memory
# ============================================================

import sqlite3
import logging
from typing import List, Optional, Tuple

from config import MEMORY_DB_PATH, MEMORY_WINDOW, ARTIFACTS_DIR, DEBUG
import os

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Stores the last *window* user/assistant exchanges in an SQLite database.

    Schema: interactions(id INTEGER PK, user_msg TEXT, intent TEXT,
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    """

    def __init__(
        self,
        db_path: str = MEMORY_DB_PATH,
        window: int = MEMORY_WINDOW,
    ):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._window  = window
        self._init_db()
        logger.debug("ConversationMemory initialised (db=%s, window=%d)", db_path, window)

    # -------------------------------------------------------------- #
    #  Internal helpers                                               #
    # -------------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        """Create the interactions table if it does not exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_msg  TEXT    NOT NULL,
                    intent    TEXT    NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        logger.debug("interactions table ready.")

    # -------------------------------------------------------------- #
    #  Public API                                                     #
    # -------------------------------------------------------------- #
    def add(self, user_msg: str, intent: str) -> None:
        """Persist a new interaction."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO interactions (user_msg, intent) VALUES (?, ?)",
                (user_msg, intent),
            )
            conn.commit()
        logger.debug("Memory.add → intent=%s", intent)

    def get_recent(self, n: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Return the *n* most-recent (user_msg, intent) pairs, oldest first.
        Defaults to the configured window size.
        """
        limit = n if n is not None else self._window
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_msg, intent
                FROM   interactions
                ORDER  BY id DESC
                LIMIT  ?
                """,
                (limit,),
            ).fetchall()
        return list(reversed(rows))

    def last_intent(self) -> Optional[str]:
        """Return the intent from the most-recent interaction, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT intent FROM interactions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None
