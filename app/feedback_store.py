import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "feedback.db"


def _connect():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down'))
        )
        """
    )
    return con


def log_feedback(message_id: str, question: str, answer: str, rating: str) -> None:
    con = _connect()
    with con:
        con.execute(
            "INSERT OR REPLACE INTO feedback (id, timestamp, question, answer, rating) "
            "VALUES (?, ?, ?, ?, ?)",
            (message_id, time.time(), question, answer, rating),
        )
    con.close()
