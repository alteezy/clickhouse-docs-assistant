import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monitoring"))
from db import connect, init_db  # noqa: E402

init_db()


def log_feedback(message_id: str, question: str, answer: str, rating: str) -> None:
    con = connect()
    with con, con.cursor() as cur:
        cur.execute(
            """
            INSERT INTO feedback (id, timestamp, question, answer, rating)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET rating = EXCLUDED.rating
            """,
            (message_id, datetime.fromtimestamp(time.time(), tz=timezone.utc), question, answer, rating),
        )
    con.close()
