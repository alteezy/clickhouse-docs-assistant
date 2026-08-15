import os

import psycopg2

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://monitoring:monitoring@localhost:5432/monitoring"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL,
    question TEXT,
    answer TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    attributes JSONB,
    PRIMARY KEY (trace_id, span_id)
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down'))
);
"""


def connect():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    con = connect()
    with con, con.cursor() as cur:
        cur.execute(SCHEMA)
    con.close()
