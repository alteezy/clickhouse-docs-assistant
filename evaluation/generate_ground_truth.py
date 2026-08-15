import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = REPO_ROOT / "ingestion" / "clickhouse_docs_ingestion.duckdb"
OUTPUT_PATH = Path(__file__).resolve().parent / "ground_truth.csv"

SAMPLE_SIZE = 300
QUESTIONS_PER_CHUNK = 3
SEED = 42
MODEL = "llama-3.3-70b-versatile"
MAX_WORKERS = 8
MAX_RETRIES = 5

INSTRUCTIONS = f"""
You emulate an engineer who uses ClickHouse and is searching the documentation
for help. You are given one chunk of the ClickHouse documentation.
Formulate {QUESTIONS_PER_CHUNK} questions this engineer might ask that are
directly answered by this chunk.

Rules:
- The chunk should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the chunk; don't copy its phrasing.
- The questions should resemble how people actually search or ask online:
  not too formal, not too short, not too long.
- Ask about the content of the chunk, not about its formatting, filename, or
  the fact that it's a documentation excerpt.
""".strip()


class Questions(BaseModel):
    questions: list[str]


def load_chunks():
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    rows = con.sql(
        "SELECT chunk_id, filename, title, content FROM clickhouse_docs.doc_chunks"
    ).fetchall()
    con.close()
    columns = ["chunk_id", "filename", "title", "content"]
    return [dict(zip(columns, row)) for row in rows]


def sample_chunks(chunks, n=SAMPLE_SIZE, seed=SEED):
    rng = random.Random(seed)
    return rng.sample(chunks, min(n, len(chunks)))


def generate_questions(client, chunk):
    user_prompt = json.dumps({"title": chunk["title"], "content": chunk["content"]})
    schema_hint = Questions.model_json_schema()
    messages = [
        {
            "role": "system",
            "content": (
                f"{INSTRUCTIONS}\n\n"
                "Respond with a single JSON object matching this schema, "
                f"and nothing else:\n{json.dumps(schema_hint)}"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
            )
            parsed = Questions.model_validate_json(response.choices[0].message.content)
            return parsed.questions
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    load_dotenv(REPO_ROOT / ".env")
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")

    chunks = load_chunks()
    sample = sample_chunks(chunks)
    print(f"Sampled {len(sample)} / {len(chunks)} chunks (seed={SEED})")

    def process(chunk):
        return chunk, generate_questions(client, chunk)

    rows = []
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process, c): c for c in sample}
        for future in tqdm(as_completed(futures), total=len(futures), desc="generating ground truth"):
            chunk = futures[future]
            try:
                _, questions = future.result()
            except Exception as e:
                errors += 1
                print(f"  skipped {chunk['chunk_id']}: {e}")
                continue
            for q in questions:
                rows.append({"question": q, "chunk_id": chunk["chunk_id"], "filename": chunk["filename"]})

    print(f"Generated {len(rows)} questions from {len(sample) - errors}/{len(sample)} chunks ({errors} failed)")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "chunk_id", "filename"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
