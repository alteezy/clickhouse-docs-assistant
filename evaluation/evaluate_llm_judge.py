import csv
import os
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rag"))
from retrieval import sparse_search  # noqa: E402

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "llm_judge_results.csv"

SAMPLE_SIZE = 20
NUM_CONTEXT_CHUNKS = 3
CONTEXT_CHAR_LIMIT = 800
MODEL = "llama-3.1-8b-instant"
GEN_MAX_TOKENS = 250
JUDGE_MAX_TOKENS = 150
MAX_RETRIES = 5
JUDGE_PARSE_RETRIES = 3
SEED = 7

VARIANTS = {
    "concise": (
        "You're a ClickHouse documentation assistant. Answer the question "
        "as briefly and directly as possible using only the provided "
        "context. No preamble, no elaboration beyond what's needed."
    ),
    "thorough": (
        "You're a ClickHouse documentation assistant. Answer the question "
        "thoroughly using only the provided context: explain the relevant "
        "mechanism, and mention any settings, caveats, or trade-offs the "
        "context describes, not just the bare fact."
    ),
    "current": (
        "You're an assistant that helps engineers use ClickHouse. Answer "
        "questions about ClickHouse SQL syntax, administration, deployment, "
        "and best practices, grounded strictly in the provided context. "
        "Only answer using facts found in the context. If the context "
        "doesn't contain the answer, say you don't have that information."
    ),
}


def load_sample():
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_chunk = {}
    for r in rows:
        by_chunk.setdefault(r["chunk_id"], r)  # one question per chunk
    pool = list(by_chunk.values())
    rng = random.Random(SEED)
    return rng.sample(pool, min(SAMPLE_SIZE, len(pool)))


def build_context(question):
    results = sparse_search(question, limit=NUM_CONTEXT_CHUNKS)
    parts = []
    for r in results:
        content = r["content"][:CONTEXT_CHAR_LIMIT]
        parts.append(f"[{r['filename']}]\n{content}")
    return "\n\n".join(parts)


def call_llm(client, system, user, max_tokens):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def generate_answers(client, question, context):
    answers = {}
    for name, instructions in VARIANTS.items():
        user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
        answers[name] = call_llm(client, instructions, user, GEN_MAX_TOKENS)
    return answers


def judge(client, question, answers, rng):
    # Plain-text WINNER/REASON format instead of JSON mode: the smaller
    # 8b-instant judge model proved unreliable at strict JSON schema output
    # (it would echo the schema back, or wrap the object in a list) even
    # though the same schema-injection technique worked fine for the 70b
    # model in Task 4's ground-truth generation. A simple two-line format
    # is much easier for a small model to produce correctly.
    names = list(answers.keys())
    rng.shuffle(names)  # avoid position bias
    labels = {f"Answer {i + 1}": name for i, name in enumerate(names)}
    label_block = "\n\n".join(f"{label}:\n{answers[name]}" for label, name in labels.items())

    system = (
        "You are judging which of several candidate answers best answers "
        "the user's question about ClickHouse, based on accuracy, "
        "completeness, and clarity.\n\n"
        "Respond with EXACTLY two lines and nothing else:\n"
        f"WINNER: <one of {', '.join(labels.keys())}>\n"
        "REASON: <one short sentence>"
    )
    user = f"QUESTION: {question}\n\n{label_block}"

    for attempt in range(JUDGE_PARSE_RETRIES):
        content = call_llm(client, system, user, JUDGE_MAX_TOKENS)
        winner_match = re.search(r"WINNER:\s*(Answer\s*\d+)", content, re.IGNORECASE)
        winner_variant = None
        if winner_match:
            winner_label = re.sub(r"\s+", " ", winner_match.group(1).strip())
            winner_variant = next(
                (v for lbl, v in labels.items() if lbl.lower() == winner_label.lower()), None
            )
        if winner_variant is not None:
            reason_match = re.search(r"REASON:\s*(.+)", content, re.IGNORECASE)
            reasoning = reason_match.group(1).strip() if reason_match else ""
            return winner_variant, reasoning
        if attempt == JUDGE_PARSE_RETRIES - 1:
            raise ValueError(f"judge did not return a parseable WINNER line: {content!r}")


def main():
    load_dotenv(REPO_ROOT / ".env")
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")

    sample = load_sample()
    print(f"Evaluating {len(VARIANTS)} prompt variants over {len(sample)} questions")

    rng = random.Random(SEED)
    rows = []
    wins = {name: 0 for name in VARIANTS}
    errors = 0

    for i, q in enumerate(sample, 1):
        question = q["question"]
        try:
            context = build_context(question)
            answers = generate_answers(client, question, context)
            winner, reasoning = judge(client, question, answers, rng)
        except Exception as e:
            errors += 1
            print(f"  [{i}/{len(sample)}] skipped: {e}")
            continue
        wins[winner] += 1
        rows.append({"question": question, "winner": winner, "reasoning": reasoning})
        print(f"  [{i}/{len(sample)}] winner: {winner}")

    print(f"\nCompleted {len(rows)}/{len(sample)} questions ({errors} failed)")
    print("Wins per variant:", wins)

    overall_winner = max(wins, key=wins.get)
    print(f"\nOverall winner: {overall_winner}")

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "winner", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
