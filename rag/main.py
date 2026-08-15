import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

from agent import rag_agent  # noqa: E402

DEFAULT_QUESTION = "How do asynchronous inserts work in ClickHouse?"


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    result = rag_agent.run_sync(question)
    print(result.output)


if __name__ == "__main__":
    main()
