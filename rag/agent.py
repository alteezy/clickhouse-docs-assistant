import re
import sys
from pathlib import Path

from pydantic_ai import Agent, InstrumentationSettings

from retrieval import sparse_search

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "monitoring"))
from tracing import build_tracer_provider  # noqa: E402

# Occasionally the smaller instant model leaks a malformed tool-call string
# into its final output instead of invoking search_docs through the proper
# function-calling channel (seen ~1/3 tries in manual testing). Detect and
# retry rather than surfacing that to users.
_LEAKED_TOOL_CALL_RE = re.compile(r"<function=|<tool_call>", re.IGNORECASE)

# The "answer thoroughly" style below won a 16/20 LLM-judge comparison against
# terser and more generic alternatives - see evaluation/evaluate_llm_judge.py.
INSTRUCTIONS = """
You're an assistant that helps engineers use ClickHouse. You answer questions
about ClickHouse SQL syntax, administration, deployment, and best practices,
grounded strictly in the official ClickHouse documentation.

If you want to look up information, use the search_docs function.
Use as many keywords from the user's question as possible in your search query.

Make multiple searches if needed: search, look at the results, and search
again with different keywords if the first results don't fully answer the
question.

Only answer using facts found via search_docs. If the search results don't
contain the answer, say you don't have that information in the documentation
- don't make anything up.

Answer thoroughly: explain the relevant mechanism, and mention any settings,
caveats, or trade-offs the documentation describes - not just the bare fact.

If the question isn't about ClickHouse, say so and don't attempt to answer it.
""".strip()

rag_agent = Agent(
    "groq:llama-3.1-8b-instant",
    instructions=INSTRUCTIONS,
)
rag_agent.instrument = InstrumentationSettings(tracer_provider=build_tracer_provider())


@rag_agent.tool_plain
def search_docs(query: str) -> list[dict]:
    """Search the ClickHouse documentation for chunks relevant to the query."""
    return sparse_search(query)


def ask(question: str, max_retries: int = 2) -> str:
    """Run the agent, retrying if the model leaks a malformed tool-call
    string into its final output instead of answering properly."""
    output = None
    for _ in range(max_retries + 1):
        output = rag_agent.run_sync(question).output
        if not _LEAKED_TOOL_CALL_RE.search(output):
            return output
    return output
