# ClickHouse Docs Assistant

A RAG assistant that answers questions about ClickHouse (SQL syntax, administration, deployment)
grounded in the official [ClickHouse docs](https://github.com/ClickHouse/clickhouse-docs) — built
as the capstone project for [DataTalksClub's LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

## Problem

ClickHouse's documentation is large, spread across many pages, and easy to misremember details
from (exact setting names, syntax edge cases, trade-offs between similar features like
materialized views vs. projections). Asking a general-purpose LLM directly risks confidently wrong
or outdated answers, since it wasn't trained specifically on ClickHouse internals and has no way
to cite where an answer comes from.

This project is a chat assistant that retrieves the actual relevant doc passages before answering,
so responses are grounded in real ClickHouse documentation rather than the model's memory. It's
built to be genuinely useful for day-to-day ClickHouse/DWH work — not just a course exercise — as
a quick way to check syntax, settings, or administration behavior without manually searching docs.

## Architecture

```
ClickHouse/clickhouse-docs (GitHub, pinned commit)
        │  dlt ingestion pipeline
        ▼
   DuckDB (raw + chunked docs)
        │  ONNX dense embedding + BM25 sparse embedding
        ▼
   Qdrant (hybrid vector store: dense + sparse named vectors)
        │  sparse BM25 retrieval (winning method, see Evaluation below)
        ▼
   Pydantic AI agent (Groq Llama 3.1 8B) ──tool call──▶ Qdrant search
        │  every LLM call + tool call auto-traced via native OTel instrumentation
        ▼
   Streamlit chat UI  ──────────────▶  Postgres (spans + user feedback)
                                              │
                                              ▼
                                   Streamlit monitoring dashboard (5 charts)
```

| Stage | Code | What it does |
|---|---|---|
| Ingestion | [`ingestion/`](ingestion/) | dlt pipeline pulls a curated subset of the ClickHouse docs repo, cleans MDX/JSX artifacts, chunks, lands in DuckDB |
| Embedding & indexing | [`embedding/`](embedding/) | ONNX `all-MiniLM-L6-v2` dense embeddings + BM25 sparse embeddings, indexed into Qdrant with named vectors |
| Retrieval + agent | [`rag/`](rag/) | Pydantic AI agent with a `search_docs` tool backed by sparse BM25 retrieval |
| Chat interface | [`app/`](app/) | Streamlit chat UI with thumbs up/down feedback |
| Monitoring | [`monitoring/`](monitoring/) | Self-hosted OTel span export → Postgres → 5-chart Streamlit dashboard |
| Evaluation | [`evaluation/`](evaluation/) | Ground-truth generation, retrieval evaluation, LLM-judge prompt evaluation |

## Dataset

The official [`ClickHouse/clickhouse-docs`](https://github.com/ClickHouse/clickhouse-docs) repo,
pinned to commit `e02ede9` for reproducibility. A curated subset is ingested — `docs/cloud`,
`docs/integrations`, `docs/use-cases`, `docs/chdb`, `docs/whats-new`, `docs/about-us` (its
`adopters.md` alone was ~18% of all chunks, a company-name list with no Q&A value), and non-English
`i18n/*` mirrors are excluded (see `ingestion/pipeline.py` for the full rationale). Result: **2,062
chunks from 243 files**, MDX/JSX-cleaned before chunking.

## Tech stack

| Component | Choice | Why |
|---|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) via [Pydantic AI](https://ai.pydantic.dev/) | Fast, free-tier inference; `llama-3.3-70b-versatile` was the original pick but hit Groq's 100k-token/day cap during ground-truth generation, so the RAG agent runs on the (separately quota'd) 8B model instead |
| Retrieval | [Qdrant](https://qdrant.tech/) hybrid (dense + sparse), sparse-only used in production | Covers both the retrieval-evaluation rubric line and a hybrid-search best-practice implementation in one build; evaluation showed sparse BM25 alone actually retrieves better (see below) |
| Ingestion | [dlt](https://dlthub.com/) | Automated, resumable ingestion pipeline (vs. a one-off script) |
| Interface + monitoring | [Streamlit](https://streamlit.io/) | One framework for both the chat UI and the monitoring dashboard |
| Monitoring backend | OpenTelemetry → Postgres (self-hosted, not Logfire) | A peer reviewer cloning this repo can reproduce the full dashboard with only `docker compose up` — no external account or API token required |
| Containerization | Docker Compose | Full stack (app, qdrant, postgres, dashboard) in one `docker-compose.yml` |

## Evaluation

**Retrieval** — three approaches compared against 375 LLM-generated ground-truth questions
(`evaluation/generate_ground_truth.py`, `evaluation/evaluate_retrieval.py`):

| Method | Hit rate | MRR |
|---|---|---|
| Dense only | 0.344 | 0.225 |
| **Sparse BM25 only (used)** | **0.616** | **0.389** |
| Hybrid (RRF fusion) | 0.584 | 0.370 |

Sparse BM25 alone wins. Hybrid fusion actually *hurts* here because the dense side (a small
general-purpose embedding model with no ClickHouse-specific vocabulary) is weak enough that RRF
drags the combined ranking down rather than lifting it. Hybrid search stays implemented and
evaluated in this repo — it's not thrown away — but the RAG agent queries sparse BM25 only, since
the rubric's standard is "the best one is used."

**LLM output** — three system-prompt variants (`concise`, `thorough`, `current`/baseline) judged
head-to-head by an LLM judge over 20 questions, with retrieved context held fixed so the prompt was
the only variable (`evaluation/evaluate_llm_judge.py`):

| Variant | Wins |
|---|---|
| **Thorough (adopted)** | **16 / 20** |
| Current (baseline) | 3 / 20 |
| Concise | 1 / 20 |

"Thorough" (explain mechanism + settings/caveats/trade-offs, not just the bare fact) won
decisively and is now the agent's system prompt in `rag/agent.py`.

**Ground-truth data note**: Groq's free tier caps at 100,000 tokens/day. Generating 900 target
questions (3 per chunk × 300 sampled chunks) hit that cap partway through, landing 375 questions
from 125 chunks before every further call started 429'ing. The rubric has no minimum dataset size,
and 375 is close to the course's own 360-question benchmark, so evaluation proceeded on the 375
that succeeded rather than waiting out the daily quota.

## Interface

A Streamlit chat app (`app/main.py`) — ask a question, get a grounded answer, with example-question
buttons and thumbs up/down feedback on every response.

<!-- TODO: screenshot of the chat UI (run `docker compose up -d`, screenshot http://localhost:8501) -->

## Monitoring

Every agent run is traced via Pydantic AI's native OpenTelemetry instrumentation (LLM calls, tool
calls, token counts) and exported to Postgres. User feedback (thumbs up/down) is stored alongside
it. A separate Streamlit dashboard (`monitoring/dashboard.py`) shows:

1. Conversations per day
2. Average latency by step (whole request / LLM call / doc search)
3. Feedback up vs. down
4. Token usage per day (input vs. output)
5. Top 10 most-asked questions

...plus a recent-conversations table and a time-range filter (7 / 30 / 90 days / all).

<!-- TODO: screenshot of the monitoring dashboard (run `docker compose up -d`, screenshot http://localhost:8502) -->

## Setup

### Quick start (Docker)

Requires Docker Desktop and a free [Groq API key](https://console.groq.com/keys).

```bash
git clone https://github.com/alteezy/clickhouse-docs-assistant.git
cd clickhouse-docs-assistant
cp .env.example .env   # fill in GROQ_API_KEY
```

**First run only** — populate the vector store (nothing is pre-baked into the repo; a fresh clone
starts with an empty Qdrant):

```bash
docker compose up -d qdrant     # index_qdrant.py needs a running Qdrant to write to
pip install -r requirements.txt
python ingestion/pipeline.py        # docs -> DuckDB
python embedding/download_model.py  # fetch the ONNX embedding model
python embedding/index_qdrant.py    # embed + index into Qdrant
```

Then bring up the rest of the stack:

```bash
docker compose up -d
```

- Chat UI: http://localhost:8501
- Monitoring dashboard: http://localhost:8502

### Local dev (no Docker)

```bash
pip install -r requirements.txt
# run the three ingestion/embedding scripts above once, against a local Qdrant
# (docker compose up -d qdrant postgres) or your own instances
streamlit run app/main.py
streamlit run monitoring/dashboard.py   # separate process, different port
```

Python 3.12 (matches the Docker image). All dependencies are version-pinned in
[`requirements.txt`](requirements.txt).

## Repo structure

```
ingestion/    dlt pipeline: GitHub docs -> DuckDB
embedding/    ONNX dense + BM25 sparse embedding, Qdrant indexing
rag/          Pydantic AI agent + retrieval
app/          Streamlit chat UI + feedback storage
monitoring/   OTel export, Postgres schema, monitoring dashboard
evaluation/   Ground-truth generation, retrieval eval, LLM-judge eval
docker-compose.yml   qdrant + postgres + app + dashboard
Dockerfile           shared image for app and dashboard services
```

## Reproducing the evaluation results

```bash
python evaluation/generate_ground_truth.py   # regenerates evaluation/ground_truth.csv (Groq TPD-capped, see note above)
python evaluation/evaluate_retrieval.py      # regenerates evaluation/retrieval_eval_results.csv
python evaluation/evaluate_llm_judge.py      # regenerates evaluation/llm_judge_results.csv
```

## Status

Tasks 1-10 (ingestion, embedding/indexing, retrieval evaluation, RAG agent, LLM-output evaluation,
chat interface, monitoring, containerization, this README) are complete. Cloud deployment (bonus)
is not yet done and is left as future work.

## License

[MIT](LICENSE)
