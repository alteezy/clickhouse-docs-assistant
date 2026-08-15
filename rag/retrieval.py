import os
import sys
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "embedding"))
from index_qdrant import COLLECTION_NAME, SPARSE_VECTOR_NAME  # noqa: E402

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
NUM_RESULTS = 3
# Groq's llama-3.1-8b-instant caps requests at 6000 tokens/minute - keep each
# search_docs call small enough that a couple of agentic search rounds (plus
# the system prompt and growing conversation history) stay well under that.
CONTENT_CHAR_LIMIT = 600

_client = QdrantClient(url=QDRANT_URL)
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def sparse_search(query: str, limit: int = NUM_RESULTS) -> list[dict]:
    """Search the ClickHouse docs collection using BM25 sparse vectors.

    Sparse BM25 was chosen over dense and hybrid retrieval based on the
    hit-rate/MRR evaluation in evaluation/evaluate_retrieval.py.
    """
    sparse_vec = next(_sparse_model.query_embed(query))
    results = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.SparseVector(
            indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
        ),
        using=SPARSE_VECTOR_NAME,
        limit=limit,
    ).points
    return [
        {
            "chunk_id": p.payload["chunk_id"],
            "filename": p.payload["filename"],
            "title": p.payload["title"],
            "content": p.payload["content"][:CONTENT_CHAR_LIMIT],
        }
        for p in results
    ]
