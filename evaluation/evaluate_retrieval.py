import csv
import os
import sys
from pathlib import Path

import pandas as pd
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "embedding"))
from embedder import Embedder  # noqa: E402
from index_qdrant import (  # noqa: E402
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    MODEL_DIR,
    SPARSE_VECTOR_NAME,
)

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "retrieval_eval_results.csv"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
NUM_RESULTS = 5
PREFETCH_LIMIT = 20

client = QdrantClient(url=QDRANT_URL)
embedder = Embedder(path=str(MODEL_DIR))
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def load_ground_truth():
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dense_search(query, limit=NUM_RESULTS):
    vec = embedder.encode(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec.tolist(),
        using=DENSE_VECTOR_NAME,
        limit=limit,
    ).points
    return [p.payload for p in results]


def sparse_search(query, limit=NUM_RESULTS):
    sparse_vec = next(sparse_model.query_embed(query))
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.SparseVector(
            indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
        ),
        using=SPARSE_VECTOR_NAME,
        limit=limit,
    ).points
    return [p.payload for p in results]


def hybrid_search(query, limit=NUM_RESULTS):
    dense_vec = embedder.encode(query)
    sparse_vec = next(sparse_model.query_embed(query))
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec.tolist(), using=DENSE_VECTOR_NAME, limit=PREFETCH_LIMIT),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
                ),
                using=SPARSE_VECTOR_NAME,
                limit=PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    ).points
    return [p.payload for p in results]


def compute_relevance(q, search_function):
    results = search_function(q["question"])
    return [int(r["chunk_id"] == q["chunk_id"]) for r in results]


def evaluate(ground_truth, search_function, desc):
    relevance_total = [
        compute_relevance(q, search_function) for q in tqdm(ground_truth, desc=desc)
    ]
    hit_rate = sum(1 for r in relevance_total if 1 in r) / len(relevance_total)
    mrr_total = 0.0
    for r in relevance_total:
        for rank, val in enumerate(r):
            if val == 1:
                mrr_total += 1 / (rank + 1)
                break
    mrr = mrr_total / len(relevance_total)
    return {"hit_rate": hit_rate, "mrr": mrr}


def main():
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground-truth questions from {GROUND_TRUTH_PATH}")

    methods = {
        "dense": dense_search,
        "sparse_bm25": sparse_search,
        "hybrid_rrf": hybrid_search,
    }

    results = {name: evaluate(ground_truth, fn, desc=name) for name, fn in methods.items()}

    df = pd.DataFrame(results).T
    print()
    print(df)

    winner = df["mrr"].idxmax()
    print(f"\nWinner: {winner} (mrr={df.loc[winner, 'mrr']:.4f}, hit_rate={df.loc[winner, 'hit_rate']:.4f})")

    df.to_csv(RESULTS_PATH)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
