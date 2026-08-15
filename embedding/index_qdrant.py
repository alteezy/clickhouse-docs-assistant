import os
import time
import uuid
from pathlib import Path

import duckdb
import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from download_model import download
from embedder import Embedder

REPO_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = REPO_ROOT / "ingestion" / "clickhouse_docs_ingestion.duckdb"
MODEL_DIR = REPO_ROOT / "models" / "Xenova" / "all-MiniLM-L6-v2"
COLLECTION_NAME = "clickhouse_docs"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM = 384
BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 100
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def load_chunks():
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    rows = con.sql(
        """
        SELECT chunk_id, filename, title, description, doc_type, slug, start, content
        FROM clickhouse_docs.doc_chunks
        ORDER BY filename, start
        """
    ).fetchall()
    con.close()
    columns = ["chunk_id", "filename", "title", "description", "doc_type", "slug", "start", "content"]
    return [dict(zip(columns, row)) for row in rows]


def embed_dense(chunks):
    if not MODEL_DIR.exists():
        print(f"Downloading dense model to {MODEL_DIR} ...")
        download("Xenova/all-MiniLM-L6-v2", dest=str(REPO_ROOT / "models"))

    embedder = Embedder(path=str(MODEL_DIR))
    vectors = []
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="dense embed"):
        batch = [c["content"] for c in chunks[i:i + BATCH_SIZE]]
        vectors.append(embedder.encode_batch(batch))
    return np.vstack(vectors)


def embed_sparse(chunks):
    model = SparseTextEmbedding(model_name="Qdrant/bm25")
    contents = [c["content"] for c in chunks]
    return list(tqdm(model.embed(contents), total=len(contents), desc="sparse embed"))


def wait_for_qdrant(client, retries=10, delay=2):
    for attempt in range(retries):
        try:
            client.get_collections()
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def ensure_collection(client):
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_DIM, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def build_points(chunks, dense_vectors, sparse_vectors):
    points = []
    for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"]))
        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vec.tolist(),
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "filename": chunk["filename"],
                    "title": chunk["title"],
                    "description": chunk["description"],
                    "doc_type": chunk["doc_type"],
                    "slug": chunk["slug"],
                    "start": chunk["start"],
                    "content": chunk["content"],
                },
            )
        )
    return points


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {DUCKDB_PATH}")

    dense_vectors = embed_dense(chunks)
    sparse_vectors = embed_sparse(chunks)

    client = QdrantClient(url=QDRANT_URL)
    wait_for_qdrant(client)
    ensure_collection(client)

    points = build_points(chunks, dense_vectors, sparse_vectors)
    for i in tqdm(range(0, len(points), UPSERT_BATCH_SIZE), desc="upsert"):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + UPSERT_BATCH_SIZE])

    count = client.count(COLLECTION_NAME).count
    print(f"Qdrant collection '{COLLECTION_NAME}' now has {count} points (expected {len(chunks)}).")


if __name__ == "__main__":
    main()
