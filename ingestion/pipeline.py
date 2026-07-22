"""
Ingestion pipeline: pull a curated subset of the ClickHouse docs from GitHub,
chunk them, and land the chunks in DuckDB via dlt.
"""

import re

import dlt
from gitsource import GithubRepositoryDataReader, chunk_documents

# Docusaurus/MDX artifacts that aren't real prose: "import Foo from '...'"
# statements and custom JSX components (always capitalized, e.g. <Image .../>,
# <TableOfContents />, <Content>...</Content>). Lowercase HTML tags like <br>
# or <table> are left alone since those are legitimate inline content.
_IMPORT_LINE = re.compile(r"(?m)^import\s+.*$\n?")
_JSX_COMMENT = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
_JSX_SELF_CLOSING = re.compile(r"<[A-Z][A-Za-z0-9_.]*(?:\s+[^>]*)?/>")
_JSX_TAG = re.compile(r"</?[A-Z][A-Za-z0-9_.]*(?:\s+[^>]*)?>")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_mdx(content: str) -> str:
    content = _IMPORT_LINE.sub("", content)
    content = _JSX_COMMENT.sub("", content)
    content = _JSX_SELF_CLOSING.sub("", content)
    content = _JSX_TAG.sub("", content)
    content = _BLANK_LINES.sub("\n\n", content)
    return content.strip()

REPO_OWNER = "ClickHouse"
REPO_NAME = "clickhouse-docs"
# Pinned commit for reproducible ingestion (main branch as of when this was written).
COMMIT = "e02ede9"

# Core ClickHouse concepts/operations/SQL usage docs. Deliberately excludes
# docs/cloud (ClickHouse Cloud console UI), docs/integrations (third-party tool
# guides), docs/use-cases (long-form tutorials), docs/chdb (separate project),
# docs/whats-new (changelogs), docs/about-us (company/marketing content - its
# adopters.md alone produced ~18% of all chunks in an early trial run, mostly
# a company-name list with no Q&A value), and any i18n/* mirrors (non-English
# duplicates) to keep the knowledge base focused on "how does ClickHouse
# itself work" rather than every adjacent tool/integration.
ALLOWED_DIRS = {
    "guides",
    "best-practices",
    "managing-data",
    "deployment-guides",
    "concepts",
    "faq",
    "operations_",
    "data-modeling",
    "materialized-view",
    "tips-and-tricks",
    "troubleshooting",
    "getting-started",
    "native-protocol",
    "tools-and-utilities",
    "kubernetes-operator",
    "dictionary",
    "data-compression",
}

# A handful of standalone pages that live directly under docs/ (not in a
# subdirectory) but are still worth including.
ALLOWED_ROOT_FILES = {
    "docs/intro.md",
    "docs/tutorial.md",
    "docs/introduction-index.md",
    "docs/deployment-modes.md",
}


def is_allowed(filepath: str) -> bool:
    if not filepath.startswith("docs/"):
        return False

    if filepath in ALLOWED_ROOT_FILES:
        return True

    parts = filepath.split("/")
    if len(parts) < 3:
        # docs/<something>.md with no subdirectory, and not in the allow-list above
        return False

    return parts[1] in ALLOWED_DIRS


def load_documents():
    reader = GithubRepositoryDataReader(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        commit_id=COMMIT,
        allowed_extensions={"md", "mdx"},
        filename_filter=is_allowed,
    )
    files = reader.read()
    documents = [f.parse() for f in files]
    for doc in documents:
        doc["content"] = clean_mdx(doc["content"])
    return documents


@dlt.resource(name="doc_chunks", write_disposition="replace")
def doc_chunks():
    documents = load_documents()
    # Sliding-window chunking (gitsource's own chunker, same one taught in the
    # course's etc/chunking.md) rather than a hand-rolled header splitter -
    # reuses proven tooling instead of reinventing it.
    chunks = chunk_documents(documents, size=2000, step=1000)

    for chunk in chunks:
        chunk["chunk_id"] = f"{chunk['filename']}::{chunk['start']}"
        yield chunk


pipeline = dlt.pipeline(
    pipeline_name="clickhouse_docs_ingestion",
    destination="duckdb",
    dataset_name="clickhouse_docs",
)


if __name__ == "__main__":
    load_info = pipeline.run(doc_chunks())
    print(load_info)
