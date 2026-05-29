"""Retrieval module for the RAG Document Assistant.

This module provides semantic search over the ChromaDB vector store:
- Query embedding with BGE query-side prefix
- Cosine-similarity search with score filtering
- Maximal Marginal Relevance (MMR) reranking for diversity
- Context formatting for downstream prompt construction

Functions:
    embed_query: Embed a user question with the BGE query prefix.
    retrieve: Perform similarity search against ChromaDB.
    mmr_rerank: Rerank retrieved chunks using MMR for diversity.
    format_context: Format retrieved chunks into a numbered context string.
"""

import logging
from typing import Dict, List

import numpy as np

from backend.ingest import get_chroma_collection, get_embedding_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

def embed_query(question: str) -> List[float]:
    """Embed a user question using the BGE query-side instruction prefix.

    The question is prefixed with
    ``Represent this question for searching relevant passages: ``
    before being encoded by the singleton SentenceTransformer model.

    Args:
        question: The natural-language question to embed.

    Returns:
        List[float]: A normalised embedding vector as a list of floats.

    Raises:
        ValueError: If *question* is empty or blank.
        RuntimeError: If the embedding model fails.
    """
    if not question or not question.strip():
        logger.error("embed_query called with empty question")
        raise ValueError("Question must be a non-empty string")

    model = get_embedding_model()
    prefixed = (
        f"Represent this question for searching relevant passages: {question}"
    )

    try:
        embedding = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = embedding.tolist()
        logger.debug("Embedded query (%d dims)", len(result))
        return result
    except Exception as exc:
        logger.exception("Failed to embed query")
        raise RuntimeError(f"Query embedding failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(question: str, top_k: int = 5) -> List[Dict]:
    """Retrieve the most relevant chunks for a question from ChromaDB.

    Steps:
    1. Embed the question with :func:`embed_query`.
    2. Perform a cosine-similarity search in ChromaDB (``n_results=top_k``).
    3. Convert ChromaDB distances to similarity scores via
       ``similarity = 1.0 - (distance / 2.0)`` (valid for cosine space).
    4. Filter out any chunk whose similarity is below 0.25.

    Args:
        question: The user's question.
        top_k: Maximum number of results to return (default 5).

    Returns:
        List[Dict]: Ranked list of chunk dictionaries, each containing:
            - ``text`` (str): The chunk's text content.
            - ``filename`` (str): Source PDF filename.
            - ``page`` (int): 1-based page number.
            - ``chunk_idx`` (int): 0-based chunk index.
            - ``score`` (float): Cosine similarity score.

    Raises:
        RuntimeError: On embedding or ChromaDB query failures.
    """
    collection = get_chroma_collection()

    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — returning no results")
        return []

    query_embedding = embed_query(question)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.exception("ChromaDB query failed")
        raise RuntimeError(f"ChromaDB query failed: {exc}") from exc

    chunks: List[Dict] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(documents, metadatas, distances):
        similarity = 1.0 - (dist / 2.0)
        if similarity < 0.25:
            logger.debug(
                "Filtering chunk (score=%.3f < 0.25): %.40s…",
                similarity,
                text,
            )
            continue
        chunks.append(
            {
                "text": text,
                "filename": meta.get("filename", "unknown"),
                "page": meta.get("page", 0),
                "chunk_idx": meta.get("chunk_idx", 0),
                "score": round(similarity, 4),
            }
        )

    logger.info(
        "Retrieved %d chunks (post-filter) for question: %.60s…",
        len(chunks),
        question,
    )
    return chunks


# ---------------------------------------------------------------------------
# MMR Reranking
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Both vectors are assumed to be already L2-normalised (as produced by
    the BGE model with ``normalize_embeddings=True``), so the cosine
    similarity reduces to a simple dot product.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        float: Cosine similarity in the range [-1, 1].
    """
    dot = float(np.dot(vec_a, vec_b))
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_rerank(
    chunks: List[Dict],
    top_n: int = 5,
    lambda_param: float = 0.7,
) -> List[Dict]:
    """Rerank chunks using Maximal Marginal Relevance (MMR).

    MMR balances *relevance* (how similar a chunk is to the query) with
    *diversity* (how different it is from chunks already selected).  The
    score for each candidate *c* is:

        MMR(c) = λ · relevance(c) − (1 − λ) · max_{s ∈ S} sim(c, s)

    where *S* is the set of already-selected chunks and ``sim`` is cosine
    similarity between chunk embeddings.

    Args:
        chunks: List of chunk dicts as returned by :func:`retrieve`.
            Each dict must contain at least ``text`` and ``score``.
        top_n: Number of chunks to return after reranking (default 5).
        lambda_param: Trade-off parameter (default 0.7).
            Higher values favour relevance; lower values favour diversity.

    Returns:
        List[Dict]: The reranked list of up to *top_n* chunks.
    """
    if len(chunks) <= 1:
        return chunks[:top_n]

    # Embed all chunk texts to obtain vectors for diversity calculation
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    try:
        embeddings = model.encode(
            [f"Represent this sentence: {t}" for t in texts],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.warning(
            "MMR embedding failed (%s) — returning chunks as-is", exc
        )
        return chunks[:top_n]

    embeddings = np.array(embeddings)

    selected_indices: List[int] = []
    remaining_indices: List[int] = list(range(len(chunks)))

    # Select the first chunk as the one with the highest relevance score
    best_first = max(remaining_indices, key=lambda i: chunks[i]["score"])
    selected_indices.append(best_first)
    remaining_indices.remove(best_first)

    while remaining_indices and len(selected_indices) < top_n:
        best_score = -float("inf")
        best_idx = remaining_indices[0]

        for idx in remaining_indices:
            relevance = chunks[idx]["score"]

            # Max similarity to any already-selected chunk
            max_sim = max(
                _cosine_similarity(embeddings[idx], embeddings[s])
                for s in selected_indices
            )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    reranked = [chunks[i] for i in selected_indices]
    logger.info(
        "MMR reranked %d chunks down to %d (λ=%.2f)",
        len(chunks),
        len(reranked),
        lambda_param,
    )
    return reranked


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a numbered context string.

    Each chunk is rendered as::

        [Source N: filename.pdf | Page P | Score 0.87]
        chunk text here
        ---

    If *chunks* is empty an explanatory placeholder is returned.

    Args:
        chunks: List of chunk dicts (must contain ``text``, ``filename``,
            ``page``, and ``score``).

    Returns:
        str: The formatted context block ready for prompt injection.
    """
    if not chunks:
        logger.warning("format_context called with no chunks")
        return "[No relevant documents found]"

    sections: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[Source {i}: {chunk.get('filename', 'unknown')} "
            f"| Page {chunk.get('page', '?')} "
            f"| Score {chunk.get('score', 0.0):.2f}]"
        )
        sections.append(f"{header}\n{chunk.get('text', '')}\n---")

    formatted = "\n\n".join(sections)
    logger.debug("Formatted context with %d sources", len(chunks))
    return formatted
