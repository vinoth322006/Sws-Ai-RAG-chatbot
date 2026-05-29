"""PDF ingestion pipeline for the RAG Document Assistant.

This module handles the complete document ingestion workflow:
- PDF text extraction using PyMuPDF (fitz)
- Text chunking with sliding window and sentence-boundary awareness
- Embedding generation using SentenceTransformer (BGE-small-en-v1.5)
- Storage and deduplication in ChromaDB with MD5-based document hashing

Functions:
    get_embedding_model: Lazy-load singleton SentenceTransformer model.
    get_chroma_collection: Lazy-load singleton ChromaDB collection.
    extract_text_from_pdf: Extract cleaned text from each page of a PDF.
    chunk_text: Split text into overlapping chunks at sentence boundaries.
    embed_chunks: Generate normalised BGE embeddings for text chunks.
    ingest_pdf: End-to-end pipeline: extract → chunk → embed → upsert.
    list_documents: List all ingested documents with chunk counts.
    delete_document: Remove all chunks belonging to a document.
"""

import fitz  # PyMuPDF
import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton embedding model
# ---------------------------------------------------------------------------
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialise the singleton SentenceTransformer embedding model.

    The model ``BAAI/bge-small-en-v1.5`` is loaded on first call and cached
    for all subsequent calls so that the expensive initialisation happens
    only once per process lifetime.

    Returns:
        SentenceTransformer: The loaded embedding model instance.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: BAAI/bge-small-en-v1.5")
        _embedding_model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5", device="cpu"
        )
        logger.info("Embedding model loaded successfully")
    return _embedding_model


# ---------------------------------------------------------------------------
# ChromaDB client singleton
# ---------------------------------------------------------------------------
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_chroma_collection():
    """Get or initialise the ChromaDB persistent collection.

    On first call the ``PersistentClient`` is created with its database
    directory set to ``<project_root>/chroma_db``.  The collection
    ``rag_documents`` is created (or fetched if it already exists) with
    HNSW cosine-similarity parameters tuned for recall.

    Returns:
        chromadb.Collection: The ``rag_documents`` ChromaDB collection.
    """
    global _chroma_client, _collection
    if _collection is None:
        db_path = str(Path(__file__).parent.parent / "chroma_db")
        logger.info("Initializing ChromaDB at %s", db_path)
        _chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name="rag_documents",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 16,
                "hnsw:ef_construction": 100,
            },
        )
        logger.info(
            "ChromaDB collection ready. Current count: %d",
            _collection.count(),
        )
    return _collection


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _clean_text(raw: str) -> str:
    """Apply normalisation passes to raw extracted text.

    1. Normalise Unicode to NFC form.
    2. Replace fancy quotes (U+2018‒U+201D) with ASCII equivalents.
    3. Replace en-dash / em-dash with a regular hyphen.
    4. Remove end-of-line hyphenation artefacts (``word-\\nword``).
    5. Collapse multiple whitespace characters into a single space.

    Args:
        raw: The raw text extracted from a PDF page.

    Returns:
        str: Cleaned and normalised text.
    """
    text = unicodedata.normalize("NFC", raw)

    # Replace Unicode quotes with ASCII equivalents
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Replace en-dash / em-dash with regular hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Remove end-of-line hyphenation artefacts  (e.g. "docu-\nment" → "document")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # Collapse multiple whitespace (incl. newlines) into single spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    """Extract and clean text from every page of a PDF file.

    Each page's raw text is cleaned via ``_clean_text`` (whitespace
    normalisation, hyphenation removal, Unicode quote/dash replacement).
    Pages that yield empty text after cleaning are silently skipped.

    Args:
        pdf_path: Path to the PDF file to process.

    Returns:
        List[Dict]: A list of dictionaries, each containing:
            - ``page`` (int): 1-based page number.
            - ``text`` (str): Cleaned text content of the page.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        RuntimeError: If PyMuPDF fails to open or read the PDF.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error("PDF file not found: %s", pdf_path)
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages: List[Dict] = []
    try:
        doc = fitz.open(str(pdf_path))
        logger.info(
            "Opened PDF '%s' with %d pages", pdf_path.name, len(doc)
        )
        for page_num in range(len(doc)):
            raw_text = doc[page_num].get_text("text")
            cleaned = _clean_text(raw_text)
            if cleaned:
                pages.append({"page": page_num + 1, "text": cleaned})
        doc.close()
        logger.info(
            "Extracted text from %d non-empty pages out of %d total",
            len(pages),
            page_num + 1,
        )
    except Exception as exc:
        logger.exception("Failed to extract text from '%s'", pdf_path)
        raise RuntimeError(
            f"Failed to extract text from '{pdf_path}': {exc}"
        ) from exc

    return pages


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """Split *text* into overlapping chunks using a sliding window.

    The function walks through the text with a window of *chunk_size*
    characters.  Before finalising each chunk it looks backwards (up to
    the last 50 characters of the window) for a sentence-ending boundary
    (``. ``, ``! ``, ``? ``, or ``\\n\\n``) and breaks there to avoid
    splitting mid-sentence.  Successive windows overlap by *overlap*
    characters so that context is preserved across chunk boundaries.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk (default 500).
        overlap: Number of overlapping characters between consecutive
            chunks (default 50).

    Returns:
        List[str]: A list of non-empty, stripped text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If we haven't reached the very end, try to break at a sentence
        # boundary within the last 50 characters of the window.
        if end < text_len:
            search_start = max(start, end - 50)
            window = text[search_start:end]

            # Look for the *last* sentence boundary in the search region
            best_break = -1
            for delimiter in [". ", "! ", "? ", "\n\n"]:
                idx = window.rfind(delimiter)
                if idx != -1:
                    # Position relative to the full text (after the delimiter)
                    candidate = search_start + idx + len(delimiter)
                    if candidate > best_break:
                        best_break = candidate

            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance the window; apply overlap
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    logger.debug(
        "Chunked %d characters into %d chunks (size=%d, overlap=%d)",
        text_len,
        len(chunks),
        chunk_size,
        overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Generate normalised embeddings for a list of text chunks.

    Each chunk is prefixed with the BGE document-side instruction
    ``Represent this sentence: `` before encoding.  Encoding is batched
    at 64 chunks per batch with L2-normalisation enabled.

    Args:
        chunks: A list of text strings to embed.

    Returns:
        List[List[float]]: One embedding vector (as a list of floats)
        per input chunk.

    Raises:
        RuntimeError: If the embedding model fails to encode.
    """
    if not chunks:
        return []

    model = get_embedding_model()
    prefixed = [f"Represent this sentence: {c}" for c in chunks]

    try:
        embeddings = model.encode(
            prefixed,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = [emb.tolist() for emb in embeddings]
        logger.debug("Embedded %d chunks", len(result))
        return result
    except Exception as exc:
        logger.exception("Embedding failed for %d chunks", len(chunks))
        raise RuntimeError(f"Embedding failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def _compute_file_md5(file_path: Path) -> str:
    """Compute the MD5 hex digest of a file.

    Args:
        file_path: Path to the file.

    Returns:
        str: Lowercase hex MD5 digest.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()


def ingest_pdf(pdf_path: Path, force: bool = False) -> Dict:
    """End-to-end PDF ingestion: extract → chunk → embed → upsert.

    The file is identified by its MD5 hash (``doc_id``).  If a document
    with the same hash already exists in ChromaDB the ingestion is
    skipped unless *force* is ``True``.

    Each chunk is stored in ChromaDB with the following metadata:

    - ``doc_id`` – MD5 hex digest of the source PDF.
    - ``filename`` – Original filename (basename).
    - ``page`` – 1-based page number the chunk originates from.
    - ``chunk_idx`` – 0-based index of the chunk within the document.

    Args:
        pdf_path: Path to the PDF file.
        force: If ``True``, re-ingest even when the document already
            exists in the collection (existing chunks are deleted first).

    Returns:
        Dict: A summary dictionary with keys ``filename`` (str),
        ``chunks_added`` (int), and ``skipped`` (bool).

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        RuntimeError: On extraction, embedding, or storage failures.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error("PDF file not found: %s", pdf_path)
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    filename = pdf_path.name
    doc_id = _compute_file_md5(pdf_path)
    logger.info(
        "Ingesting '%s' (doc_id=%s, force=%s)", filename, doc_id, force
    )

    collection = get_chroma_collection()

    # --- Deduplication check ---
    try:
        existing = collection.get(where={"doc_id": doc_id}, limit=1)
        already_exists = bool(existing and existing["ids"])
    except Exception:
        already_exists = False

    if already_exists and not force:
        logger.info("Document '%s' already ingested — skipping.", filename)
        return {"filename": filename, "chunks_added": 0, "skipped": True}

    # If forcing re-ingestion, remove old chunks first
    if already_exists and force:
        logger.info("Force flag set — deleting old chunks for '%s'", filename)
        delete_document(doc_id)

    # --- Extract ---
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        logger.warning("No text extracted from '%s'", filename)
        return {"filename": filename, "chunks_added": 0, "skipped": False}

    # --- Chunk ---
    all_chunks: List[str] = []
    chunk_metadata: List[Dict] = []
    for page_info in pages:
        page_chunks = chunk_text(page_info["text"])
        for idx, chunk in enumerate(page_chunks):
            all_chunks.append(chunk)
            chunk_metadata.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_info["page"],
                    "chunk_idx": len(chunk_metadata),
                }
            )

    if not all_chunks:
        logger.warning("Chunking produced no chunks for '%s'", filename)
        return {"filename": filename, "chunks_added": 0, "skipped": False}

    logger.info("Produced %d chunks from '%s'", len(all_chunks), filename)

    # --- Embed ---
    embeddings = embed_chunks(all_chunks)

    # --- Upsert into ChromaDB ---
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(all_chunks))]
    try:
        # ChromaDB recommends batching large upserts
        batch_size = 256
        for batch_start in range(0, len(ids), batch_size):
            batch_end = min(batch_start + batch_size, len(ids))
            collection.upsert(
                ids=ids[batch_start:batch_end],
                documents=all_chunks[batch_start:batch_end],
                embeddings=embeddings[batch_start:batch_end],
                metadatas=chunk_metadata[batch_start:batch_end],
            )
        logger.info(
            "Upserted %d chunks for '%s' into ChromaDB", len(ids), filename
        )
    except Exception as exc:
        logger.exception("ChromaDB upsert failed for '%s'", filename)
        raise RuntimeError(
            f"ChromaDB upsert failed for '{filename}': {exc}"
        ) from exc

    return {"filename": filename, "chunks_added": len(ids), "skipped": False}


# ---------------------------------------------------------------------------
# Document management helpers
# ---------------------------------------------------------------------------

def list_documents() -> List[Dict]:
    """List all ingested documents with their chunk counts.

    Queries ChromaDB for all stored metadata and aggregates by unique
    ``doc_id``.

    Returns:
        List[Dict]: A list of dictionaries, each containing:
            - ``doc_id`` (str): The MD5 hash identifying the document.
            - ``filename`` (str): The original PDF filename.
            - ``chunk_count`` (int): Number of chunks stored for the document.
    """
    collection = get_chroma_collection()
    total = collection.count()
    if total == 0:
        logger.info("No documents in collection")
        return []

    try:
        # Fetch all metadata (documents not needed, embeddings not needed)
        results = collection.get(
            limit=total,
            include=["metadatas"],
        )
    except Exception as exc:
        logger.exception("Failed to list documents from ChromaDB")
        raise RuntimeError(
            f"Failed to list documents: {exc}"
        ) from exc

    doc_map: Dict[str, Dict] = {}
    for meta in results.get("metadatas", []):
        if not meta:
            continue
        did = meta.get("doc_id", "unknown")
        if did not in doc_map:
            doc_map[did] = {
                "doc_id": did,
                "filename": meta.get("filename", "unknown"),
                "chunk_count": 0,
            }
        doc_map[did]["chunk_count"] += 1

    documents = list(doc_map.values())
    logger.info("Found %d unique documents in collection", len(documents))
    return documents


def delete_document(doc_id: str) -> int:
    """Delete all chunks belonging to a document from ChromaDB.

    Args:
        doc_id: The MD5-based document identifier whose chunks should be
            removed.

    Returns:
        int: The number of chunks that were deleted.

    Raises:
        RuntimeError: If the deletion operation fails.
    """
    collection = get_chroma_collection()

    try:
        # First determine how many chunks exist for this doc_id
        existing = collection.get(
            where={"doc_id": doc_id},
            limit=collection.count(),
            include=[],
        )
        chunk_ids = existing.get("ids", [])
        count = len(chunk_ids)

        if count == 0:
            logger.info("No chunks found for doc_id=%s", doc_id)
            return 0

        collection.delete(ids=chunk_ids)
        logger.info(
            "Deleted %d chunks for doc_id=%s", count, doc_id
        )
        return count

    except Exception as exc:
        logger.exception(
            "Failed to delete chunks for doc_id=%s", doc_id
        )
        raise RuntimeError(
            f"Failed to delete document '{doc_id}': {exc}"
        ) from exc
