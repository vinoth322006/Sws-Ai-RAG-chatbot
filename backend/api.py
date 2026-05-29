"""
FastAPI Application for the RAG Chatbot System.

Provides RESTful (and SSE-streaming) endpoints for:
    - PDF document upload, listing, and deletion
    - Chat with retrieval-augmented generation (streaming)
    - Application settings management
    - Health checks
    - Static frontend serving

Run directly with ``python api.py`` to start the development server on
port 8000.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_DATA_DIR: Path = _PROJECT_ROOT / "data"
_PDF_DIR: Path = _DATA_DIR / "pdfs"
_CONFIG_DIR: Path = _PROJECT_ROOT / "config"
_FRONTEND_DIR: Path = _PROJECT_ROOT / "frontend"
_MODELS_DIR: Path = _PROJECT_ROOT / "models"
_CHROMA_DIR: Path = _PROJECT_ROOT / "chroma_db"

# ---------------------------------------------------------------------------
# Late / conditional sibling-module imports
# ---------------------------------------------------------------------------
# These are imported inside functions that need them so the module can be
# parsed even when the sibling modules have unmet heavy dependencies (e.g.
# chromadb, sentence-transformers).  We cache references at module level
# once successfully imported.
_llm_mod: Optional[Any] = None
_retrieval_mod: Optional[Any] = None


def _get_llm() -> Any:
    """Lazy-import and cache the ``backend.llm`` module."""
    global _llm_mod
    if _llm_mod is None:
        try:
            from backend import llm as _mod  # type: ignore[import-untyped]
            _llm_mod = _mod
        except ImportError:
            # Fall back to a relative import when running this file directly
            import llm as _mod  # type: ignore[import-untyped]
            _llm_mod = _mod
    return _llm_mod


def _get_retrieval() -> Any:
    """Lazy-import and cache the retrieval helpers.

    The retrieval module is expected to expose:
        - ``ingest_pdf(file_path: str) -> int``
        - ``list_documents() -> List[Dict]``
        - ``delete_document(doc_id: str) -> int``
        - ``retrieve_chunks(query: str, top_k: int, min_score: float) -> List[Dict]``
        - ``get_collection_count() -> int``
        - ``get_embedding_model_name() -> str``
        - ``preload_embedding_model() -> None``

    If the module is not yet implemented a lightweight stub is used so the
    API can still start.
    """
    global _retrieval_mod
    if _retrieval_mod is not None:
        return _retrieval_mod

    try:
        from backend import retrieval as _mod  # type: ignore[import-untyped]
        _retrieval_mod = _mod
        return _retrieval_mod
    except ImportError:
        pass

    try:
        import retrieval as _mod  # type: ignore[import-untyped]
        _retrieval_mod = _mod
        return _retrieval_mod
    except ImportError:
        pass

    # ----- Minimal stub so the server boots even without retrieval.py ------
    class _Stub:
        """Placeholder when the real retrieval module is unavailable."""

        @staticmethod
        def ingest_pdf(file_path: str) -> int:
            logger.warning("retrieval.ingest_pdf stub called — no real ingestion performed.")
            return 0

        @staticmethod
        def list_documents() -> List[Dict[str, Any]]:
            logger.warning("retrieval.list_documents stub called.")
            return []

        @staticmethod
        def delete_document(doc_id: str) -> int:
            logger.warning("retrieval.delete_document stub called.")
            return 0

        @staticmethod
        def retrieve_chunks(
            query: str, top_k: int = 5, min_score: float = 0.25
        ) -> List[Dict[str, Any]]:
            logger.warning("retrieval.retrieve_chunks stub called.")
            return []

        @staticmethod
        def get_collection_count() -> int:
            return 0

        @staticmethod
        def get_embedding_model_name() -> str:
            return "none (stub)"

        @staticmethod
        def preload_embedding_model() -> None:
            logger.warning("retrieval.preload_embedding_model stub called.")

    logger.warning(
        "Retrieval module not found — using stub. "
        "Upload/search features will be non-functional."
    )
    _retrieval_mod = _Stub()
    return _retrieval_mod


# ===================================================================
# FastAPI application
# ===================================================================

app: FastAPI = FastAPI(
    title="RAG Chatbot API",
    description="Backend API for the Retrieval-Augmented Generation chatbot.",
    version="1.0.0",
)

# ---- CORS (permissive for local development) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
# Startup event
# ===================================================================


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise directories, pre-load models, and log system info."""
    # Ensure required directories exist
    for directory in (_DATA_DIR, _PDF_DIR, _CONFIG_DIR, _MODELS_DIR, _CHROMA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Directory ready: %s", directory)

    # Pre-load the embedding model so the first query is fast
    try:
        retrieval = _get_retrieval()
        retrieval.preload_embedding_model()
        logger.info("Embedding model pre-loaded successfully.")
    except Exception as exc:
        logger.warning("Could not pre-load embedding model: %s", exc)

    # System information
    logger.info("=" * 60)
    logger.info("RAG Chatbot API starting up")
    logger.info("Python      : %s", sys.version)
    logger.info("Platform    : %s", platform.platform())
    logger.info("Project root: %s", _PROJECT_ROOT)
    logger.info("=" * 60)


# ===================================================================
# Endpoint helpers
# ===================================================================


def _mask_api_key(key: str) -> str:
    """Return a masked version of *key*, showing only the last 4 characters.

    Args:
        key: The raw API key string.

    Returns:
        str: A masked string such as ``"****abcd"`` or ``""`` when empty.
    """
    if not key or len(key) <= 4:
        return key
    return "*" * (len(key) - 4) + key[-4:]


def _mmr_rerank(
    chunks: List[Dict[str, Any]], query: str, top_k: int = 5, diversity: float = 0.3
) -> List[Dict[str, Any]]:
    """Apply a simple Maximal Marginal Relevance (MMR) re-ranking.

    This lightweight heuristic deduplicates near-identical chunks by
    penalising candidates whose text is very similar to already-selected
    results.

    Args:
        chunks: Candidate chunks, each expected to have at least a
                ``"content"`` (or ``"text"``) key and a ``"score"`` key.
        query: The original user query (unused in this simple version but
               included for API symmetry with heavier MMR implementations).
        top_k: Maximum number of chunks to return.
        diversity: Weight given to diversity vs. relevance (0–1).

    Returns:
        List[Dict]: Re-ranked (and possibly pruned) list of chunks.
    """
    if not chunks:
        return []

    selected: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = list(chunks)

    while candidates and len(selected) < top_k:
        best_idx: int = 0
        best_score: float = -1.0

        for idx, cand in enumerate(candidates):
            relevance: float = float(cand.get("score", 0.0))

            # Compute max similarity to already-selected items (simple
            # Jaccard on word sets as a cheap proxy).
            max_sim: float = 0.0
            cand_text: str = cand.get("content", cand.get("text", ""))
            cand_words = set(cand_text.lower().split())

            for sel in selected:
                sel_text: str = sel.get("content", sel.get("text", ""))
                sel_words = set(sel_text.lower().split())
                if cand_words or sel_words:
                    intersection: int = len(cand_words & sel_words)
                    union: int = len(cand_words | sel_words)
                    sim: float = intersection / union if union else 0.0
                    max_sim = max(max_sim, sim)

            mmr_score: float = (1 - diversity) * relevance - diversity * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(candidates.pop(best_idx))

    return selected


def _build_chat_messages(
    question: str,
    context_chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]],
    max_context_chars: int = 6000,
) -> List[Dict[str, str]]:
    """Construct the messages list sent to the LLM.

    The system prompt instructs the model to answer based on the provided
    context.  Retrieved chunks are concatenated (trimmed to
    *max_context_chars*) and placed in a system message.  Previous
    conversation turns from *history* are appended, followed by the
    current user question.

    Args:
        question: The user's current question.
        context_chunks: Retrieved document chunks.
        history: Prior conversation turns
                 (``[{"role": "user"/"assistant", "content": "…"}, …]``).
        max_context_chars: Maximum total characters of context to include.

    Returns:
        List[Dict[str, str]]: Messages ready for the LLM provider.
    """
    # Build context string from chunks
    context_parts: List[str] = []
    total_chars: int = 0

    for i, chunk in enumerate(context_chunks):
        text: str = chunk.get("content", chunk.get("text", ""))
        source: str = chunk.get("source", chunk.get("metadata", {}).get("source", "unknown"))
        snippet: str = f"[Source: {source}]\n{text}"

        if total_chars + len(snippet) > max_context_chars:
            remaining: int = max_context_chars - total_chars
            if remaining > 100:
                context_parts.append(snippet[:remaining] + "…")
            break

        context_parts.append(snippet)
        total_chars += len(snippet)

    context_text: str = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

    system_prompt: str = (
        "You are a helpful AI assistant. Answer the user's question based on "
        "the following context retrieved from their documents. If the context "
        "does not contain enough information to answer, say so honestly.\n\n"
        "## Retrieved Context\n\n"
        f"{context_text}"
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Append conversation history (limit to last 10 turns to stay within context)
    if history:
        for turn in history[-10:]:
            role: str = turn.get("role", "user")
            content: str = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Current question
    messages.append({"role": "user", "content": question})

    return messages


# ===================================================================
# API Endpoints
# ===================================================================


# ------- Upload ---------------------------------------------------

@app.post("/api/upload", response_class=JSONResponse)
async def upload_documents(files: List[UploadFile] = File(...)) -> JSONResponse:
    """Upload one or more PDF files for ingestion.

    Files are validated to be ``.pdf`` only.  Duplicate filenames receive
    a timestamp suffix to avoid overwrites.

    Returns:
        JSONResponse with ``uploaded`` list and ``total_files`` count.
    """
    uploaded: List[Dict[str, Any]] = []
    errors: List[str] = []

    for file in files:
        filename: str = file.filename or "unknown.pdf"

        # Validate extension
        if not filename.lower().endswith(".pdf"):
            errors.append(f"Rejected '{filename}': only .pdf files are accepted.")
            logger.warning("Non-PDF upload rejected: %s", filename)
            continue

        # Handle duplicate filenames
        dest: Path = _PDF_DIR / filename
        if dest.exists():
            stem: str = dest.stem
            suffix: str = dest.suffix
            timestamp: str = str(int(time.time()))
            filename = f"{stem}_{timestamp}{suffix}"
            dest = _PDF_DIR / filename
            logger.info("Duplicate filename detected — renamed to %s", filename)

        # Save the file
        try:
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info("Saved PDF: %s", dest)
        except Exception as exc:
            errors.append(f"Failed to save '{filename}': {exc}")
            logger.error("Error saving PDF %s: %s", filename, exc)
            continue

        # Ingest into the vector store
        try:
            retrieval = _get_retrieval()
            chunk_count: int = retrieval.ingest_pdf(str(dest))
            uploaded.append({
                "filename": filename,
                "chunks": chunk_count,
                "path": str(dest),
            })
            logger.info("Ingested %s — %d chunks created.", filename, chunk_count)
        except Exception as exc:
            errors.append(f"Ingestion failed for '{filename}': {exc}")
            logger.error("Ingestion error for %s: %s", filename, exc)

    # Count total files on disk
    total_files: int = len(list(_PDF_DIR.glob("*.pdf")))

    response_body: Dict[str, Any] = {
        "uploaded": uploaded,
        "total_files": total_files,
    }
    if errors:
        response_body["errors"] = errors

    status_code: int = 200 if uploaded else (400 if errors else 200)
    return JSONResponse(content=response_body, status_code=status_code)


# ------- List documents ------------------------------------------

@app.get("/api/documents", response_class=JSONResponse)
async def get_documents() -> JSONResponse:
    """List all ingested documents and their chunk counts.

    Returns:
        JSONResponse with ``documents`` list and ``total_chunks``.
    """
    try:
        retrieval = _get_retrieval()
        documents: List[Dict[str, Any]] = retrieval.list_documents()
        total_chunks: int = retrieval.get_collection_count()

        return JSONResponse(content={
            "documents": documents,
            "total_chunks": total_chunks,
        })
    except Exception as exc:
        logger.error("Error listing documents: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {exc}",
        ) from exc


# ------- Delete document -----------------------------------------

@app.delete("/api/documents/{doc_id}", response_class=JSONResponse)
async def delete_document(doc_id: str) -> JSONResponse:
    """Delete a document and its chunks from the vector store.

    If the corresponding PDF file still exists on disk it is also
    removed.

    Args:
        doc_id: Identifier of the document to delete (typically the
                filename or a hash).

    Returns:
        JSONResponse with ``deleted_chunks`` and ``doc_id``.
    """
    try:
        retrieval = _get_retrieval()
        deleted_chunks: int = retrieval.delete_document(doc_id)
        logger.info("Deleted %d chunks for doc_id=%s", deleted_chunks, doc_id)

        # Attempt to remove the PDF file from disk
        pdf_path: Path = _PDF_DIR / doc_id
        if pdf_path.exists() and pdf_path.is_file():
            pdf_path.unlink()
            logger.info("Removed PDF file: %s", pdf_path)
        else:
            # Try matching without extension or with .pdf appended
            alt_path: Path = _PDF_DIR / f"{doc_id}.pdf"
            if alt_path.exists() and alt_path.is_file():
                alt_path.unlink()
                logger.info("Removed PDF file: %s", alt_path)

        return JSONResponse(content={
            "deleted_chunks": deleted_chunks,
            "doc_id": doc_id,
        })
    except Exception as exc:
        logger.error("Error deleting document %s: %s", doc_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document '{doc_id}': {exc}",
        ) from exc


# ------- Chat (SSE streaming) ------------------------------------

@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    """Handle a chat request with retrieval-augmented generation.

    Accepts a JSON body with ``question`` (required) and ``history``
    (optional list of prior turns).  The response is an SSE stream:

    1. A ``sources`` event with retrieved document metadata.
    2. A series of ``data`` events with ``{"token": "…"}`` payloads.
    3. A final ``data: [DONE]`` sentinel.

    Returns:
        StreamingResponse with ``text/event-stream`` content type.
    """
    try:
        body: Dict[str, Any] = await request.json()
    except Exception as exc:
        logger.error("Invalid JSON body in /api/chat: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc

    question: str = body.get("question", "").strip()
    history: List[Dict[str, str]] = body.get("history", [])

    if not question:
        raise HTTPException(status_code=400, detail="'question' field is required and cannot be empty.")

    logger.info("Chat request — question: %.120s…", question)

    # ---- Retrieval ----
    settings: Dict[str, Any] = _get_llm().load_settings()
    retrieval_cfg: Dict[str, Any] = settings.get("retrieval", {})
    top_k: int = int(retrieval_cfg.get("top_k", 5))
    min_score: float = float(retrieval_cfg.get("min_score", 0.25))

    try:
        retrieval = _get_retrieval()
        raw_chunks: List[Dict[str, Any]] = retrieval.retrieve_chunks(
            query=question, top_k=top_k * 2, min_score=min_score
        )
    except Exception as exc:
        logger.error("Retrieval error: %s", exc)
        raw_chunks = []

    # ---- MMR re-rank ----
    chunks: List[Dict[str, Any]] = _mmr_rerank(raw_chunks, question, top_k=top_k)
    logger.info("Retrieved %d raw chunks → %d after MMR re-rank.", len(raw_chunks), len(chunks))

    # ---- Build messages ----
    messages: List[Dict[str, str]] = _build_chat_messages(question, chunks, history)

    # ---- SSE generator ----
    def _event_stream() -> Generator[str, None, None]:
        """Yield SSE events for the chat response."""
        # 1. Sources event
        sources: List[Dict[str, Any]] = []
        for chunk in chunks:
            source_name: str = chunk.get("source", chunk.get("metadata", {}).get("source", "unknown"))
            score: float = float(chunk.get("score", 0.0))
            preview: str = chunk.get("content", chunk.get("text", ""))[:200]
            sources.append({
                "source": source_name,
                "score": round(score, 4),
                "preview": preview,
            })

        yield f"data: {json.dumps({'sources': sources})}\n\n"

        # 2. Token stream from LLM
        try:
            llm = _get_llm()
            for sse_chunk in llm.stream_response(messages):
                yield sse_chunk
        except Exception as exc:
            logger.error("Streaming error in /api/chat: %s", exc)
            yield f"data: {json.dumps({'token': f'Error: {exc}'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------- Settings -------------------------------------------------

@app.get("/api/settings", response_class=JSONResponse)
async def get_settings() -> JSONResponse:
    """Return application settings with API keys masked.

    Only the last 4 characters of each API key are visible.

    Returns:
        JSONResponse containing the masked settings dictionary.
    """
    try:
        llm = _get_llm()
        settings: Dict[str, Any] = llm.load_settings()

        # Mask API keys in provider sub-dicts
        for provider_key in ("gemini", "openrouter", "fireworks", "grok"):
            provider_cfg: Dict[str, Any] = settings.get(provider_key, {})
            if "api_key" in provider_cfg:
                provider_cfg["api_key"] = _mask_api_key(provider_cfg["api_key"])

        return JSONResponse(content=settings)
    except Exception as exc:
        logger.error("Error loading settings: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to load settings: {exc}"
        ) from exc


@app.post("/api/settings", response_class=JSONResponse)
async def update_settings(request: Request) -> JSONResponse:
    """Update and persist application settings.

    The provider value is validated against the list of known providers.

    Returns:
        JSONResponse ``{"saved": true}``.
    """
    try:
        new_settings: Dict[str, Any] = await request.json()
    except Exception as exc:
        logger.error("Invalid JSON body in POST /api/settings: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc

    # Validate provider
    valid_providers: List[str] = ["local", "gemini", "openrouter", "fireworks", "grok"]
    provider: str = new_settings.get("provider", "local").lower().strip()

    if provider not in valid_providers:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid provider '{provider}'. "
                f"Must be one of: {', '.join(valid_providers)}"
            ),
        )

    new_settings["provider"] = provider

    # Merge with existing settings so partial updates are safe.  If the
    # caller sends a masked API key (starts with *) we keep the old value.
    try:
        llm = _get_llm()
        existing: Dict[str, Any] = llm.load_settings()

        for provider_key in ("gemini", "openrouter", "fireworks", "grok"):
            incoming_cfg: Dict[str, Any] = new_settings.get(provider_key, {})
            existing_cfg: Dict[str, Any] = existing.get(provider_key, {})
            incoming_key: str = incoming_cfg.get("api_key", "")

            # If the key looks masked, preserve the original value
            if incoming_key and incoming_key.startswith("*"):
                incoming_cfg["api_key"] = existing_cfg.get("api_key", "")

            # Merge: existing ← incoming
            merged_cfg: Dict[str, Any] = {**existing_cfg, **incoming_cfg}
            new_settings[provider_key] = merged_cfg

        # Fill in any keys not provided by the caller
        merged_settings: Dict[str, Any] = {**existing, **new_settings}

        llm.save_settings(merged_settings)
        logger.info("Settings updated — provider=%s", provider)

        return JSONResponse(content={"saved": True})
    except Exception as exc:
        logger.error("Error saving settings: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to save settings: {exc}"
        ) from exc


# ------- Health check ---------------------------------------------

@app.get("/api/health", response_class=JSONResponse)
async def health_check() -> JSONResponse:
    """Return system health information.

    Includes the current provider, document/chunk counts, and the name
    of the embedding model in use.

    Returns:
        JSONResponse with system health data.
    """
    try:
        llm = _get_llm()
        settings: Dict[str, Any] = llm.load_settings()
        retrieval = _get_retrieval()

        doc_list: List[Dict[str, Any]] = []
        try:
            doc_list = retrieval.list_documents()
        except Exception:
            pass

        chunk_count: int = 0
        try:
            chunk_count = retrieval.get_collection_count()
        except Exception:
            pass

        embedding_model: str = "unknown"
        try:
            embedding_model = retrieval.get_embedding_model_name()
        except Exception:
            pass

        return JSONResponse(content={
            "status": "healthy",
            "provider": settings.get("provider", "local"),
            "document_count": len(doc_list),
            "chunk_count": chunk_count,
            "embedding_model": embedding_model,
        })
    except Exception as exc:
        logger.error("Health check error: %s", exc)
        return JSONResponse(
            content={"status": "unhealthy", "error": str(exc)},
            status_code=500,
        )


# ------- Frontend -------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend() -> HTMLResponse:
    """Serve the frontend ``index.html``.

    Returns a 404-style HTML message if the file does not exist yet.

    Returns:
        HTMLResponse with the contents of ``frontend/index.html``.
    """
    index_path: Path = _FRONTEND_DIR / "index.html"

    if not index_path.exists():
        logger.warning("Frontend index.html not found at %s", index_path)
        return HTMLResponse(
            content=(
                "<html><body>"
                "<h1>RAG Chatbot</h1>"
                "<p>Frontend not found. Place <code>index.html</code> in "
                f"<code>{_FRONTEND_DIR}</code>.</p>"
                "</body></html>"
            ),
            status_code=200,
        )

    try:
        html_content: str = index_path.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as exc:
        logger.error("Error reading frontend index.html: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to serve frontend.") from exc


# Mount static files from the frontend directory (CSS, JS, images, etc.)
# This is placed after explicit routes so that ``/`` is handled by the
# route above rather than the static file mount.
if _FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(_FRONTEND_DIR)),
        name="static",
    )


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    import uvicorn  # type: ignore[import-untyped]

    logger.info("Starting RAG Chatbot API server …")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
