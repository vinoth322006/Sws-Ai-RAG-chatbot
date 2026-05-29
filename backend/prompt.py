"""Prompt construction module for the RAG Document Assistant.

This module builds OpenAI-compatible ``messages`` lists that combine a
system prompt, optional conversation history, retrieved context, and the
current user question.  It also provides lightweight token-estimation
utilities used to keep context within LLM budget limits.

Constants:
    SYSTEM_PROMPT: The fixed system-level instruction sent to the LLM.

Functions:
    build_messages: Assemble a complete messages list for the LLM API.
    estimate_tokens: Rough token count estimation (chars / 4).
    trim_context_to_budget: Trim a list of chunks to fit a token budget.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a helpful document assistant powered by Retrieval-Augmented "
    "Generation (RAG). Your role is to answer questions accurately using "
    "ONLY the context provided below. Follow these rules strictly:\n"
    "\n"
    "1. Base your answers exclusively on the provided context.\n"
    "2. If the context does not contain enough information to answer the "
    "question, say: \"I don't have enough information in the uploaded "
    "documents to answer this question.\"\n"
    "3. Always cite which source(s) you used (mention the filename and "
    "page number).\n"
    "4. Be concise but thorough — prefer short paragraphs and bullet "
    "points.\n"
    "5. Do NOT make up information or use knowledge outside the provided "
    "context.\n"
    "6. If the question is ambiguous, ask the user for clarification.\n"
    "7. When multiple sources agree, synthesise the information rather "
    "than repeating each source verbatim."
)


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def build_messages(
    context: str,
    question: str,
    history: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Build an OpenAI-compatible ``messages`` list for the chat API.

    The returned list is structured as follows:

    1. **System message** — contains :data:`SYSTEM_PROMPT`.
    2. **History messages** — the last 6 turns (3 user + 3 assistant) from
       *history*, preserving chronological order.
    3. **User message** — the current *question* prefixed with the
       retrieved *context* block.

    Args:
        context: Formatted context string (produced by
            :func:`backend.retrieve.format_context`).
        question: The current user question.
        history: Optional list of prior message dicts, each with ``role``
            (``"user"`` or ``"assistant"``) and ``content`` keys.  Only the
            last 6 entries are used to limit context length.

    Returns:
        List[Dict]: A list of message dictionaries suitable for direct
        submission to the OpenAI Chat Completions API (or compatible
        endpoints).
    """
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append the most recent conversation history (last 6 turns)
    if history:
        recent_history = history[-6:]
        for turn in recent_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        logger.debug(
            "Included %d history turns (from %d total)",
            len(recent_history),
            len(history),
        )

    # Build the final user message with context + question
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    messages.append({"role": "user", "content": user_content})

    logger.debug(
        "Built messages list with %d entries (system + %d history + user)",
        len(messages),
        len(messages) - 2,  # subtract system and final user
    )
    return messages


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text*.

    Uses the simple heuristic of 1 token ≈ 4 characters, which provides a
    reasonable approximation for English text without requiring a full
    tokeniser.

    Args:
        text: The input text to estimate.

    Returns:
        int: Estimated token count (always ≥ 0).
    """
    if not text:
        return 0
    return len(text) // 4


# ---------------------------------------------------------------------------
# Context budget trimming
# ---------------------------------------------------------------------------

def trim_context_to_budget(
    chunks: List[Dict],
    max_tokens: int = 3000,
) -> List[Dict]:
    """Remove chunks from the end of the list until within a token budget.

    Starting from the full list of *chunks*, the function checks whether
    the combined estimated token count of all chunk texts exceeds
    *max_tokens*.  If so, chunks are dropped from the **end** of the list
    (i.e., lowest-priority chunks) one at a time until the budget is met.
    At least one chunk is always retained regardless of its size.

    Args:
        chunks: Ordered list of chunk dicts (each must have a ``text`` key).
            Chunks should be pre-sorted by relevance (most relevant first).
        max_tokens: The maximum allowed token budget (default 3000).

    Returns:
        List[Dict]: A (possibly shorter) list of chunks that fits within
        the token budget, preserving original order.
    """
    if not chunks:
        logger.debug("trim_context_to_budget called with empty chunk list")
        return []

    def _total_tokens(chunk_list: List[Dict]) -> int:
        """Sum estimated tokens across all chunks."""
        return sum(estimate_tokens(c.get("text", "")) for c in chunk_list)

    current = list(chunks)
    total = _total_tokens(current)

    if total <= max_tokens:
        logger.debug(
            "All %d chunks fit within budget (%d / %d tokens)",
            len(current),
            total,
            max_tokens,
        )
        return current

    # Trim from the end, keeping at least 1 chunk
    while len(current) > 1 and _total_tokens(current) > max_tokens:
        removed = current.pop()
        logger.debug(
            "Trimmed chunk (score=%.3f) to fit budget — %d chunks remaining",
            removed.get("score", 0.0),
            len(current),
        )

    final_tokens = _total_tokens(current)
    logger.info(
        "Trimmed context from %d to %d chunks (%d tokens, budget %d)",
        len(chunks),
        len(current),
        final_tokens,
        max_tokens,
    )
    return current
