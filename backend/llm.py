"""
LLM Provider Module for RAG Chatbot System.

Supports five LLM providers with streaming capabilities:
    - Local (llama.cpp via llama-cpp-python)
    - Google Gemini
    - OpenRouter
    - Fireworks AI
    - Grok (xAI)

Each provider implements a generator-based streaming interface that yields
token strings. The unified ``stream_response`` function routes to the
correct provider based on the persisted settings and wraps output as
Server-Sent Events (SSE).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singleton for the local llama_cpp model
# ---------------------------------------------------------------------------
_llama_model: Optional[Any] = None
_llama_model_path: Optional[str] = None

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_CONFIG_DIR: Path = _PROJECT_ROOT / "config"
_SETTINGS_FILE: Path = _CONFIG_DIR / "settings.json"

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
_DEFAULT_SETTINGS: Dict[str, Any] = {
    "provider": "local",
    "local": {
        "model_path": "models/model.gguf",
        "n_ctx": 4096,
        "n_threads": 6,
        "n_gpu_layers": 0,
        "temperature": 0.7,
    },
    "gemini": {
        "api_key": "",
        "model": "gemini-1.5-flash",
    },
    "openrouter": {
        "api_key": "",
        "model": "mistralai/mistral-7b-instruct:free",
    },
    "fireworks": {
        "api_key": "",
        "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    },
    "grok": {
        "api_key": "",
        "model": "grok-beta",
    },
    "retrieval": {
        "top_k": 5,
        "min_score": 0.25,
    },
    "chunk_size": 500,
    "chunk_overlap": 50,
}


# ===================================================================
# Settings persistence
# ===================================================================


def load_settings() -> Dict[str, Any]:
    """Load application settings from *config/settings.json*.

    If the file does not exist or cannot be parsed the built-in defaults
    are returned instead.  Missing keys in a partially-filled settings
    file are back-filled from the defaults so callers always receive a
    complete dictionary.

    Returns:
        Dict[str, Any]: The merged settings dictionary.
    """
    try:
        if _SETTINGS_FILE.exists():
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as fh:
                user_settings: Dict[str, Any] = json.load(fh)
            # Merge defaults for any missing top-level keys
            merged: Dict[str, Any] = _DEFAULT_SETTINGS.copy()
            for key, value in user_settings.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
            logger.info("Settings loaded from %s", _SETTINGS_FILE)
            return merged
        logger.info(
            "Settings file not found at %s — using defaults", _SETTINGS_FILE
        )
        return _DEFAULT_SETTINGS.copy()
    except json.JSONDecodeError as exc:
        logger.error(
            "Malformed JSON in settings file %s: %s — using defaults",
            _SETTINGS_FILE,
            exc,
        )
        return _DEFAULT_SETTINGS.copy()
    except Exception as exc:
        logger.error(
            "Unexpected error loading settings: %s — using defaults", exc
        )
        return _DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> None:
    """Persist *settings* to *config/settings.json*.

    The configuration directory is created automatically if it does not
    already exist.

    Args:
        settings: The complete settings dictionary to write.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2, ensure_ascii=False)
        logger.info("Settings saved to %s", _SETTINGS_FILE)
    except Exception as exc:
        logger.error("Failed to save settings to %s: %s", _SETTINGS_FILE, exc)
        raise


# ===================================================================
# Provider: Local (llama-cpp-python)
# ===================================================================


def _get_llama_model(settings: Dict[str, Any]) -> Any:
    """Return a cached ``Llama`` model instance, creating it on first call.

    The model is lazily loaded and stored in a module-level singleton.  If
    the requested model path differs from the cached one the old instance
    is discarded and a new model is loaded.

    Args:
        settings: The full settings dict (``settings['local']`` is used).

    Returns:
        A ``llama_cpp.Llama`` model instance.

    Raises:
        FileNotFoundError: If the GGUF file does not exist.
        ImportError: If ``llama-cpp-python`` is not installed.
    """
    global _llama_model, _llama_model_path

    local_cfg: Dict[str, Any] = settings.get("local", {})
    model_path_str: str = local_cfg.get("model_path", "models/model.gguf")

    # Resolve relative paths against the project root
    model_path: Path = Path(model_path_str)
    if not model_path.is_absolute():
        model_path = _PROJECT_ROOT / model_path

    resolved: str = str(model_path.resolve())

    # Return cached model if path has not changed
    if _llama_model is not None and _llama_model_path == resolved:
        return _llama_model

    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model file not found: {resolved}. "
            "Please download a GGUF model and place it at the configured path."
        )

    try:
        from llama_cpp import Llama  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'llama-cpp-python' package is required for the local "
            "provider.  Install it with: pip install llama-cpp-python"
        ) from exc

    logger.info("Loading local GGUF model from %s …", resolved)
    _llama_model = Llama(
        model_path=resolved,
        n_ctx=int(local_cfg.get("n_ctx", 4096)),
        n_threads=int(local_cfg.get("n_threads", 6)),
        n_gpu_layers=int(local_cfg.get("n_gpu_layers", 0)),
        verbose=False,
    )
    _llama_model_path = resolved
    logger.info("Local model loaded successfully.")
    return _llama_model


def stream_local(
    messages: List[Dict[str, str]], settings: Dict[str, Any]
) -> Generator[str, None, None]:
    """Stream tokens from a local GGUF model via *llama-cpp-python*.

    The model instance is lazily loaded and cached as a module-level
    singleton.  Subsequent calls reuse the same instance unless the model
    path changes.

    Args:
        messages: Chat messages in OpenAI-compatible format
                  (``[{"role": "…", "content": "…"}, …]``).
        settings: Full application settings dictionary.

    Yields:
        str: Individual token strings as they are generated.
    """
    local_cfg: Dict[str, Any] = settings.get("local", {})
    temperature: float = float(local_cfg.get("temperature", 0.7))

    try:
        model = _get_llama_model(settings)
    except FileNotFoundError as exc:
        logger.error("Local model not found: %s", exc)
        yield f"Error: {exc}"
        return
    except ImportError as exc:
        logger.error("llama-cpp-python not installed: %s", exc)
        yield f"Error: {exc}"
        return
    except Exception as exc:
        logger.error("Failed to load local model: %s", exc)
        yield f"Error loading local model: {exc}"
        return

    try:
        response = model.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=temperature,
        )
        for chunk in response:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                token: Optional[str] = delta.get("content")
                if token:
                    yield token
    except Exception as exc:
        logger.error("Error during local model streaming: %s", exc)
        yield f"Error during generation: {exc}"


# ===================================================================
# Provider: Google Gemini
# ===================================================================


def stream_gemini(
    messages: List[Dict[str, str]], settings: Dict[str, Any]
) -> Generator[str, None, None]:
    """Stream tokens from the Google Gemini API.

    Uses the ``google-generativeai`` SDK.  The API key is read from
    ``settings['gemini']['api_key']``.

    Args:
        messages: Chat messages in OpenAI-compatible format.
        settings: Full application settings dictionary.

    Yields:
        str: Text chunks as they arrive from the API.
    """
    gemini_cfg: Dict[str, Any] = settings.get("gemini", {})
    api_key: str = gemini_cfg.get("api_key", "")
    model_name: str = gemini_cfg.get("model", "gemini-1.5-flash")

    if not api_key:
        logger.error("Gemini API key is not configured.")
        yield "Error: Gemini API key is not configured. Please add it in Settings."
        return

    try:
        import google.generativeai as genai  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.error("google-generativeai not installed: %s", exc)
        yield "Error: The 'google-generativeai' package is required. Install with: pip install google-generativeai"
        return

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # Convert OpenAI-style messages to Gemini format
        gemini_history: List[Dict[str, Any]] = []
        current_prompt: str = ""

        for msg in messages:
            role: str = msg.get("role", "user")
            content: str = msg.get("content", "")

            if role == "system":
                # Gemini doesn't have a native system role — prepend to first
                # user message or use as standalone user message.
                current_prompt = content + "\n\n"
            elif role == "user":
                gemini_history.append(
                    {"role": "user", "parts": [current_prompt + content]}
                )
                current_prompt = ""
            elif role == "assistant":
                gemini_history.append(
                    {"role": "model", "parts": [content]}
                )

        # If all we have is a system prompt with no user message, send as user
        if not gemini_history and current_prompt:
            gemini_history.append(
                {"role": "user", "parts": [current_prompt.strip()]}
            )

        # Start a chat or generate directly
        if len(gemini_history) > 1:
            chat = model.start_chat(history=gemini_history[:-1])
            last_parts = gemini_history[-1].get("parts", [""])
            response = chat.send_message(last_parts[0], stream=True)
        else:
            prompt_text: str = (
                gemini_history[0]["parts"][0] if gemini_history else "Hello"
            )
            response = model.generate_content(prompt_text, stream=True)

        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text

    except Exception as exc:
        logger.error("Error streaming from Gemini: %s", exc)
        yield f"Error communicating with Gemini API: {exc}"


# ===================================================================
# Helpers for OpenAI-compatible SSE streaming APIs
# ===================================================================


def _stream_openai_compatible(
    url: str,
    headers: Dict[str, str],
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    provider_name: str = "API",
) -> Generator[str, None, None]:
    """Stream tokens from an OpenAI-compatible chat-completions endpoint.

    Handles the SSE wire-format shared by OpenRouter, Fireworks AI, and
    Grok (xAI).  Lines beginning with ``data: `` are parsed as JSON;
    ``[DONE]`` signals the end of the stream.

    Args:
        url: The completions endpoint URL.
        headers: HTTP headers including authorisation.
        messages: Chat messages in OpenAI format.
        model: Model identifier string.
        temperature: Sampling temperature.
        provider_name: Human-readable provider name for log messages.

    Yields:
        str: Token text fragments.
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.error("requests library not installed: %s", exc)
        yield "Error: The 'requests' package is required. Install with: pip install requests"
        return

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }

    try:
        logger.info("Streaming from %s (model=%s) …", provider_name, model)
        response = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=120
        )
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue  # skip keep-alive blanks

            line: str = raw_line.strip()

            if not line.startswith("data: "):
                continue

            data_str: str = line[len("data: "):]

            if data_str.strip() == "[DONE]":
                logger.debug("%s stream completed.", provider_name)
                break

            try:
                data: Dict[str, Any] = json.loads(data_str)
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    token: Optional[str] = delta.get("content")
                    if token:
                        yield token
            except json.JSONDecodeError:
                logger.warning(
                    "Non-JSON SSE data from %s: %s", provider_name, data_str
                )
                continue

    except Exception as exc:
        logger.error("Error streaming from %s: %s", provider_name, exc)
        yield f"Error communicating with {provider_name}: {exc}"


# ===================================================================
# Provider: OpenRouter
# ===================================================================


def stream_openrouter(
    messages: List[Dict[str, str]], settings: Dict[str, Any]
) -> Generator[str, None, None]:
    """Stream tokens from the OpenRouter API.

    Uses the ``/api/v1/chat/completions`` endpoint with SSE streaming.

    Args:
        messages: Chat messages in OpenAI-compatible format.
        settings: Full application settings dictionary.

    Yields:
        str: Token text fragments.
    """
    or_cfg: Dict[str, Any] = settings.get("openrouter", {})
    api_key: str = or_cfg.get("api_key", "")
    model: str = or_cfg.get("model", "mistralai/mistral-7b-instruct:free")

    if not api_key:
        logger.error("OpenRouter API key is not configured.")
        yield "Error: OpenRouter API key is not configured. Please add it in Settings."
        return

    url: str = "https://openrouter.ai/api/v1/chat/completions"
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "RAG Chatbot",
        "Content-Type": "application/json",
    }

    yield from _stream_openai_compatible(
        url=url,
        headers=headers,
        messages=messages,
        model=model,
        provider_name="OpenRouter",
    )


# ===================================================================
# Provider: Fireworks AI
# ===================================================================


def stream_fireworks(
    messages: List[Dict[str, str]], settings: Dict[str, Any]
) -> Generator[str, None, None]:
    """Stream tokens from the Fireworks AI API.

    Uses the ``/inference/v1/chat/completions`` endpoint with SSE
    streaming.

    Args:
        messages: Chat messages in OpenAI-compatible format.
        settings: Full application settings dictionary.

    Yields:
        str: Token text fragments.
    """
    fw_cfg: Dict[str, Any] = settings.get("fireworks", {})
    api_key: str = fw_cfg.get("api_key", "")
    model: str = fw_cfg.get(
        "model", "accounts/fireworks/models/llama-v3p1-8b-instruct"
    )

    if not api_key:
        logger.error("Fireworks API key is not configured.")
        yield "Error: Fireworks API key is not configured. Please add it in Settings."
        return

    url: str = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    yield from _stream_openai_compatible(
        url=url,
        headers=headers,
        messages=messages,
        model=model,
        provider_name="Fireworks",
    )


# ===================================================================
# Provider: Grok (xAI)
# ===================================================================


def stream_grok(
    messages: List[Dict[str, str]], settings: Dict[str, Any]
) -> Generator[str, None, None]:
    """Stream tokens from the Grok (xAI) API.

    Uses the ``/v1/chat/completions`` endpoint with SSE streaming.

    Args:
        messages: Chat messages in OpenAI-compatible format.
        settings: Full application settings dictionary.

    Yields:
        str: Token text fragments.
    """
    grok_cfg: Dict[str, Any] = settings.get("grok", {})
    api_key: str = grok_cfg.get("api_key", "")
    model: str = grok_cfg.get("model", "grok-beta")

    if not api_key:
        logger.error("Grok API key is not configured.")
        yield "Error: Grok API key is not configured. Please add it in Settings."
        return

    url: str = "https://api.x.ai/v1/chat/completions"
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    yield from _stream_openai_compatible(
        url=url,
        headers=headers,
        messages=messages,
        model=model,
        provider_name="Grok",
    )


# ===================================================================
# Unified streaming router
# ===================================================================

_PROVIDER_HANDLERS: Dict[
    str,
    Any,  # Callable signature is the same for all
] = {
    "local": stream_local,
    "gemini": stream_gemini,
    "openrouter": stream_openrouter,
    "fireworks": stream_fireworks,
    "grok": stream_grok,
}


def stream_response(
    messages: List[Dict[str, str]],
) -> Generator[str, None, None]:
    """Route a streaming request to the active provider and emit SSE.

    Each token yielded by the underlying provider is wrapped in an SSE
    ``data:`` frame containing a JSON object with a ``token`` key.  When
    the provider finishes (or an error occurs) a final ``data: [DONE]``
    sentinel is emitted so the client knows the stream has ended.

    Args:
        messages: Chat messages in OpenAI-compatible format.

    Yields:
        str: SSE-formatted strings (``"data: {…}\\n\\n"``).
    """
    settings: Dict[str, Any] = load_settings()
    provider: str = settings.get("provider", "local").lower().strip()
    handler = _PROVIDER_HANDLERS.get(provider)

    logger.info("Streaming response via provider '%s'", provider)

    if handler is None:
        error_msg: str = (
            f"Unknown LLM provider '{provider}'. "
            f"Valid options: {', '.join(_PROVIDER_HANDLERS.keys())}"
        )
        logger.error(error_msg)
        yield f"data: {json.dumps({'token': error_msg})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        for token in handler(messages, settings):
            yield f"data: {json.dumps({'token': token})}\n\n"
    except Exception as exc:
        logger.error(
            "Unhandled exception in provider '%s': %s", provider, exc
        )
        yield f"data: {json.dumps({'token': f'Error: {exc}'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"
