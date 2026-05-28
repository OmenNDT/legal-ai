"""
LightRAG Configuration & Factory
- LLM: Claude Haiku 4.5 via OAuth API (x-api-key header)
- Embedding: qwen3-embedding:8b via Ollama (same as ChromaDB)
- Storage: NanoVectorDB + NetworkX + JsonKV (defaults)
"""
import os

# Set embedding timeout BEFORE importing LightRAG (it reads env at import time)
os.environ.setdefault("EMBEDDING_TIMEOUT", "300")
import json
import logging
import time
from pathlib import Path

import httpx
import numpy as np
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

logger = logging.getLogger(__name__)

LIGHTRAG_DIR = Path(os.getenv("LIGHTRAG_DIR", "./data/lightrag"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:8b")
CLAUDE_MODEL = os.getenv("LIGHTRAG_LLM_MODEL", "claude-haiku-4-5-20251001")
CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"


def _read_claude_token() -> str:
    """Read OAuth access token from Claude Code credentials."""
    if not CREDENTIALS_PATH.exists():
        raise RuntimeError(f"Claude credentials not found: {CREDENTIALS_PATH}")

    data = json.loads(CREDENTIALS_PATH.read_text())
    oauth = data.get("claudeAiOauth", {})
    token = oauth.get("accessToken")
    if not token:
        raise RuntimeError("No accessToken in claudeAiOauth credentials")

    # Warn if token near expiry
    expires_at = oauth.get("expiresAt")
    if expires_at:
        remaining = (expires_at / 1000) - time.time()
        if remaining < 300:
            logger.warning(f"Claude OAuth token expires in {remaining:.0f}s — re-login soon")

    return token


async def haiku_llm_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """Custom LLM function using Claude Haiku 4.5 via OAuth API."""
    token = _read_claude_token()

    messages = []
    if history_messages:
        for msg in history_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 4096 if not keyword_extraction else 1024,
        "messages": messages,
    }
    if system_prompt:
        body["system"] = system_prompt

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

    # Extract text from response content blocks
    content = data.get("content", [])
    return "".join(block.get("text", "") for block in content if block.get("type") == "text")


async def _ollama_embed_raw(texts: list[str]) -> np.ndarray:
    """Raw embedding function using qwen3-embedding:8b via Ollama."""
    async with httpx.AsyncClient(timeout=180.0) as client:
        embeddings = []
        for text in texts:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return np.array(embeddings)


def _create_embedding_func() -> EmbeddingFunc:
    """Create EmbeddingFunc wrapper for LightRAG."""
    return EmbeddingFunc(
        embedding_dim=4096,  # qwen3-embedding:8b output dim
        func=_ollama_embed_raw,
        max_token_size=8192,
    )


# Singleton instance
_lightrag_instance: LightRAG | None = None
_lightrag_initialized = False


def get_lightrag_instance() -> LightRAG:
    """Get or create singleton LightRAG instance (sync part only)."""
    global _lightrag_instance
    if _lightrag_instance is not None:
        return _lightrag_instance

    LIGHTRAG_DIR.mkdir(parents=True, exist_ok=True)

    _lightrag_instance = LightRAG(
        working_dir=str(LIGHTRAG_DIR),
        llm_model_func=haiku_llm_func,
        llm_model_max_async=4,
        embedding_func=_create_embedding_func(),
        embedding_batch_num=4,
        default_embedding_timeout=300,
    )

    logger.info(f"LightRAG initialized: dir={LIGHTRAG_DIR}, llm={CLAUDE_MODEL}, embed={OLLAMA_EMBEDDING_MODEL}")
    return _lightrag_instance


async def ensure_lightrag_ready() -> LightRAG:
    """Get LightRAG instance and ensure async storages are initialized."""
    global _lightrag_initialized
    rag = get_lightrag_instance()
    if not _lightrag_initialized:
        await rag.initialize_storages()
        _lightrag_initialized = True
        logger.info("LightRAG storages initialized")
    return rag
