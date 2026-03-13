"""Generate embeddings using Ollama."""

from langchain_ollama import OllamaEmbeddings
from app.config.config import OLLAMA_BASE_URL, EMBEDDING_MODEL

# Singleton embedding model
_embeddings = None


def get_embedding_model() -> OllamaEmbeddings:
    """Get or create the Ollama embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=EMBEDDING_MODEL,
        )
    return _embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Generate embedding for a single query."""
    model = get_embedding_model()
    return model.embed_query(text)
