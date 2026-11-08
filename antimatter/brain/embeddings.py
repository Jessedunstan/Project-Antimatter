"""Local vector embedding generator powered by FastEmbed (ONNX).

Runs entirely in-process on CPU via ONNX Runtime without PyTorch or external API calls.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)


class LocalEmbeddingEngine:
    """Fast, local semantic embedding engine."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                logger.info("Initializing FastEmbed model: %s", self.model_name)
                self._model = TextEmbedding(model_name=self.model_name)
            except ImportError:
                raise ImportError(
                    "FastEmbed is not installed. Install via `pip install fastembed` "
                    "for local offline vector embeddings."
                )

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of text chunks."""
        self._ensure_loaded()
        embeddings = list(self._model.embed(documents))
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for a single search query."""
        results = self.embed_documents([query])
        return results[0]
