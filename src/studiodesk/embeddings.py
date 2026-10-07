"""Text embedders.

`Embedder` is the protocol the rest of the app depends on. `SentenceTransformerEmbedder`
is the production implementation (Hugging Face sentence-transformers, normalized vectors,
CPU by default). The heavy `sentence_transformers`/`torch` import happens only when a model
is actually loaded, so importing this module stays cheap.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from studiodesk.config import Settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Turns text into fixed-size vectors suitable for cosine similarity."""

    @property
    def dim(self) -> int:
        """Vector dimension produced by this embedder."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document texts, one vector per input, in order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


class SentenceTransformerEmbedder:
    """`Embedder` backed by an already-loaded `SentenceTransformer` model."""

    def __init__(self, model: "SentenceTransformer", batch_size: int = 64) -> None:
        """Wrap `model`; `batch_size` bounds how many texts are encoded per forward pass."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dim = model.get_embedding_dimension()
        if dim is None:
            raise ValueError("embedding model does not report a fixed dimension")
        self._model = model
        self._batch_size = batch_size
        self._dim = int(dim)

    @classmethod
    def from_settings(cls, settings: Settings) -> "SentenceTransformerEmbedder":
        """Load `settings.embedding_model` at the pinned revision on the configured device.

        Downloads from Hugging Face on first use unless the model is already in the local
        cache (`HF_HOME`). This is slow; call it once (app lifespan or CLI), not per request.
        """
        from sentence_transformers import SentenceTransformer

        logger.info(
            "loading embedding model",
            extra={
                "model": settings.embedding_model,
                "revision": settings.embedding_model_revision,
                "device": settings.embedding_device,
            },
        )
        model = SentenceTransformer(
            settings.embedding_model,
            revision=settings.embedding_model_revision,
            device=settings.embedding_device,
            trust_remote_code=False,
        )
        return cls(model, batch_size=settings.embedding_batch_size)

    @property
    def dim(self) -> int:
        """Vector dimension of the loaded model."""
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches, returning L2-normalized vectors."""
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        rows: list[list[float]] = np.asarray(vectors, dtype=np.float32).tolist()
        return rows

    def embed_query(self, text: str) -> list[float]:
        """Embed one query with the same normalization as documents."""
        return self.embed_documents([text])[0]
