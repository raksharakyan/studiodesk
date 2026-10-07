"""SentenceTransformerEmbedder wrapper behaviour with a fake model (no download)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from studiodesk.embeddings import Embedder, SentenceTransformerEmbedder


class _FakeModel:
    def __init__(self, dim: int | None = 3) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def get_embedding_dimension(self) -> int | None:
        return self.dim

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        self.calls.append({"texts": list(texts), **kwargs})
        return np.array([[float(len(t)), 0.0, 0.0] for t in texts], dtype=np.float32)


def test_embed_documents_passes_normalize_and_batch_size() -> None:
    model = _FakeModel()
    embedder = SentenceTransformerEmbedder(model, batch_size=8)  # type: ignore[arg-type]

    vectors = embedder.embed_documents(["ab", "abcd"])

    assert vectors == [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    assert all(isinstance(x, float) for row in vectors for x in row)
    [call] = model.calls
    assert call["normalize_embeddings"] is True
    assert call["batch_size"] == 8
    assert call["show_progress_bar"] is False
    assert embedder.dim == 3


def test_embed_query_returns_single_vector() -> None:
    embedder = SentenceTransformerEmbedder(_FakeModel())  # type: ignore[arg-type]

    assert embedder.embed_query("abc") == [3.0, 0.0, 0.0]


def test_empty_input_does_not_call_model() -> None:
    model = _FakeModel()

    assert SentenceTransformerEmbedder(model).embed_documents([]) == []  # type: ignore[arg-type]
    assert model.calls == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_rejects_non_positive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerEmbedder(_FakeModel(), batch_size=batch_size)  # type: ignore[arg-type]


def test_rejects_model_without_fixed_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        SentenceTransformerEmbedder(_FakeModel(dim=None))  # type: ignore[arg-type]


def test_embedders_satisfy_protocol(fake_embedder: Embedder) -> None:
    assert isinstance(SentenceTransformerEmbedder(_FakeModel()), Embedder)  # type: ignore[arg-type]
    assert isinstance(fake_embedder, Embedder)


def test_fake_embedder_is_deterministic_and_normalised(fake_embedder: Embedder) -> None:
    a, b = fake_embedder.embed_query("Save File"), fake_embedder.embed_query("save file")

    assert a == b
    assert len(a) == 64
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
