"""Offline doubles so the harness runs with no keys, no network and no model download.

`HashingEmbedder` is the same bag-of-words hashing embedder as `tests/conftest.py`'s
`FakeEmbedder`; `OfflineLLM` returns fixed, schema-valid placeholder outputs. Scores
produced with these are meaningless; `--offline` only proves the harness end to end.
"""

import hashlib
import math
from collections.abc import Sequence

from pydantic import BaseModel

from evals.judge import JudgedClaim, JudgeVerdict
from studiodesk.llm import LLMInvalidOutput
from studiodesk.models.answer import LLMAnswer
from studiodesk.models.bugs import CandidateJudgement, DuplicateJudgements

OFFLINE_MODEL = "offline-fake-llm"
OFFLINE_EMBEDDER = "offline-hashing-embedder"


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (unit-normalised)."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for word in text.lower().split():
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest) % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


def _tag_ids(user_content: str, tag: str) -> list[str]:
    """Ids of `<tag id="...">` elements in a built prompt, in order."""
    return [part.split('"', 1)[0] for part in user_content.split(f'<{tag} id="')[1:]]


class OfflineLLM:
    """`LLMClient` with deterministic placeholder answers for each schema the evals use."""

    def __init__(self) -> None:
        self.calls = 0

    def structured[SchemaT: BaseModel](
        self, system: str, user_content: str, schema: type[SchemaT]
    ) -> SchemaT:
        self.calls += 1
        result: BaseModel
        if schema is LLMAnswer:
            ids = _tag_ids(user_content, "document")
            result = LLMAnswer(
                answer=f"Offline placeholder answer [{ids[0]}]." if ids else "No documents.",
                cited_ids=ids[:1],
                insufficient_context=not ids,
            )
        elif schema is DuplicateJudgements:
            ids = _tag_ids(user_content, "document")
            result = DuplicateJudgements(
                judgements=[
                    CandidateJudgement(
                        candidate_id=doc_id,
                        is_duplicate=index == 0,
                        confidence=0.5,
                        reason="offline placeholder",
                    )
                    for index, doc_id in enumerate(ids)
                ]
            )
        elif schema is JudgeVerdict:
            ids = _tag_ids(user_content, "source")
            result = JudgeVerdict(
                claims=[JudgedClaim(text="offline placeholder", supported=True, source_ids=ids[:1])]
            )
        else:
            raise LLMInvalidOutput(f"offline LLM has no response for {schema.__name__}")
        if not isinstance(result, schema):
            raise LLMInvalidOutput("offline LLM schema mismatch")
        return result

    def close(self) -> None:
        """Nothing to release."""
