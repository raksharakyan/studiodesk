"""SearchRequest / SearchFilters validation and snippet building."""

from typing import Any

import pytest
from pydantic import ValidationError

from studiodesk.models.chunks import Chunk
from studiodesk.models.documents import DocType
from studiodesk.models.search import (
    QUERY_MAX_CHARS,
    SNIPPET_MAX_CHARS,
    TOP_K_DEFAULT,
    TOP_K_MAX,
    SearchFilters,
    SearchHit,
    SearchRequest,
    make_snippet,
)


def test_defaults() -> None:
    req = SearchRequest(query="  cryo save  ")

    assert req.query == "cryo save"
    assert req.top_k == TOP_K_DEFAULT == 5
    assert req.filters == SearchFilters()


def test_full_valid_request() -> None:
    req = SearchRequest.model_validate(
        {
            "query": "crash",
            "top_k": TOP_K_MAX,
            "filters": {
                "doc_types": ["bug_report", "crash_log"],
                "platforms": ["pc", "switch"],
                "severities": ["critical"],
                "components": ["rendering"],
                "version_min": "1.0.0",
                "version_max": "1.2.0",
            },
        }
    )

    assert req.top_k == 20
    assert req.filters.version_max == "1.2.0"


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * (QUERY_MAX_CHARS + 1)},
        {"query": 123},
        {},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "top_k": TOP_K_MAX + 1},
        {"query": "ok", "top_k": -3},
        {"query": "ok", "top_k": "many"},
        {"query": "ok", "unknown": 1},
        {"query": "ok", "filters": {"platforms": ["playstation"]}},
        {"query": "ok", "filters": {"severities": ["urgent"]}},
        {"query": "ok", "filters": {"doc_types": ["email"]}},
        {"query": "ok", "filters": {"components": ["kernel"]}},
        {"query": "ok", "filters": {"platforms": "pc"}},
        {"query": "ok", "filters": {"platforms": ["pc"] * 5}},
        {"query": "ok", "filters": {"version_min": "1.0"}},
        {"query": "ok", "filters": {"version_max": "latest"}},
        {"query": "ok", "filters": {"version_min": "1.2.0", "version_max": "1.1.9"}},
        {"query": "ok", "filters": {"version_min": "1.100.0"}},
        {"query": "ok", "filters": {"must": [{"key": "doc_id", "match": {"value": "x"}}]}},
        {"query": "ok", "filters": {"title": "anything"}},
    ],
)
def test_invalid_requests_rejected(body: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(body)


def test_query_at_max_length_accepted() -> None:
    assert len(SearchRequest(query="x" * QUERY_MAX_CHARS).query) == QUERY_MAX_CHARS


def test_equal_version_bounds_accepted() -> None:
    assert SearchFilters(version_min="1.1.0", version_max="1.1.0").version_min == "1.1.0"


def test_make_snippet_short_text_unchanged() -> None:
    assert make_snippet("hello") == "hello"
    assert make_snippet("x" * SNIPPET_MAX_CHARS) == "x" * SNIPPET_MAX_CHARS


def test_make_snippet_truncates_with_ellipsis() -> None:
    snippet = make_snippet("word " * 300)

    assert len(snippet) <= SNIPPET_MAX_CHARS
    assert snippet.endswith("...")
    assert not snippet.endswith(" ...")


def test_search_hit_from_chunk() -> None:
    chunk = Chunk(
        doc_id="BUG-0001",
        doc_type=DocType.BUG_REPORT,
        chunk_index=0,
        title="t",
        text="y" * 2000,
        version="1.0.0",
        duplicate_of="BUG-0002",
    )

    hit = SearchHit.from_chunk(chunk, 0.5)

    assert hit.doc_id == "BUG-0001"
    assert len(hit.snippet) <= SNIPPET_MAX_CHARS
    assert hit.metadata.duplicate_of == "BUG-0002"
    assert hit.metadata.version == "1.0.0"
    assert hit.score == 0.5
