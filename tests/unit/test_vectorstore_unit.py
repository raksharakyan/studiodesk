"""Filter building and Qdrant client construction (no server, no network)."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from studiodesk import vectorstore
from studiodesk.config import Settings
from studiodesk.models.search import SearchFilters
from studiodesk.vectorstore import build_filter, build_qdrant_client


def _conditions(f: qm.Filter | None) -> dict[str, qm.FieldCondition]:
    assert f is not None
    assert f.should is None
    assert f.must_not is None
    assert isinstance(f.must, list)
    result: dict[str, qm.FieldCondition] = {}
    for cond in f.must:
        assert isinstance(cond, qm.FieldCondition)
        result[cond.key] = cond
    return result


@pytest.mark.parametrize("filters", [None, SearchFilters()])
def test_empty_filters_build_nothing(filters: SearchFilters | None) -> None:
    assert build_filter(filters) is None


@pytest.mark.parametrize(
    ("field", "key", "values"),
    [
        ("doc_types", "doc_type", ["crash_log", "bug_report"]),
        ("platforms", "platform", ["switch", "pc"]),
        ("severities", "severity", ["high", "critical"]),
        ("components", "component", ["save_system"]),
    ],
)
def test_each_list_filter_becomes_match_any(field: str, key: str, values: list[str]) -> None:
    conds = _conditions(build_filter(SearchFilters.model_validate({field: values})))

    assert list(conds) == [key]
    match = conds[key].match
    assert isinstance(match, qm.MatchAny)
    assert match.any == sorted(values)


def test_duplicate_list_values_deduplicated() -> None:
    conds = _conditions(build_filter(SearchFilters.model_validate({"platforms": ["pc", "pc"]})))

    match = conds["platform"].match
    assert isinstance(match, qm.MatchAny)
    assert match.any == ["pc"]


@pytest.mark.parametrize(
    ("vmin", "vmax", "gte", "lte"),
    [
        ("1.1.0", "1.3.2", 10100, 10302),
        ("1.1.0", None, 10100, None),
        (None, "1.0.1", None, 10001),
    ],
)
def test_version_range_on_version_num(
    vmin: str | None, vmax: str | None, gte: int | None, lte: int | None
) -> None:
    conds = _conditions(build_filter(SearchFilters(version_min=vmin, version_max=vmax)))

    assert list(conds) == ["version_num"]
    assert conds["version_num"].range == qm.Range(gte=gte, lte=lte)


def test_combined_filters_are_anded() -> None:
    f = SearchFilters.model_validate(
        {"platforms": ["ps5"], "severities": ["high"], "version_min": "1.0.0"}
    )

    assert set(_conditions(build_filter(f))) == {"platform", "severity", "version_num"}


class _RecordingClient:
    """Stands in for QdrantClient so no connection is attempted."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).calls.append(kwargs)


def test_remote_client_gets_url_plain_key_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingClient.calls = []
    monkeypatch.setattr(vectorstore, "QdrantClient", _RecordingClient)
    settings = Settings(
        _env_file=None,
        qdrant_url="https://cluster.example.cloud.qdrant.io",
        qdrant_api_key=SecretStr("qk-test-123"),
        qdrant_timeout_s=7,
    )

    build_qdrant_client(settings)

    assert _RecordingClient.calls == [
        {"url": "https://cluster.example.cloud.qdrant.io", "api_key": "qk-test-123", "timeout": 7}
    ]


def test_remote_client_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingClient.calls = []
    monkeypatch.setattr(vectorstore, "QdrantClient", _RecordingClient)

    build_qdrant_client(Settings(_env_file=None, qdrant_url="http://localhost:6333"))

    assert _RecordingClient.calls[0]["api_key"] is None


def test_in_memory_client_when_no_url() -> None:
    client = build_qdrant_client(Settings(_env_file=None, qdrant_local_path=":memory:"))
    try:
        assert isinstance(client, QdrantClient)
        assert client.get_collections().collections == []
    finally:
        client.close()


def test_on_disk_client_when_no_url(tmp_path: Path) -> None:
    path = tmp_path / "qdrant"
    client = build_qdrant_client(Settings(_env_file=None, qdrant_local_path=str(path)))
    try:
        assert client.get_collections().collections == []
    finally:
        client.close()
    assert path.exists()
