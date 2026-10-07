"""resolve_client_ip, trusted_proxy_ips parsing, and per-client rate-limit buckets."""

import ipaddress
from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from studiodesk.api.ratelimit import UNKNOWN_CLIENT, resolve_client_ip
from studiodesk.config import Settings
from studiodesk.embeddings import Embedder
from studiodesk.main import create_app
from studiodesk.vectorstore import QdrantStore

V4_PROXIES = [ipaddress.ip_network("10.0.0.0/8")]
V6_PROXIES = [ipaddress.ip_network("fd00::/8")]
MIXED = V4_PROXIES + V6_PROXIES


def _resolve(peer: str | None, xff: Sequence[str], trusted: Sequence[object]) -> str:
    return resolve_client_ip(peer, xff, trusted)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("peer", "xff", "trusted", "expected"),
    [
        # Untrusted peer: XFF ignored, whatever it says.
        ("203.0.113.9", ["198.51.100.1"], V4_PROXIES, "203.0.113.9"),
        ("203.0.113.9", ["198.51.100.1"], [], "203.0.113.9"),
        ("10.0.0.1", ["198.51.100.1"], [], "10.0.0.1"),
        # Trusted peer, single XFF entry.
        ("10.0.0.1", ["198.51.100.1"], V4_PROXIES, "198.51.100.1"),
        # Spoofed leftmost entry: rightmost untrusted hop wins.
        ("10.0.0.1", ["6.6.6.6, 198.51.100.1"], V4_PROXIES, "198.51.100.1"),
        # Chain through further trusted hops.
        ("10.0.0.1", ["6.6.6.6, 198.51.100.1, 10.2.0.1, 10.3.0.1"], V4_PROXIES, "198.51.100.1"),
        # Multiple XFF headers are concatenated in order.
        ("10.0.0.1", ["6.6.6.6", "198.51.100.1, 10.2.0.1"], V4_PROXIES, "198.51.100.1"),
        # No XFF: the trusted proxy itself is the key.
        ("10.0.0.1", [], V4_PROXIES, "10.0.0.1"),
        ("10.0.0.1", ["", " , "], V4_PROXIES, "10.0.0.1"),
        # Every hop trusted: the leftmost entry is used.
        ("10.0.0.1", ["10.5.0.1, 10.6.0.1"], V4_PROXIES, "10.5.0.1"),
        # Unparsable hop stops the walk and falls back to the peer.
        ("10.0.0.1", ["198.51.100.1, not-an-ip"], V4_PROXIES, "10.0.0.1"),
        ("10.0.0.1", ["garbage, 10.2.0.1"], V4_PROXIES, "10.0.0.1"),
        ("10.0.0.1", ["198.51.100.1:443"], V4_PROXIES, "10.0.0.1"),
        # Whitespace is tolerated.
        ("10.0.0.1", ["  198.51.100.1  "], V4_PROXIES, "198.51.100.1"),
        # IPv6, normalised.
        ("fd00::1", ["2001:DB8::0001"], V6_PROXIES, "2001:db8::1"),
        ("fd00::1", ["2001:db8::1, fd12::7"], V6_PROXIES, "2001:db8::1"),
        # Mixed v4/v6: v4 trust does not cover a v6 peer, and vice versa.
        ("fd00::1", ["198.51.100.1"], V4_PROXIES, "fd00::1"),
        ("10.0.0.1", ["198.51.100.1"], V6_PROXIES, "10.0.0.1"),
        ("10.0.0.1", ["2001:db8::1, fd00::2"], MIXED, "2001:db8::1"),
        ("fd00::1", ["198.51.100.1, 10.9.9.9"], MIXED, "198.51.100.1"),
        # Non-IP peer (e.g. a unix socket / TestClient name) is used as-is.
        ("testclient", ["198.51.100.1"], V4_PROXIES, "testclient"),
    ],
)
def test_resolve_client_ip(peer: str, xff: list[str], trusted: list[object], expected: str) -> None:
    assert _resolve(peer, xff, trusted) == expected


def test_missing_peer_is_unknown() -> None:
    assert _resolve(None, ["198.51.100.1"], V4_PROXIES) == UNKNOWN_CLIENT == "unknown"


def test_single_host_trust() -> None:
    trusted = [ipaddress.ip_network("192.0.2.7/32")]

    assert _resolve("192.0.2.7", ["198.51.100.1"], trusted) == "198.51.100.1"
    assert _resolve("192.0.2.8", ["198.51.100.1"], trusted) == "192.0.2.8"


# --- settings parsing --------------------------------------------------------------------


@pytest.fixture
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRUSTED_PROXY_IPS", raising=False)


@pytest.mark.usefixtures("_no_proxy_env")
def test_default_trusts_nobody() -> None:
    assert Settings(_env_file=None).trusted_proxy_ips == []


def test_comma_separated_env_with_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTED_PROXY_IPS", " 10.0.0.0/8 , 192.0.2.7 ,, fd00::/8 ")

    nets = Settings(_env_file=None).trusted_proxy_ips

    assert [str(n) for n in nets] == ["10.0.0.0/8", "192.0.2.7/32", "fd00::/8"]


@pytest.mark.parametrize("value", ["", "  ", " , "])
def test_empty_env_gives_empty_list(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TRUSTED_PROXY_IPS", value)

    assert Settings(_env_file=None).trusted_proxy_ips == []


@pytest.mark.usefixtures("_no_proxy_env")
def test_list_input() -> None:
    s = Settings(_env_file=None, trusted_proxy_ips=["10.0.0.0/8", "::1"])  # type: ignore[list-item]

    assert [str(n) for n in s.trusted_proxy_ips] == ["10.0.0.0/8", "::1/128"]


@pytest.mark.usefixtures("_no_proxy_env")
@pytest.mark.parametrize(
    "value",
    [
        "10.0.0.0/33",
        "not-an-ip",
        "10.0.0.1/8",  # host bits set
        "fd00::1/8",  # host bits set (v6)
        "10.0.0.0/8;rm -rf",
        "*",
    ],
)
def test_invalid_entries_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, trusted_proxy_ips=value)  # type: ignore[arg-type]


@pytest.mark.usefixtures("_no_proxy_env")
def test_at_most_64_entries() -> None:
    nets = [f"10.0.{i}.0/24" for i in range(65)]

    assert (
        len(Settings(_env_file=None, trusted_proxy_ips=",".join(nets[:64])).trusted_proxy_ips) == 64
    )  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Settings(_env_file=None, trusted_proxy_ips=",".join(nets))  # type: ignore[arg-type]


# --- end to end: rate-limit buckets ------------------------------------------------------


def _limited_client(
    trusted: str, fake_embedder: Embedder, ingested_store: QdrantStore
) -> TestClient:
    settings = Settings(
        _env_file=None, app_env="test", search_rate_limit="1/minute", trusted_proxy_ips=trusted
    )  # type: ignore[arg-type]
    app = create_app(settings, embedder=fake_embedder, store=ingested_store)
    return TestClient(app, client=("10.1.1.1", 5000))


def _post(c: TestClient, xff: str | None) -> int:
    headers = {"x-forwarded-for": xff} if xff else {}
    return c.post("/search", json={"query": "save"}, headers=headers).status_code


def test_trusted_proxy_gives_each_forwarded_client_its_own_bucket(
    fake_embedder: Embedder, ingested_store: QdrantStore
) -> None:
    with _limited_client("10.0.0.0/8", fake_embedder, ingested_store) as c:
        assert _post(c, "198.51.100.1") == 200
        assert _post(c, "198.51.100.1") == 429
        assert _post(c, "198.51.100.2") == 200
        # Spoofing the leftmost entry does not buy a fresh bucket.
        assert _post(c, "6.6.6.6, 198.51.100.2") == 429
        # No XFF: keyed on the proxy itself.
        assert _post(c, None) == 200
        assert _post(c, None) == 429


def test_without_trust_everyone_shares_the_peer_bucket(
    fake_embedder: Embedder, ingested_store: QdrantStore
) -> None:
    with _limited_client("", fake_embedder, ingested_store) as c:
        assert _post(c, "198.51.100.1") == 200
        assert _post(c, "198.51.100.2") == 429
        assert _post(c, None) == 429
