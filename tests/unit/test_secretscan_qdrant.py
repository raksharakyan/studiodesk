"""The secret scan flags real Qdrant Cloud cluster hosts but not the documented placeholder."""

import pytest

from studiodesk.secretscan import TOKEN_PATTERNS

QDRANT_PATTERN = dict(TOKEN_PATTERNS)["Qdrant Cloud cluster URL"]


@pytest.mark.parametrize(
    "text",
    [
        b"https://6bbd246a-e957-4762-955e-6eca328365bf.eu-west-2-0.aws.cloud.qdrant.io:6333",
        b"QDRANT_URL=https://abc123.us-east4-0.gcp.cloud.qdrant.io",
        b'{"url": "https://x.cloud.qdrant.io"}',
    ],
)
def test_real_cluster_hosts_are_flagged(text: bytes) -> None:
    assert QDRANT_PATTERN.search(text)


@pytest.mark.parametrize(
    "text",
    [
        b"https://your-cluster.cloud.qdrant.io:6333",
        b"see https://cloud.qdrant.io for a free cluster",
        b"qdrant.io",
    ],
)
def test_placeholder_and_docs_links_are_not_flagged(text: bytes) -> None:
    assert not QDRANT_PATTERN.search(text)
