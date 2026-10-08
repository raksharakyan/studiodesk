"""In-memory proposal store: single-use tokens, TTL, capacity, hashing, concurrency."""

import re
import threading
from datetime import UTC, datetime, timedelta

import pytest

from studiodesk.actions.proposals import (
    InMemoryProposalStore,
    ProposalCapacityError,
    ProposalExpiredError,
    ProposalNotFoundError,
    ProposalUsedError,
)
from studiodesk.models.actions import ACTION_ID_PATTERN, CONFIRM_TOKEN_PATTERN, IssueDraft

DRAFT = IssueDraft(title="Title", body="Body", labels=["bug"])
TTL = 60


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(clock: Clock) -> InMemoryProposalStore:
    return InMemoryProposalStore(TTL, max_pending=3, clock=clock)


def test_happy_path(store: InMemoryProposalStore, clock: Clock) -> None:
    proposal = store.create(DRAFT)

    assert re.fullmatch(CONFIRM_TOKEN_PATTERN, proposal.confirm_token)
    assert re.fullmatch(ACTION_ID_PATTERN, proposal.action_id)
    assert proposal.expires_at == clock.now + timedelta(seconds=TTL)
    assert store.claim(proposal.action_id, proposal.confirm_token) == DRAFT


def test_ids_and_tokens_are_unique(store: InMemoryProposalStore) -> None:
    store = InMemoryProposalStore(TTL, max_pending=100)
    proposals = [store.create(DRAFT) for _ in range(50)]
    assert len({p.action_id for p in proposals}) == 50
    assert len({p.confirm_token for p in proposals}) == 50


def test_wrong_token_and_unknown_id_are_not_found(store: InMemoryProposalStore) -> None:
    proposal = store.create(DRAFT)
    other = store.create(DRAFT)

    with pytest.raises(ProposalNotFoundError):
        store.claim(proposal.action_id, other.confirm_token)
    with pytest.raises(ProposalNotFoundError):
        store.claim("unknown-action-id-123", proposal.confirm_token)
    with pytest.raises(ProposalNotFoundError):
        store.cancel(proposal.action_id, "x" * 43)
    # A wrong guess does not burn the proposal.
    assert store.claim(proposal.action_id, proposal.confirm_token) == DRAFT


def test_reuse_is_rejected(store: InMemoryProposalStore) -> None:
    proposal = store.create(DRAFT)
    store.claim(proposal.action_id, proposal.confirm_token)

    with pytest.raises(ProposalUsedError):
        store.claim(proposal.action_id, proposal.confirm_token)
    with pytest.raises(ProposalUsedError):
        store.cancel(proposal.action_id, proposal.confirm_token)


def test_cancelled_cannot_be_confirmed(store: InMemoryProposalStore) -> None:
    proposal = store.create(DRAFT)
    store.cancel(proposal.action_id, proposal.confirm_token)

    with pytest.raises(ProposalUsedError):
        store.claim(proposal.action_id, proposal.confirm_token)
    with pytest.raises(ProposalUsedError):
        store.cancel(proposal.action_id, proposal.confirm_token)


def test_expiry_and_purge(store: InMemoryProposalStore, clock: Clock) -> None:
    proposal = store.create(DRAFT)

    clock.advance(TTL - 1)
    late = store.create(DRAFT)
    clock.advance(1)  # exactly at expiry
    with pytest.raises(ProposalExpiredError):
        store.claim(proposal.action_id, proposal.confirm_token)
    with pytest.raises(ProposalExpiredError):
        store.cancel(proposal.action_id, proposal.confirm_token)

    clock.advance(TTL)  # one TTL past expiry: purged, now unknown
    with pytest.raises(ProposalNotFoundError):
        store.claim(proposal.action_id, proposal.confirm_token)
    with pytest.raises(ProposalExpiredError):
        store.claim(late.action_id, late.confirm_token)


def test_capacity_counts_only_pending(store: InMemoryProposalStore, clock: Clock) -> None:
    first = store.create(DRAFT)
    store.create(DRAFT)
    store.create(DRAFT)
    with pytest.raises(ProposalCapacityError):
        store.create(DRAFT)

    store.claim(first.action_id, first.confirm_token)
    store.create(DRAFT)  # a slot was freed
    with pytest.raises(ProposalCapacityError):
        store.create(DRAFT)

    clock.advance(TTL)  # everything pending expires
    store.create(DRAFT)


def test_token_is_stored_only_as_a_hash(store: InMemoryProposalStore) -> None:
    proposal = store.create(DRAFT)

    state = repr(vars(store)) + "".join(repr(vars(r)) for r in vars(store)["_records"].values())

    assert proposal.confirm_token not in state
    assert proposal.confirm_token.encode() not in [
        getattr(r, "token_digest", b"") for r in vars(store)["_records"].values()
    ]


def test_concurrent_claims_succeed_exactly_once() -> None:
    store = InMemoryProposalStore(TTL, max_pending=10)
    proposal = store.create(DRAFT)
    barrier = threading.Barrier(16)
    results: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            store.claim(proposal.action_id, proposal.confirm_token)
            outcome = "ok"
        except ProposalUsedError:
            outcome = "used"
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count("ok") == 1
    assert results.count("used") == 15


@pytest.mark.parametrize(("ttl", "cap"), [(0, 1), (10, 0), (-1, 5)])
def test_invalid_construction(ttl: int, cap: int) -> None:
    with pytest.raises(ValueError):
        InMemoryProposalStore(ttl, max_pending=cap)
