"""In-memory proposal store: single-use tokens, TTL, capacity, hashing, concurrency."""

import re
import threading
from datetime import UTC, datetime, timedelta

import pytest

from studiodesk.actions.proposals import (
    InMemoryProposalStore,
    ProposalCapacityError,
    ProposalDailyLimitError,
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


# --------------------------------------------------------------------------- daily cap


def capped(clock: Clock, n: int) -> InMemoryProposalStore:
    return InMemoryProposalStore(900, max_pending=100, max_per_day=n, clock=clock)


def test_daily_cap_blocks_n_plus_one_and_keeps_proposal_pending(clock: Clock) -> None:
    clock.now = datetime(2026, 1, 1, 23, 50, tzinfo=UTC)
    store = capped(clock, 2)
    proposals = [store.create(DRAFT) for _ in range(3)]
    for p in proposals[:2]:
        store.claim(p.action_id, p.confirm_token)
        store.release(p.action_id, created=True)

    assert store.daily_limit_reached() is True
    last = proposals[2]
    with pytest.raises(ProposalDailyLimitError):
        store.claim(last.action_id, last.confirm_token)

    clock.now = datetime(2026, 1, 2, 0, 0, 5, tzinfo=UTC)  # UTC rollover, within TTL
    assert store.daily_limit_reached() is False
    assert store.claim(last.action_id, last.confirm_token) == DRAFT


def test_reserved_slots_count_and_failed_creation_frees_them(clock: Clock) -> None:
    store = capped(clock, 2)
    a, b, c = (store.create(DRAFT) for _ in range(3))
    store.claim(a.action_id, a.confirm_token)
    store.claim(b.action_id, b.confirm_token)  # both reserved, none created yet

    with pytest.raises(ProposalDailyLimitError):
        store.claim(c.action_id, c.confirm_token)

    store.release(a.action_id, created=False)  # e.g. GitHub 5xx
    assert store.claim(c.action_id, c.confirm_token) == DRAFT


def test_only_created_releases_count(clock: Clock) -> None:
    store = capped(clock, 1)
    for _ in range(5):
        p = store.create(DRAFT)
        store.claim(p.action_id, p.confirm_token)
        store.release(p.action_id, created=False)
    assert store.daily_limit_reached() is False
    p = store.create(DRAFT)
    store.claim(p.action_id, p.confirm_token)
    store.release(p.action_id, created=True)
    assert store.daily_limit_reached() is True


def test_slot_reserved_before_midnight_does_not_count_for_new_day(clock: Clock) -> None:
    clock.now = datetime(2026, 1, 1, 23, 59, tzinfo=UTC)
    store = capped(clock, 1)
    a, b = store.create(DRAFT), store.create(DRAFT)
    store.claim(a.action_id, a.confirm_token)

    clock.now = datetime(2026, 1, 2, 0, 1, tzinfo=UTC)
    assert store.daily_limit_reached() is False  # yesterday's reservation is not today's
    store.release(a.action_id, created=True)  # counted for its own (previous) day only
    assert store.daily_limit_reached() is False
    assert store.claim(b.action_id, b.confirm_token) == DRAFT
    assert store.daily_limit_reached() is True


def test_release_of_unknown_id_is_harmless(clock: Clock) -> None:
    store = capped(clock, 1)
    store.release("never-claimed", created=True)
    assert store.daily_limit_reached() is False


def test_no_cap_by_default(clock: Clock) -> None:
    store = InMemoryProposalStore(TTL, max_pending=100, clock=clock)
    for _ in range(50):
        p = store.create(DRAFT)
        store.claim(p.action_id, p.confirm_token)
        store.release(p.action_id, created=True)
    assert store.daily_limit_reached() is False


def test_concurrent_claims_never_exceed_daily_cap() -> None:
    store = InMemoryProposalStore(900, max_pending=100, max_per_day=4)
    proposals = [store.create(DRAFT) for _ in range(24)]
    barrier = threading.Barrier(len(proposals))
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(action_id: str, token: str) -> None:
        barrier.wait()
        try:
            store.claim(action_id, token)
            store.release(action_id, created=True)
            outcome = "ok"
        except ProposalDailyLimitError:
            outcome = "limit"
        with lock:
            outcomes.append(outcome)

    threads = [
        threading.Thread(target=worker, args=(p.action_id, p.confirm_token)) for p in proposals
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count("ok") == 4
    assert outcomes.count("limit") == 20


def test_invalid_daily_cap() -> None:
    with pytest.raises(ValueError):
        InMemoryProposalStore(TTL, max_pending=1, max_per_day=0)
