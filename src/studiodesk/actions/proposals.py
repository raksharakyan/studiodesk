"""Propose-then-confirm: a server-side store of pending actions with single-use tokens.

`/bugs/check` stores the exact `IssueDraft` and returns an `action_id` plus a
`confirm_token` (`secrets.token_urlsafe(32)`); only a SHA-256 digest of the token is kept
and it is compared with `hmac.compare_digest`. Confirm executes the stored draft, never
anything from the confirm request. A proposal moves from `pending` to exactly one of
`confirmed`, `cancelled` or `expired`; the token is single use.

Errors map to HTTP status codes in the API: unknown id or wrong token -> 404 (existence is
not revealed without the token), already confirmed/cancelled -> 409, expired -> 410, store
full -> 503. Records are kept for one extra TTL after expiry so late or repeated requests
still get 409/410 rather than 404, then purged.

`InMemoryProposalStore` keeps proposals in process memory, like the rate limiter in
`studiodesk.api.ratelimit`: with N uvicorn workers or N replicas, a confirm only succeeds
on the process that created the proposal (elsewhere it is a 404), and a restart drops all
pending proposals (users simply re-run `/bugs/check`). That is acceptable for the current
single-process deployment; scaling out needs a shared backend (e.g. Redis with per-key
TTLs and an atomic compare-and-set for `claim`) behind the same `ProposalStore` protocol.
"""

import hashlib
import hmac
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from studiodesk.models.actions import IssueDraft

TOKEN_BYTES = 32
ACTION_ID_BYTES = 16

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """Current time as an aware UTC datetime (the default clock)."""
    return datetime.now(UTC)


class ProposalState(StrEnum):
    """Lifecycle of a proposal."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ProposalError(Exception):
    """Base class for proposal lookup failures."""


class ProposalNotFoundError(ProposalError):
    """Unknown action id, or the token does not match (404)."""


class ProposalUsedError(ProposalError):
    """The proposal was already confirmed or cancelled (409)."""


class ProposalExpiredError(ProposalError):
    """The proposal expired before it was confirmed (410)."""


class ProposalCapacityError(ProposalError):
    """Too many pending proposals are held (503)."""


@dataclass(frozen=True)
class NewProposal:
    """What the creator receives once: the id, the plain token and the expiry."""

    action_id: str
    confirm_token: str
    expires_at: datetime


@dataclass
class _Record:
    """Stored proposal: the draft, the token digest, expiry and state."""

    draft: IssueDraft
    token_digest: bytes
    expires_at: datetime
    state: ProposalState = ProposalState.PENDING


class ProposalStore(Protocol):
    """Holds proposed actions until they are confirmed, cancelled or expire."""

    def create(self, draft: IssueDraft) -> NewProposal:
        """Store `draft` as pending and return its id, single-use token and expiry."""
        ...

    def claim(self, action_id: str, token: str) -> IssueDraft:
        """Atomically mark the proposal confirmed and return its stored draft.

        Raises:
            ProposalError: subclass for not found / used / expired.
        """
        ...

    def cancel(self, action_id: str, token: str) -> None:
        """Mark the proposal cancelled.

        Raises:
            ProposalError: subclass for not found / used / expired.
        """
        ...


def _digest(token: str) -> bytes:
    """SHA-256 of the token, so plain tokens are never held server-side."""
    return hashlib.sha256(token.encode("utf-8")).digest()


class InMemoryProposalStore:
    """Thread-safe in-memory `ProposalStore` with a TTL and a cap on held records."""

    def __init__(self, ttl_s: int, *, max_pending: int, clock: Clock = utc_now) -> None:
        """Proposals expire `ttl_s` seconds after creation; at most `max_pending` are held."""
        if ttl_s <= 0 or max_pending <= 0:
            raise ValueError("ttl_s and max_pending must be positive")
        self._ttl = timedelta(seconds=ttl_s)
        self._max_pending = max_pending
        self._clock = clock
        self._records: dict[str, _Record] = {}
        self._lock = threading.Lock()

    def create(self, draft: IssueDraft) -> NewProposal:
        """Store `draft` as pending; raises `ProposalCapacityError` when full."""
        token = secrets.token_urlsafe(TOKEN_BYTES)
        action_id = secrets.token_urlsafe(ACTION_ID_BYTES)
        with self._lock:
            now = self._clock()
            self._purge(now)
            pending = sum(r.state is ProposalState.PENDING for r in self._records.values())
            if pending >= self._max_pending:
                raise ProposalCapacityError("too many pending proposals")
            expires_at = now + self._ttl
            self._records[action_id] = _Record(draft, _digest(token), expires_at)
        return NewProposal(action_id=action_id, confirm_token=token, expires_at=expires_at)

    def claim(self, action_id: str, token: str) -> IssueDraft:
        """Confirm the proposal (single use) and return the stored draft."""
        with self._lock:
            record = self._pending_record(action_id, token)
            record.state = ProposalState.CONFIRMED
            return record.draft

    def cancel(self, action_id: str, token: str) -> None:
        """Cancel the pending proposal."""
        with self._lock:
            record = self._pending_record(action_id, token)
            record.state = ProposalState.CANCELLED

    def _pending_record(self, action_id: str, token: str) -> _Record:
        """Return the record if the token matches and it is still pending (lock held)."""
        now = self._clock()
        self._purge(now)
        record = self._records.get(action_id)
        if record is None or not hmac.compare_digest(record.token_digest, _digest(token)):
            raise ProposalNotFoundError("unknown action or token")
        if record.state is ProposalState.PENDING and now >= record.expires_at:
            record.state = ProposalState.EXPIRED
        if record.state is ProposalState.EXPIRED:
            raise ProposalExpiredError("proposal expired")
        if record.state is not ProposalState.PENDING:
            raise ProposalUsedError(f"proposal already {record.state.value}")
        return record

    def _purge(self, now: datetime) -> None:
        """Drop records more than one TTL past their expiry; expire stale pending ones."""
        for action_id, record in list(self._records.items()):
            if record.state is ProposalState.PENDING and now >= record.expires_at:
                record.state = ProposalState.EXPIRED
            if now >= record.expires_at + self._ttl:
                del self._records[action_id]
