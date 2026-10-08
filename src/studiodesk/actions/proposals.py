"""Propose-then-confirm: a server-side store of pending actions with single-use tokens.

`/bugs/check` stores the exact `IssueDraft` and returns an `action_id` plus a
`confirm_token` (`secrets.token_urlsafe(32)`); only a SHA-256 digest of the token is kept
and it is compared with `hmac.compare_digest`. Confirm executes the stored draft, never
anything from the confirm request. A proposal moves from `pending` to exactly one of
`confirmed`, `cancelled` or `expired`; the token is single use.

Errors map to HTTP status codes in the API: unknown id or wrong token -> 404 (existence is
not revealed without the token), already confirmed/cancelled -> 409, expired -> 410, store
full or daily cap reached -> 503. Records are kept for one extra TTL after expiry so late
or repeated requests still get 409/410 rather than 404, then purged.

Daily cap: with `max_per_day` set, at most that many issues are filed per UTC day.
`claim` reserves a slot under the same lock that marks the proposal confirmed (so
concurrent confirms cannot overshoot), and the caller must `release` it afterwards with
`created=True` only if the issue was actually created; failed creations free the slot.
When no slot is left, `claim` raises `ProposalDailyLimitError` and the proposal stays
pending (its token is not consumed).

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
from datetime import UTC, date, datetime, timedelta
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


class ProposalDailyLimitError(ProposalError):
    """The daily cap on filed issues is reached (503); the proposal stays pending."""


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

    def release(self, action_id: str, *, created: bool) -> None:
        """Release the daily slot reserved by `claim`, counting it only if `created`."""
        ...

    def daily_limit_reached(self) -> bool:
        """True if no daily slot is left (created plus reserved >= cap)."""
        ...


def _digest(token: str) -> bytes:
    """SHA-256 of the token, so plain tokens are never held server-side."""
    return hashlib.sha256(token.encode("utf-8")).digest()


class InMemoryProposalStore:
    """Thread-safe in-memory `ProposalStore` with a TTL, a cap on held records and an
    optional daily cap on filed issues (UTC day)."""

    def __init__(
        self,
        ttl_s: int,
        *,
        max_pending: int,
        max_per_day: int | None = None,
        clock: Clock = utc_now,
    ) -> None:
        """Proposals expire `ttl_s` seconds after creation; at most `max_pending` are held.

        Args:
            ttl_s: Seconds a proposal can be confirmed.
            max_pending: Maximum pending proposals held at once.
            max_per_day: Maximum issues filed per UTC day (None = no cap).
            clock: Returns the current aware UTC datetime (injectable for tests).
        """
        if ttl_s <= 0 or max_pending <= 0 or (max_per_day is not None and max_per_day <= 0):
            raise ValueError("ttl_s, max_pending and max_per_day must be positive")
        self._ttl = timedelta(seconds=ttl_s)
        self._max_pending = max_pending
        self._max_per_day = max_per_day
        self._clock = clock
        self._records: dict[str, _Record] = {}
        self._lock = threading.Lock()
        self._day: date = clock().date()
        self._created_today = 0
        self._reserved: dict[str, date] = {}

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
        """Confirm the proposal (single use), reserve a daily slot, return the draft.

        Raises:
            ProposalDailyLimitError: no slot left today (the proposal stays pending).
        """
        with self._lock:
            record = self._pending_record(action_id, token)
            if self._max_per_day is not None and self._slots_used() >= self._max_per_day:
                raise ProposalDailyLimitError("daily issue limit reached")
            record.state = ProposalState.CONFIRMED
            self._reserved[action_id] = self._day
            return record.draft

    def release(self, action_id: str, *, created: bool) -> None:
        """Free the slot reserved by `claim`; count it for its day only if `created`."""
        with self._lock:
            day = self._reserved.pop(action_id, None)
            self._roll_day()
            if created and day == self._day:
                self._created_today += 1

    def daily_limit_reached(self) -> bool:
        """True if the daily cap is set and created plus reserved slots reach it."""
        with self._lock:
            return self._max_per_day is not None and self._slots_used() >= self._max_per_day

    def _roll_day(self) -> None:
        """Reset the daily count when the UTC day changes (lock held)."""
        today = self._clock().date()
        if today != self._day:
            self._day = today
            self._created_today = 0

    def _slots_used(self) -> int:
        """Issues created today plus slots reserved today (lock held)."""
        self._roll_day()
        return self._created_today + sum(day == self._day for day in self._reserved.values())

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
