"""Process-wide daily cap on minted voice sessions (UTC day).

Each `GET /voice/session` reserves one slot with `reserve()` *before* calling ElevenLabs,
under a lock, so concurrent requests can never overshoot the cap. If the upstream call
fails, the caller gives the slot back with `release()`; a successful call keeps it. The
count resets when the UTC day changes; a reservation from an earlier day is never
released against the new day's count.

Like the proposal store, the counter lives in process memory: with N workers or replicas
the effective cap is N times the setting, and a restart resets it.
"""

import math
import threading
from datetime import date, datetime, time, timedelta

from studiodesk.actions.proposals import Clock, utc_now


class VoiceDailyLimitError(Exception):
    """No voice session slot is left today; `retry_after_s` is the wait until UTC midnight."""

    def __init__(self, retry_after_s: int) -> None:
        """Store the seconds until the next UTC day starts (at least 1)."""
        super().__init__("daily voice session limit reached")
        self.retry_after_s = retry_after_s


def seconds_until_utc_midnight(now: datetime) -> int:
    """Whole seconds from aware UTC `now` until the next UTC midnight (at least 1)."""
    midnight = datetime.combine(now.date() + timedelta(days=1), time.min, tzinfo=now.tzinfo)
    return max(math.ceil((midnight - now).total_seconds()), 1)


class VoiceSessionQuota:
    """Thread-safe counter of voice sessions started per UTC day, capped at `max_per_day`."""

    def __init__(self, max_per_day: int, *, clock: Clock = utc_now) -> None:
        """Allow at most `max_per_day` sessions per UTC day.

        Args:
            max_per_day: Maximum sessions (reserved or used) per UTC day; must be positive.
            clock: Returns the current aware UTC datetime (injectable for tests).
        """
        if max_per_day <= 0:
            raise ValueError("max_per_day must be positive")
        self._max_per_day = max_per_day
        self._clock = clock
        self._lock = threading.Lock()
        self._day: date = clock().date()
        self._used = 0

    def reserve(self) -> date:
        """Atomically take one slot for today and return the day it was counted against.

        Raises:
            VoiceDailyLimitError: when today's slots are used up (nothing is reserved).
        """
        with self._lock:
            now = self._clock()
            self._roll_day(now.date())
            if self._used >= self._max_per_day:
                raise VoiceDailyLimitError(seconds_until_utc_midnight(now))
            self._used += 1
            return self._day

    def release(self, day: date) -> None:
        """Give back a slot taken by `reserve` on `day` (no-op if the day has rolled over)."""
        with self._lock:
            self._roll_day(self._clock().date())
            if day == self._day and self._used > 0:
                self._used -= 1

    def used_today(self) -> int:
        """Slots taken today (reserved or used)."""
        with self._lock:
            self._roll_day(self._clock().date())
            return self._used

    def remaining_today(self) -> int:
        """Slots still free today (read-only; never negative)."""
        with self._lock:
            self._roll_day(self._clock().date())
            return max(self._max_per_day - self._used, 0)

    def _roll_day(self, today: date) -> None:
        """Reset the count when the UTC day changes (lock held)."""
        if today != self._day:
            self._day = today
            self._used = 0
