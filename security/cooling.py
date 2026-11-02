"""Delayed-transfer cooling queue with cancellation windows.

Every write gets a reversible cooling window.  The executor stores
``created_at``/``release_at`` and the UI uses :func:`describe_window` to show
how long is left and whether an undo is still possible.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

ZERO = timedelta(0)
TWO_HOURS = timedelta(hours=2)
TWENTY_FOUR_HOURS = timedelta(hours=24)

SCAM_HOLD = TWENTY_FOUR_HOURS


def cooling_for(
    tier: str,
    *,
    scam_flags: frozenset[str] = frozenset(),
    elder: bool = False,
    arrival_delay: timedelta = ZERO,
) -> timedelta:
    """Return the cooling period for a step.

    * ``green`` writes run immediately.
    * ``yellow`` and ``red`` writes wait two hours; red is reversible for the
      same window.
    * An elder's red action waits 24 hours.
    * Any active scam flag holds the action for 24 hours.
    """

    # PBOC 261: the customer's chosen delayed arrival is a floor; it never
    # shortens the tier's own window.
    return max(_tier_window(tier, scam_flags=scam_flags, elder=elder), arrival_delay)


def _tier_window(tier: str, *, scam_flags: frozenset[str], elder: bool) -> timedelta:
    if scam_flags:
        return SCAM_HOLD
    if tier == "green":
        return ZERO
    if tier == "yellow":
        return TWO_HOURS
    if tier == "red":
        return TWENTY_FOUR_HOURS if elder else TWO_HOURS
    raise ValueError(f"unknown tier: {tier!r}")


def _as_datetime(value: date | datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError("expected a date, datetime or ISO datetime string")


def describe_window(
    created_at: date | datetime | str,
    release_at: date | datetime | str,
    now: date | datetime | str,
) -> dict[str, object]:
    """Describe a cooling window for the UI."""

    _as_datetime(created_at)
    release = _as_datetime(release_at)
    current = _as_datetime(now)

    seconds_left = int((release - current).total_seconds())
    if seconds_left < 0:
        seconds_left = 0

    return {
        "seconds_left": seconds_left,
        "can_undo": current < release,
        "release_at": release.isoformat(),
    }
