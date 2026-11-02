"""Tests for signed errand tickets."""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timezone

import pytest

from errand.ticket import (
    MAX_BUDGET_FEN,
    ErrandTicket,
    TicketError,
    issue_ticket,
    sign_ticket,
    ticket_brief,
    verify_ticket,
)

NOW = datetime(2026, 11, 1, 9, 0, 0)
TODAY = date(2026, 11, 1)
KEY = b"errand-test-key-0123456789"
OTHER_KEY = b"wrong-key-9876543210"


def _issue(**overrides):
    kwargs = dict(
        category="flowers",
        budget_fen=29900,
        deliver_by=date(2026, 11, 3),
        now=NOW,
        key=KEY,
        card_token="<VCARD_7>",
        ttl_minutes=60,
        note="红玫瑰",
    )
    kwargs.update(overrides)
    return issue_ticket(**kwargs)


def test_issue_and_verify_passes():
    ticket, signature = _issue()
    assert isinstance(ticket, ErrandTicket)
    assert ticket.ticket_id.startswith("T")
    assert len(ticket.ticket_id) == 11
    assert ticket.issued_at == NOW
    assert ticket.expires_at == datetime(2026, 11, 1, 10, 0, 0)
    assert ticket.note == "红玫瑰"
    assert signature == sign_ticket(ticket, KEY)
    verify_ticket(ticket, signature, KEY, NOW)


def test_tampered_budget_bad_signature():
    ticket, signature = _issue()
    tampered = dataclasses.replace(ticket, budget_fen=50000)
    with pytest.raises(TicketError) as exc:
        verify_ticket(tampered, signature, KEY, NOW)
    assert exc.value.code == "bad_signature"


def test_wrong_key_bad_signature():
    ticket, signature = _issue()
    with pytest.raises(TicketError) as exc:
        verify_ticket(ticket, signature, OTHER_KEY, NOW)
    assert exc.value.code == "bad_signature"


def test_expired_at_exactly_expires_at():
    ticket, signature = _issue()
    with pytest.raises(TicketError) as exc:
        verify_ticket(ticket, signature, KEY, ticket.expires_at)
    assert exc.value.code == "expired"
    verify_ticket(ticket, signature, KEY, NOW)


def test_bad_signature_wins_over_expired():
    ticket, signature = _issue()
    tampered = dataclasses.replace(ticket, card_token="<VCARD_2>")
    with pytest.raises(TicketError) as exc:
        verify_ticket(tampered, signature, KEY, datetime(2030, 1, 1))
    assert exc.value.code == "bad_signature"


@pytest.mark.parametrize("budget", [0, 100_001, True, 99.5])
def test_invalid_budget(budget):
    with pytest.raises(TicketError) as exc:
        _issue(budget_fen=budget)
    assert exc.value.code == "invalid"


def test_max_budget_allowed():
    ticket, signature = _issue(budget_fen=MAX_BUDGET_FEN)
    verify_ticket(ticket, signature, KEY, NOW)


def test_deliver_by_yesterday_invalid():
    with pytest.raises(TicketError) as exc:
        _issue(deliver_by=date(2026, 10, 31))
    assert exc.value.code == "invalid"


@pytest.mark.parametrize(
    "note",
    [
        "<VCARD_9>",
        "line1\nline2",
        "a" * 21,
        "quote\"here",
        "brace{}",
        "Zhang Wei 13800138000",
        "roses, please",
        "roses. please",
    ],
)
def test_invalid_note(note):
    with pytest.raises(TicketError) as exc:
        _issue(note=note)
    assert exc.value.code == "invalid"


def test_note_allows_letters_cjk_spaces_apostrophe_and_hyphen():
    ticket, signature = _issue(note="Zhang's red-roses")
    assert ticket.note == "Zhang's red-roses"
    verify_ticket(ticket, signature, KEY, NOW)


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc),
        "2026-11-01",
        date(2026, 11, 1),
        None,
    ],
)
def test_verify_ticket_requires_a_naive_datetime(now):
    ticket, signature = _issue()
    with pytest.raises(TicketError) as exc:
        verify_ticket(ticket, signature, KEY, now)
    assert exc.value.code == "invalid"


def test_chinese_note_ok():
    ticket, signature = _issue(note="红玫瑰")
    assert ticket.note == "红玫瑰"
    verify_ticket(ticket, signature, KEY, NOW)


@pytest.mark.parametrize("token", ["<VCARD_0>", "VCARD_1", "<VCARD_1000>"])
def test_bad_card_token_invalid(token):
    with pytest.raises(TicketError) as exc:
        _issue(card_token=token)
    assert exc.value.code == "invalid"


@pytest.mark.parametrize("ttl", [0, 241])
def test_bad_ttl_invalid(ttl):
    with pytest.raises(TicketError) as exc:
        _issue(ttl_minutes=ttl)
    assert exc.value.code == "invalid"


def test_invalid_category():
    with pytest.raises(TicketError) as exc:
        _issue(category="pizza")
    assert exc.value.code == "invalid"


def test_ticket_brief_contents():
    ticket, signature = _issue(note="roses")
    brief = ticket_brief(ticket, TODAY)
    assert '"budget_yuan":"299.00"' in brief.replace(" ", "")
    assert "299.00" in brief
    assert ticket.ticket_id in brief
    assert "<VCARD_7>" in brief
    assert "2026-11-03" in brief
    assert "2026-11-01" in brief
    assert signature not in brief
    assert KEY.decode() not in brief
