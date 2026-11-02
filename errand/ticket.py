"""Signed errand tickets: the only thing that crosses from the bank side to the errand agent."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

Category = Literal["flowers", "cake"]

MAX_BUDGET_FEN = 100_000

_NOTE_RE = re.compile(r"[A-Za-z一-鿿 '\-]*")
_CARD_TOKEN_RE = re.compile(r"<VCARD_[1-9][0-9]{0,2}>")


class TicketError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ErrandTicket:
    ticket_id: str
    category: Category
    budget_fen: int
    deliver_by: date
    card_token: str
    issued_at: datetime
    expires_at: datetime
    note: str = ""


def _canonical(ticket: ErrandTicket) -> bytes:
    payload = {
        "ticket_id": ticket.ticket_id,
        "category": ticket.category,
        "budget_fen": ticket.budget_fen,
        "deliver_by": ticket.deliver_by.isoformat(),
        "card_token": ticket.card_token,
        "issued_at": ticket.issued_at.isoformat(),
        "expires_at": ticket.expires_at.isoformat(),
        "note": ticket.note,
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sign_ticket(ticket: ErrandTicket, key: bytes) -> str:
    return hmac.new(key, _canonical(ticket), hashlib.sha256).hexdigest()


def issue_ticket(
    *,
    category: Category,
    budget_fen: int,
    deliver_by: date,
    now: datetime,
    key: bytes,
    card_token: str = "<VCARD_1>",
    ttl_minutes: int = 60,
    note: str = "",
    ticket_id: str | None = None,
) -> tuple[ErrandTicket, str]:
    if category not in ("flowers", "cake"):
        raise TicketError("invalid")
    if not isinstance(budget_fen, int) or isinstance(budget_fen, bool):
        raise TicketError("invalid")
    if not 1 <= budget_fen <= MAX_BUDGET_FEN:
        raise TicketError("invalid")
    if deliver_by < now.date():
        raise TicketError("invalid")
    if not isinstance(ttl_minutes, int) or isinstance(ttl_minutes, bool):
        raise TicketError("invalid")
    if not 1 <= ttl_minutes <= 240:
        raise TicketError("invalid")
    if not isinstance(note, str):
        raise TicketError("invalid")
    cleaned_note = note.strip()
    if len(cleaned_note) > 20 or _NOTE_RE.fullmatch(cleaned_note) is None:
        raise TicketError("invalid")
    if not isinstance(card_token, str) or _CARD_TOKEN_RE.fullmatch(card_token) is None:
        raise TicketError("invalid")

    ticket = ErrandTicket(
        ticket_id=ticket_id if ticket_id is not None else "T" + secrets.token_hex(5),
        category=category,
        budget_fen=budget_fen,
        deliver_by=deliver_by,
        card_token=card_token,
        issued_at=now,
        expires_at=now + timedelta(minutes=ttl_minutes),
        note=cleaned_note,
    )
    return ticket, sign_ticket(ticket, key)


def verify_ticket(
    ticket: ErrandTicket, signature: str, key: bytes, now: datetime
) -> None:
    if not isinstance(now, datetime) or now.tzinfo is not None:
        raise TicketError("invalid")
    expected = sign_ticket(ticket, key)
    if not isinstance(signature, str) or not hmac.compare_digest(
        expected.encode("utf-8"), signature.encode("utf-8")
    ):
        raise TicketError("bad_signature")
    if now >= ticket.expires_at:
        raise TicketError("expired")


def ticket_brief(ticket: ErrandTicket, today: date) -> str:
    brief = {
        "ticket_id": ticket.ticket_id,
        "category": ticket.category,
        "budget_fen": ticket.budget_fen,
        "budget_yuan": f"{ticket.budget_fen // 100}.{ticket.budget_fen % 100:02d}",
        "deliver_by": ticket.deliver_by.isoformat(),
        "today": today.isoformat(),
        "card_token": ticket.card_token,
        "note": ticket.note,
    }
    return json.dumps(brief, sort_keys=True, ensure_ascii=False)
