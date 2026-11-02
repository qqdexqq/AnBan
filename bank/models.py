"""Dataclasses for customers, accounts, transactions, subscriptions and cards."""

from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class Customer:
    id: str
    persona_id: str
    name: str
    phone: str


@dataclass
class Contact:
    id: str
    name: str
    relation: str
    note: str
    account_no: str
    phone: str
    bank_name: str
    birthday: date | None
    verified: bool


@dataclass
class Account:
    id: str
    type: str  # "current" | "savings"
    balance_fen: int


@dataclass
class Pot:
    id: str
    name: str
    balance_fen: int


@dataclass
class Card:
    id: str
    kind: str  # "physical" | "virtual"
    last4: str
    status: str
    online_limit_fen: int
    account_id: str


@dataclass
class Subscription:
    id: str
    merchant: str
    amount_fen: int
    day_of_month: int
    status: str
    last_used_at: date


@dataclass
class Transaction:
    id: str
    account_id: str
    ts: datetime
    amount_fen: int  # signed: positive = money in, negative = money out
    counterparty: str
    category: str
    memo: str
    channel: str
    country: str
    card_present: bool


@dataclass
class Product:
    id: str
    name: str
    risk_level: int  # 1-5
    annual_rate_bp: int
    min_fen: int


@dataclass
class Holding:
    product_id: str
    principal_fen: int
    value_fen: int


@dataclass
class Transfer:
    id: str
    to_contact_id: str
    amount_fen: int
    note: str
    idempotency_key: str
    status: str
    created_at: datetime
    release_at: datetime
    account_id: str


@dataclass
class ScheduledTransfer:
    id: str
    to_contact_id: str
    amount_fen: int
    day_of_month: int
    note: str
    status: str


@dataclass
class PaymentRequest:
    id: str
    from_contact_id: str
    amount_fen: int
    note: str
    status: str


@dataclass
class MerchantOrder:
    id: str
    merchant: str
    item: str
    amount_fen: int
    deliver_on: date
    status: str
    created_at: datetime
    funding_pot_id: str | None = None


@dataclass
class BankState:
    customer: Customer
    contacts: list[Contact]
    accounts: list[Account]
    pots: list[Pot]
    cards: list[Card]
    subscriptions: list[Subscription]
    transactions: list[Transaction]
    products: list[Product]
    holdings: list[Holding]
