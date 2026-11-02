"""SQLite persistence for the mock core bank.

All customer-owned tables are scoped by ``customer_id``. Products are global.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

from .models import BankState

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,
    persona_id TEXT NOT NULL,
    name TEXT NOT NULL,
    phone TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS contacts (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    id TEXT NOT NULL,
    name TEXT NOT NULL,
    relation TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    account_no TEXT NOT NULL,
    phone TEXT NOT NULL,
    bank_name TEXT NOT NULL,
    birthday TEXT,
    verified INTEGER NOT NULL,
    PRIMARY KEY (customer_id, id)
);

CREATE TABLE IF NOT EXISTS accounts (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    id TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('current', 'savings')),
    balance_fen INTEGER NOT NULL,
    PRIMARY KEY (customer_id, id)
);

CREATE TABLE IF NOT EXISTS pots (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    id TEXT NOT NULL,
    name TEXT NOT NULL,
    balance_fen INTEGER NOT NULL,
    PRIMARY KEY (customer_id, id),
    UNIQUE (customer_id, name)
);

CREATE TABLE IF NOT EXISTS cards (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('physical', 'virtual')),
    last4 TEXT NOT NULL,
    status TEXT NOT NULL,
    online_limit_fen INTEGER NOT NULL,
    account_id TEXT NOT NULL,
    PRIMARY KEY (customer_id, id),
    FOREIGN KEY (customer_id, account_id) REFERENCES accounts(customer_id, id)
);

CREATE TABLE IF NOT EXISTS subscriptions (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    id TEXT NOT NULL,
    merchant TEXT NOT NULL,
    amount_fen INTEGER NOT NULL,
    day_of_month INTEGER NOT NULL,
    status TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, id)
);

CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    risk_level INTEGER NOT NULL,
    annual_rate_bp INTEGER NOT NULL,
    min_fen INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS holdings (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    product_id TEXT NOT NULL REFERENCES products(id),
    principal_fen INTEGER NOT NULL,
    value_fen INTEGER NOT NULL,
    PRIMARY KEY (customer_id, product_id)
);

CREATE TABLE IF NOT EXISTS transactions (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    amount_fen INTEGER NOT NULL,
    counterparty TEXT NOT NULL,
    category TEXT NOT NULL,
    memo TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    country TEXT NOT NULL,
    card_present INTEGER NOT NULL,
    PRIMARY KEY (customer_id, id),
    FOREIGN KEY (customer_id, account_id) REFERENCES accounts(customer_id, id)
);

CREATE TABLE IF NOT EXISTS transfers (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    id TEXT NOT NULL,
    to_contact_id TEXT NOT NULL,
    amount_fen INTEGER NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    release_at TEXT NOT NULL,
    account_id TEXT NOT NULL,
    PRIMARY KEY (customer_id, id),
    FOREIGN KEY (customer_id, to_contact_id) REFERENCES contacts(customer_id, id),
    FOREIGN KEY (customer_id, account_id) REFERENCES accounts(customer_id, id)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_transfers_customer_idempotency
    ON transfers(customer_id, idempotency_key);

CREATE TABLE IF NOT EXISTS scheduled_transfers (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    id TEXT NOT NULL,
    to_contact_id TEXT NOT NULL,
    amount_fen INTEGER NOT NULL,
    day_of_month INTEGER NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    PRIMARY KEY (customer_id, id),
    FOREIGN KEY (customer_id, to_contact_id) REFERENCES contacts(customer_id, id)
);

CREATE TABLE IF NOT EXISTS payment_requests (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    id TEXT NOT NULL,
    from_contact_id TEXT NOT NULL,
    amount_fen INTEGER NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    PRIMARY KEY (customer_id, id),
    FOREIGN KEY (customer_id, from_contact_id) REFERENCES contacts(customer_id, id)
);

CREATE TABLE IF NOT EXISTS merchant_orders (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    id TEXT NOT NULL,
    merchant TEXT NOT NULL,
    item TEXT NOT NULL,
    amount_fen INTEGER NOT NULL,
    deliver_on TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, id)
);
"""

_TABLES_IN_DELETE_ORDER = (
    "holdings",
    "transactions",
    "transfers",
    "scheduled_transfers",
    "payment_requests",
    "merchant_orders",
    "cards",
    "subscriptions",
    "contacts",
    "pots",
    "accounts",
    "products",
    "customers",
    "meta",
)


def connect(path: str = ":memory:") -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all bank tables and indexes."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()


def reset(conn: sqlite3.Connection) -> None:
    """Delete every row while keeping the schema."""
    with conn:
        for table in _TABLES_IN_DELETE_ORDER:
            conn.execute(f"DELETE FROM {table}")


def _dt(value: datetime) -> str:
    return value.isoformat()


def _date(value: date) -> str:
    return value.isoformat()


def load_state(conn: sqlite3.Connection, state: BankState) -> None:
    """Insert one customer and their data without touching other customers."""
    customer = state.customer
    customer_id = customer.id
    with conn:
        conn.execute(
            "INSERT INTO customers (id, persona_id, name, phone) VALUES (?, ?, ?, ?)",
            (customer.id, customer.persona_id, customer.name, customer.phone),
        )

        conn.executemany(
            """
            INSERT INTO contacts
                (customer_id, id, name, relation, note, account_no, phone,
                 bank_name, birthday, verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    customer_id,
                    contact.id,
                    contact.name,
                    contact.relation,
                    contact.note,
                    contact.account_no,
                    contact.phone,
                    contact.bank_name,
                    _date(contact.birthday) if contact.birthday else None,
                    int(contact.verified),
                )
                for contact in state.contacts
            ],
        )

        conn.executemany(
            """
            INSERT INTO accounts (customer_id, id, type, balance_fen)
            VALUES (?, ?, ?, ?)
            """,
            [
                (customer_id, account.id, account.type, account.balance_fen)
                for account in state.accounts
            ],
        )

        conn.executemany(
            """
            INSERT INTO pots (customer_id, id, name, balance_fen)
            VALUES (?, ?, ?, ?)
            """,
            [
                (customer_id, pot.id, pot.name, pot.balance_fen)
                for pot in state.pots
            ],
        )

        conn.executemany(
            """
            INSERT INTO cards
                (customer_id, id, kind, last4, status, online_limit_fen, account_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    customer_id,
                    card.id,
                    card.kind,
                    card.last4,
                    card.status,
                    card.online_limit_fen,
                    card.account_id,
                )
                for card in state.cards
            ],
        )

        conn.executemany(
            """
            INSERT INTO subscriptions
                (customer_id, id, merchant, amount_fen, day_of_month, status,
                 last_used_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    customer_id,
                    sub.id,
                    sub.merchant,
                    sub.amount_fen,
                    sub.day_of_month,
                    sub.status,
                    _date(sub.last_used_at),
                )
                for sub in state.subscriptions
            ],
        )

        # Products are global and identical for every persona.
        conn.executemany(
            """
            INSERT OR IGNORE INTO products
                (id, name, risk_level, annual_rate_bp, min_fen)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    product.id,
                    product.name,
                    product.risk_level,
                    product.annual_rate_bp,
                    product.min_fen,
                )
                for product in state.products
            ],
        )

        conn.executemany(
            """
            INSERT INTO holdings
                (customer_id, product_id, principal_fen, value_fen)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    customer_id,
                    holding.product_id,
                    holding.principal_fen,
                    holding.value_fen,
                )
                for holding in state.holdings
            ],
        )

        conn.executemany(
            """
            INSERT INTO transactions
                (customer_id, id, account_id, ts, amount_fen, counterparty,
                 category, memo, channel, country, card_present)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    customer_id,
                    txn.id,
                    txn.account_id,
                    _dt(txn.ts),
                    txn.amount_fen,
                    txn.counterparty,
                    txn.category,
                    txn.memo,
                    txn.channel,
                    txn.country,
                    int(txn.card_present),
                )
                for txn in state.transactions
            ],
        )
