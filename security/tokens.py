"""Semantic-token vault: minting, HMAC integrity and session binding.

The planner only ever sees token labels such as ``<PAYEE_1>``.  This module is
the only place where a real contact/account/amount/date is connected to such a
label, and the connection is signed so it cannot be forged or edited by the
model.
"""

from __future__ import annotations

import calendar
import hashlib
import hmac
import json
import re
import secrets
import unicodedata
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from bank import money as bank_money
from bank.categories import category_for
from bank.merchants import CATALOGUE
from bank.redemption import REDEEMABLE_PRODUCTS
from security.english_amounts import find_english_rmb

TOKEN_KINDS = frozenset(
    {
        "PAYEE",
        "ACCT",
        "POT",
        "CARD",
        "SUB",
        "PROD",
        "MERCHANT",
        "AMT",
        "DATE",
        "RAW",
    }
)

ORIGINS = ("untrusted", "user_text", "bank_context")

# Lower rank == less trusted.  When an entity is minted more than once in a
# session the least trusted origin wins, so untrusted text can never launder a
# token into a trusted one.
_ORIGIN_RANK = {"untrusted": 0, "user_text": 1, "bank_context": 2}

_TOKEN_LABEL_RE = re.compile(r"<[A-Z]+_[A-Za-z0-9]+>")
_NAME_PART_RE = re.compile(r"[\s,，/、·]+")
_ASCII_PHRASE_RE = re.compile(r"[A-Za-z]+(?:[ \-][A-Za-z]+)*")
_PAREN_RE = re.compile(r"[（(].*?[）)]")

_CONTEXT_RELATIONS = frozenset({"spouse", "parent", "child", "landlord"})

# ---------------------------------------------------------------------------
# Pinyin aliases for the seeded contacts.  Matching is case- and
# space-insensitive; the values are the canonical contact names in the seed.
# ---------------------------------------------------------------------------

PINYIN_ALIASES: dict[str, str] = {
    "lina": "李娜 Lina",
    "li na": "李娜 Lina",
    "zhang wei": "张伟",
    "wang jianguo": "王建国",
    "wang jian guo": "王建国",
    "xiao wang": "室友 小王",
    "xiao li": "室友 小李",
    "a jie": "室友 阿杰",
    "wang fang": "王芳",
    "lao jia": "老家 爸妈",
    "ba ma": "老家 爸妈",
    "zhou xiaobao": "周小宝",
    "zhou xiao bao": "周小宝",
    "chen jing": "女儿 陈静",
    "er zi": "儿子",
    "wang ayi": "社区王阿姨",
    "shequ wang ayi": "社区王阿姨",
    "lin tao": "林涛",
    "youeryuan": "幼儿园",
    "po po": "婆婆",
    "a yi": "阿姨 (nanny)",
    "lao chen": "老陈 蔬菜批发",
    "shucai pifa": "老陈 蔬菜批发",
    "xin da maoyi": "鑫达贸易",
    "xiao fang": "店员 小芳",
    "he li": "何丽",
    "mr wang": "王先生 Mr Wang",
    "wang xiansheng": "王先生 Mr Wang",
    "tom": "Tom",
    "li ming": "李明 Li Ming",
}

# Subscription brand aliases -> canonical subscription merchant name.
SUB_ALIASES: dict[str, str] = {
    "爱奇艺": "爱奇艺 VIP",
    "爱奇艺vip": "爱奇艺 VIP",
    "iqiyi": "爱奇艺 VIP",
    "网易云": "网易云音乐",
    "网易云音乐": "网易云音乐",
    "qq音乐": "QQ音乐",
    "keep": "Keep",
    "百度网盘": "百度网盘",
    "icloud": "iCloud",
    "b站": "B站大会员",
    "bilibili": "B站大会员",
    "腾讯视频": "腾讯视频",
    "王者荣耀": "王者荣耀 月卡",
    "夸克网盘": "夸克网盘",
    "老年手机": "老年手机 套餐",
    "广场舞": "广场舞 会员",
    "少儿英语": "少儿英语 App",
    "儿童绘本": "儿童绘本 会员",
    "家庭宽带": "家庭宽带",
    "netflix": "Netflix via VPN card",
    "spotify": "Spotify",
    "中国移动": "中国移动 套餐",
}

_KNOWN_SUB_BRANDS = frozenset(SUB_ALIASES.values())

# Merchant display names are generic; these aliases let user text reference them.
MERCHANT_SPECS: tuple[tuple[str, str], ...] = (
    ("花店", "florist"),
    ("蛋糕店", "cake shop"),
)

MERCHANT_ALIASES: dict[str, str] = {
    "花店": "花店",
    "鲜花": "花店",
    "flower": "花店",
    "flowers": "花店",
    "florist": "花店",
    "蛋糕店": "蛋糕店",
    "蛋糕": "蛋糕店",
    "cake": "蛋糕店",
    "cake shop": "蛋糕店",
    "bakery": "蛋糕店",
}


def _price_text(price_fen: int) -> str:
    if price_fen % 100 == 0:
        return f"¥{price_fen // 100}"
    return f"¥{price_fen // 100}.{price_fen % 100:02d}"


def _catalogue_items_text(real_name: str) -> str | None:
    items = CATALOGUE.get(real_name, {})
    if not items:
        return None
    catalogue = ", ".join(
        f"{code} {_price_text(item[2])}" for code, item in items.items()
    )
    return f"items: {catalogue}"


def _merchant_tags(real_name: str) -> OrderedDict[str, Any]:
    """Tags for a MERCHANT token, including its public catalogue prices."""

    label = dict(MERCHANT_SPECS).get(real_name, real_name)
    tags: OrderedDict[str, Any] = OrderedDict([("kind", label)])
    items = _catalogue_items_text(real_name)
    if items is not None:
        tags["items"] = items
    return tags


# Generic words that should not match a subscription/merchant on their own.
_GENERIC_BRAND_PARTS = frozenset(
    {"vip", "app", "via", "vpn", "card", "会员", "套餐", "月卡"}
)


# Relation words a user may say instead of a contact name.
RELATION_WORDS: dict[str, str] = {
    "妈妈": "parent",
    "妈": "parent",
    "我妈": "parent",
    "老妈": "parent",
    "mom": "parent",
    "mother": "parent",
    "mum": "parent",
    "mommy": "parent",
    "mama": "parent",
    "爸爸": "parent",
    "爸": "parent",
    "我爸": "parent",
    "老爸": "parent",
    "dad": "parent",
    "father": "parent",
    "papa": "parent",
    "老婆": "spouse",
    "wife": "spouse",
    "媳妇": "spouse",
    "爱人": "spouse",
    "老伴": "spouse",
    "老公": "spouse",
    "husband": "spouse",
    "hubby": "spouse",
    "儿子": "child",
    "son": "child",
    "女儿": "child",
    "daughter": "child",
    "闺女": "child",
    "丫头": "child",
    "孩子": "child",
    "房东": "landlord",
    "landlord": "landlord",
    "朋友": "friend",
    "friend": "friend",
    "闺蜜": "friend",
    "哥们": "friend",
    "同事": "colleague",
    "colleague": "colleague",
}

# Gendered relation words: a word from one set must not resolve to a contact
# whose own name/label carries a word from the other set (e.g. 爸爸 != 妈妈).
_FEMALE_WORDS = {
    "妈妈",
    "妈",
    "我妈",
    "老妈",
    "mom",
    "mother",
    "mum",
    "mommy",
    "mama",
    "老婆",
    "wife",
    "媳妇",
    "女儿",
    "daughter",
    "闺女",
    "丫头",
}
_MALE_WORDS = {
    "爸爸",
    "爸",
    "我爸",
    "老爸",
    "dad",
    "father",
    "papa",
    "老公",
    "husband",
    "hubby",
    "儿子",
    "son",
}


# ---------------------------------------------------------------------------
# Session and vault
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """One planner conversation, bound to a plan and a bank customer."""

    session_id: str = field(default_factory=lambda: secrets.token_hex(16))
    plan_id: str = field(default_factory=lambda: secrets.token_hex(8))
    customer_id: str = ""


@dataclass
class TokenEntry:
    """A minted label and the real value it stands for."""

    label: str
    kind: str
    real_ref: Any
    tags: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    origin: str = "user_text"
    session_id: str = ""
    plan_id: str = ""
    customer_id: str = ""
    sig: str = ""


class TokenError(Exception):
    """A token could not be resolved safely."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _canonical(
    session_id: str,
    plan_id: str,
    customer_id: str,
    label: str,
    kind: str,
    real_ref: Any,
    origin: str,
    tags: Any,
) -> bytes:
    if isinstance(real_ref, (date, datetime)):
        real_ref = real_ref.isoformat()
    return json.dumps(
        [session_id, plan_id, customer_id, label, kind, real_ref, origin, tags],
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


class TokenVault:
    """Server-side in-memory store of signed semantic tokens."""

    def __init__(self, secret_key: bytes) -> None:
        if not isinstance(secret_key, (bytes, bytearray)):
            raise TypeError("secret_key must be bytes")
        self._secret = bytes(secret_key)
        self._entries: dict[str, dict[str, TokenEntry]] = {}
        self._by_ref: dict[str, dict[tuple[str, Any], str]] = {}
        self._counters: dict[str, dict[str, int]] = {}
        self._closed_sessions: set[str] = set()
        self._closed_plans: set[tuple[str, str]] = set()
        self.unknown_labels: list[str] = []

    # -- public accessors ---------------------------------------------------

    @property
    def store(self) -> dict[str, dict[str, TokenEntry]]:
        """The raw session -> label -> entry map (tests / diagnostics)."""

        return self._entries

    def entries(self, session: Session) -> list[TokenEntry]:
        return list(self._entries.get(session.session_id, {}).values())

    # -- minting / resolving ------------------------------------------------

    def mint(
        self,
        session: Session,
        kind: str,
        real_ref: Any,
        tags: dict[str, Any] | None = None,
        origin: str = "user_text",
    ) -> TokenEntry:
        if kind not in TOKEN_KINDS:
            raise ValueError(f"unknown token kind: {kind!r}")
        if origin not in _ORIGIN_RANK:
            raise ValueError(f"unknown token origin: {origin!r}")

        sid = session.session_id
        session_entries = self._entries.setdefault(sid, {})
        by_ref = self._by_ref.setdefault(sid, {})
        key = (kind, real_ref)

        existing_label = by_ref.get(key)
        if existing_label is not None:
            entry = session_entries[existing_label]
            if not self._is_expired(entry):
                if _ORIGIN_RANK[origin] < _ORIGIN_RANK[entry.origin]:
                    entry.origin = origin
                if tags:
                    entry.tags.update(tags)
                entry.sig = self._sign(entry)
                return entry
            # The plan that owns this entry was closed: mint a fresh token for
            # the session's current plan instead of reusing the stale label.

        counters = self._counters.setdefault(sid, {})
        index = counters.get(kind, 0)
        counters[kind] = index + 1

        label = self._make_label(kind, index)
        entry = TokenEntry(
            label=label,
            kind=kind,
            real_ref=real_ref,
            tags=OrderedDict(tags or {}),
            origin=origin,
            session_id=sid,
            plan_id=session.plan_id,
            customer_id=session.customer_id,
        )
        entry.sig = self._sign(entry)
        session_entries[label] = entry
        by_ref[key] = label
        return entry

    def resolve(self, label: str, session: Session) -> TokenEntry:
        entry = self._entries.get(session.session_id, {}).get(label)
        if entry is None:
            raise TokenError("unknown")
        if (
            entry.session_id != session.session_id
            or entry.plan_id != session.plan_id
            or entry.customer_id != session.customer_id
        ):
            raise TokenError("unknown")
        if self._is_expired(entry):
            raise TokenError("expired")
        if not hmac.compare_digest(entry.sig, self._sign(entry)):
            raise TokenError("tampered")
        return entry

    def close_plan(self, session: Session) -> None:
        plan_ids = {entry.plan_id for entry in self.entries(session)}
        plan_ids.add(session.plan_id)
        for plan_id in plan_ids:
            self._closed_plans.add((session.session_id, plan_id))

    def close_session(self, session: Session) -> None:
        self._closed_sessions.add(session.session_id)

    def known_tokens(self, session: Session) -> dict[str, str]:
        known: dict[str, str] = {}
        for label, entry in self._entries.get(session.session_id, {}).items():
            if self._is_expired(entry):
                continue
            if not hmac.compare_digest(entry.sig, self._sign(entry)):
                continue
            known[label] = entry.kind
        return known

    # -- internals ----------------------------------------------------------

    def _sign(self, entry: TokenEntry) -> str:
        payload = _canonical(
            entry.session_id,
            entry.plan_id,
            entry.customer_id,
            entry.label,
            entry.kind,
            entry.real_ref,
            entry.origin,
            entry.tags,
        )
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def _is_expired(self, entry: TokenEntry) -> bool:
        if entry.session_id in self._closed_sessions:
            return True
        return (entry.session_id, entry.plan_id) in self._closed_plans

    @staticmethod
    def _make_label(kind: str, index: int) -> str:
        if kind == "ACCT":
            value = index + 1
            letters = ""
            while value:
                value, remainder = divmod(value - 1, 26)
                letters = chr(ord("A") + remainder) + letters
            return f"<ACCT_{letters}>"
        return f"<{kind}_{index + 1}>"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(entry: TokenEntry) -> str:
    """Render a token exactly as the planner sees it (no real values)."""

    parts = [entry.label.strip("<>")]
    for key, value in entry.tags.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value)
            continue
        text = str(value)
        if key == "last":
            text = f"last: {text}"
        elif key in {"risk", "rate"}:
            text = f"{key} {text}"
        elif isinstance(value, bool):
            text = f"{key}={'true' if value else 'false'}"
        parts.append(text)

    if entry.kind == "MERCHANT" and "items" not in entry.tags:
        items = _catalogue_items_text(str(entry.real_ref))
        if items is not None:
            parts.append(items)

    return "<" + " | ".join(parts) + ">"


# ---------------------------------------------------------------------------
# Buckets / relative dates
# ---------------------------------------------------------------------------


def amount_bucket(amount_fen: int) -> str:
    """Amount bucket in yuan, with 1,000 always an edge."""

    fen = abs(int(amount_fen))
    if fen < 10_000:
        return "<100"
    if fen < 50_000:
        return "100-500"
    if fen < 100_000:
        return "500-1000"
    if fen < 500_000:
        return "1000-5000"
    return "5000+"


def rate_bucket(annual_rate_bp: int) -> str:
    bp = int(annual_rate_bp)
    if bp < 200:
        return "<2%"
    if bp < 300:
        return "2-3%"
    if bp < 500:
        return "3-5%"
    return "5%+"


def relative_date_text(target: date, now: date) -> str:
    delta = (target - now).days
    if delta == 0:
        return "today"
    if delta > 0:
        return f"in {delta} days"
    return f"{-delta} days ago"


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError("now must be a date or datetime")


# ---------------------------------------------------------------------------
# Bank helpers
# ---------------------------------------------------------------------------


def _all_contacts(bank: Any) -> list[Any]:
    find = getattr(bank, "find_contact", None)
    if find is None:
        return []
    return list(find(""))


def _all_products(bank: Any) -> list[Any]:
    lister = getattr(bank, "list_products", None)
    if lister is None:
        return []
    return list(lister())


def _held_product_ids(bank: Any) -> set[str]:
    portfolio = getattr(bank, "get_portfolio", None)
    if portfolio is None:
        return set()
    return {
        holding.product_id
        for holding in portfolio()
        if int(holding.value_fen) > 0
    }


def _account_rows(bank: Any) -> list[tuple[str, str]]:
    conn = getattr(bank, "conn", None)
    if conn is None:
        return [(f"acc_current_{bank.customer_id}", "current")]
    rows = conn.execute(
        "SELECT id, type FROM accounts WHERE customer_id = ? ORDER BY id",
        (bank.customer_id,),
    ).fetchall()
    return [(row["id"], row["type"]) for row in rows]


def _trust_for(contact: Any) -> str:
    note = (getattr(contact, "note", "") or "").upper()
    if "FLAG" in note or "FRAUD" in note:
        return "flagged"
    if note.startswith("NEW") or not getattr(contact, "verified", True):
        return "new_payee"
    return "verified"


def _contact_transactions(bank: Any, contact: Any, now: date) -> list[Any]:
    list_transactions = getattr(bank, "list_transactions", None)
    if list_transactions is None:
        return []
    return list_transactions(counterparty=contact.name, limit=None)


def _frequency_for(bank: Any, contact: Any, now: date) -> str:
    outgoing = [
        txn
        for txn in _contact_transactions(bank, contact, now)
        if txn.amount_fen < 0
    ]
    if not outgoing:
        return "new"

    now_dt = datetime.combine(now, time.max)
    cutoff_90 = now_dt - timedelta(days=90)
    cutoff_30 = now_dt - timedelta(days=30)
    count_90 = sum(1 for txn in outgoing if txn.ts >= cutoff_90)
    count_30 = sum(1 for txn in outgoing if txn.ts >= cutoff_30)

    if count_30 >= 4:
        return "weekly"
    if count_90 >= 3:
        return "monthly"
    return "rare"


def _payee_tags(bank: Any, contact: Any, now: date) -> OrderedDict[str, Any]:
    tags: OrderedDict[str, Any] = OrderedDict()
    tags["relation"] = getattr(contact, "relation", "other")
    tags["frequency"] = _frequency_for(bank, contact, now)

    outgoing = [
        txn
        for txn in _contact_transactions(bank, contact, now)
        if txn.amount_fen < 0
    ]
    if outgoing:
        last = max(outgoing, key=lambda txn: txn.ts)
        tags["last"] = amount_bucket(last.amount_fen)

    tags["trust"] = _trust_for(contact)
    return tags


def _subscription_tags(subscription: Any, now: date) -> OrderedDict[str, Any]:
    tags: OrderedDict[str, Any] = OrderedDict()
    # The merchant string comes from a third party; only a known brand name
    # reaches the planner, otherwise just the category.
    if subscription.merchant in _KNOWN_SUB_BRANDS:
        tags["brand"] = subscription.merchant
    tags["category"] = category_for(subscription.merchant)
    tags["amount"] = f"{amount_bucket(subscription.amount_fen)}/mo"
    if (now - subscription.last_used_at).days >= 60:
        tags["unused"] = "unused 60d+"
    return tags


def _product_tags(
    product: Any, held_product_ids: set[str] | None = None
) -> OrderedDict[str, Any]:
    tags: OrderedDict[str, Any] = OrderedDict()
    tags["risk"] = product.risk_level
    tags["rate"] = rate_bucket(product.annual_rate_bp)
    # Only safe booleans: the bank decides what is held, never the model.
    held = held_product_ids is not None and product.id in held_product_ids
    tags["held"] = held
    tags["redeemable"] = held and product.id in REDEEMABLE_PRODUCTS
    return tags


# ---------------------------------------------------------------------------
# User text tokenization
# ---------------------------------------------------------------------------

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10_000, "亿": 100_000_000}
_CN_NUMERAL_RE = re.compile(
    r"[零〇一二三四五六七八九十百千万亿两壹贰叁肆伍陆柒捌玖拾佰仟萬]+"
)
# Banker's (大写) numerals only need folding while parsing amounts; RAW
# phone/account detection stays on the plain digit forms.
_CN_BANKER_FOLD = str.maketrans(
    "零壹贰叁肆伍陆柒捌玖拾佰仟萬",
    "零一二三四五六七八九十百千万",
)
# A comma groups digits only as "12,000"; elsewhere it separates numbers, so
# "转1500，5号" keeps ¥1,500 and "1000，1000" is two amounts (eval F47).
_ARABIC_NUMBER_RE = re.compile(
    r"(?<![\d.])((?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)(?![\d])"
)
_RAW_PHONE_RE = re.compile(r"(?<![0-9])1[3-9](?:[ \-.]?[0-9]){9}(?![0-9])")
_RAW_LONG_RE = re.compile(r"(?<![0-9])[0-9](?:[ \-.]?[0-9]){11,18}(?![0-9])")
_RAW_CN_RE = re.compile(
    r"[零〇一二三四五六七八九幺两]{7,}(?![零〇一二三四五六七八九幺两元块万千百])"
)
_RAW_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]+"
)
# Chinese resident ID numbers: 18 characters with a trailing X.
_RAW_ID_RE = re.compile(
    r"(?<![0-9A-Za-z])[0-9]{17}[0-9Xx](?![0-9A-Za-z])"
)
_SENSITIVE_VALUE_RE = re.compile(
    r"(?i)(?:password|passcode|pin|otp|密码|验证码)\s*(?:is\s+|[:：=]\s*|是\s*)([^\s,，。;；]+)"
    r"|(?:my name is|i am called)\s+([^,.;\n]{2,60})(?=[,.;\n]|$)"
    r"|(?:我叫|姓名是)([\u4e00-\u9fff]{2,5})(?=[,，。;；\s]|$)"
    r"|(?:i live at|my address is|住在|地址是)\s*([^,，。;；\n]{3,100})"
)
_DIGIT_UNIT_RE = re.compile(
    r"(?<![\d.])\d+(?:\.\d+)?\s*[万千百](?:\s*\d+(?:\s*[万千百十])?)*(?![\d.])"
)
_DIGIT_UNIT_PART_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([万千百十]?)")
_UNIT_VALUES = {"万": 10_000, "千": 1_000, "百": 100, "十": 10}


def _digit_unit_value(run: str) -> Decimal | None:
    """Yuan value of "3千5百", "1万2千3百", "1万5" (a trailing bare digit
    takes the next smaller unit), or None when the run is not well formed."""

    total = Decimal(0)
    section = Decimal(0)
    previous = None
    for number, unit in _DIGIT_UNIT_PART_RE.findall(run):
        value = Decimal(number)
        if unit:
            size = _UNIT_VALUES[unit]
        elif previous is not None and previous > 10:
            size = previous // 10
        else:
            return None
        if previous is not None and size >= previous and unit != "万":
            return None
        if unit == "万":
            total += (section + value) * 10_000
            section = Decimal(0)
        else:
            section += value * size
        previous = size
    result = total + section
    return result if result > 0 else None


_KB_SHORT_RE = re.compile(r"(?<![\d.])([0-9]+)\s*[kK]\s*([0-9])(?![0-9])")
_WAN_SHORT_RE = re.compile(r"(?<![\d.])([0-9]+)\s*万\s*([0-9])(?![0-9])")
_DOT_THREE_RE = re.compile(r"[0-9]+\.[0-9]{3}")
_FOREIGN_CURRENCY_PREFIXES = (
    "usd",
    "us$",
    "hkd",
    "eur",
    "$",
    "美元",
    "港币",
    "欧元",
)
_FOREIGN_CURRENCY_SUFFIXES = ("usd", "hkd", "eur", "美元", "港币", "欧元")
_MAX_AMOUNT_FEN = 100_000_000_000

# Units/classifiers that turn a number into something that is not money.
_NON_MONEY_NEXT = set(
    "天次个月年号日点分秒周岁人位名件个只张杯份碗瓶包袋斤公里米步楼层家间"
    "套台车辆本页章门课队双对组批起遍趟场桌盘盒箱条件项期"
)
_MONEY_SUFFIX_RE = re.compile(r"(元|块|快|rmb|RMB|[kKwW](?![A-Za-z])|万|¥|￥)")
_CURRENCY_PREFIXES = ("¥", "￥")

# English count/duration words: a bare integer followed by one of these is a
# count, not money ("1200 over 12 periods", "12 installments of 200").  Only
# this closed list is honoured, so an arbitrary English word after a number
# (or an explicit currency suffix) still leaves the amount alone.
_ENGLISH_COUNT_WORDS = (
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "hour",
    "hours",
    "minute",
    "minutes",
    "second",
    "seconds",
    "night",
    "nights",
    "time",
    "times",
    "period",
    "periods",
    "instalment",
    "instalments",
    "installment",
    "installments",
)

# A frequency adjective is only a count marker when it directly precedes a
# period/installment noun ("12 monthly installments").  On its own it must not
# suppress money: "send 300 monthly" is still ¥300.
_FREQUENCY_ADJECTIVES = ("monthly", "weekly", "daily", "yearly", "annual")
_PERIOD_NOUNS = (
    "period",
    "periods",
    "instalment",
    "instalments",
    "installment",
    "installments",
)
_ENGLISH_COUNT_RE = re.compile(
    r"(?i)(?:"
    + "|".join(
        [
            rf"{adjective}\s+{noun}"
            for adjective in _FREQUENCY_ADJECTIVES
            for noun in _PERIOD_NOUNS
        ]
        + sorted(_ENGLISH_COUNT_WORDS, key=len, reverse=True)
    )
    + r")\b"
)


def _parse_chinese_number(text: str) -> int:
    total = 0
    section = 0
    number = 0

    for char in text:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char == "亿":
            section += number
            total = (total + section) * 100_000_000
            section = 0
            number = 0
        elif char == "万":
            section += number
            total += section * 10_000
            section = 0
            number = 0
        elif char in _CN_UNITS:
            if number == 0:
                number = 1
            section += number * _CN_UNITS[char]
            number = 0

    section += number
    value = total + section

    # Colloquial shorthand: 一千五 / 两百五 / 一万五.
    if (
        len(text) >= 2
        and text[-1] in _CN_DIGITS
        and text[-2] in {"十", "百", "千", "万", "亿"}
        and "零" not in text
        and "〇" not in text
    ):
        digit = _CN_DIGITS[text[-1]]
        unit = _CN_UNITS[text[-2]]
        value = value - digit + digit * (unit // 10)

    return value


def _to_fen(yuan_value: Decimal) -> int:
    return int((yuan_value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _normalise(text: str) -> str:
    """Fold compatibility forms and drop zero-width/format characters."""

    text = unicodedata.normalize("NFKC", text)
    return "".join(char for char in text if unicodedata.category(char) != "Cf")


def _has_foreign_currency(text: str, start: int, end: int) -> bool:
    prefix = text[:start].rstrip().lower()
    if prefix.endswith(_FOREIGN_CURRENCY_PREFIXES):
        return True
    suffix = text[end:].lstrip().lower()
    return suffix.startswith(_FOREIGN_CURRENCY_SUFFIXES)


def _surface_matches(text: str, surface: str) -> Iterable[re.Match[str]]:
    chars = [char for char in surface if not char.isspace()]
    if not chars:
        return ()

    # Space-insensitive for every surface: ``zhang wei`` == ``zhangwei`` and
    # ``爱奇艺 VIP`` == ``爱奇艺VIP``.
    body = r"\s*".join(re.escape(char) for char in chars)
    prefix = (
        r"(?<![A-Za-z])"
        if chars[0].isascii() and chars[0].isalpha()
        else ""
    )
    suffix = (
        r"(?![A-Za-z])"
        if chars[-1].isascii() and chars[-1].isalpha()
        else ""
    )
    pattern = re.compile(rf"{prefix}{body}{suffix}", re.IGNORECASE)
    return pattern.finditer(text)


def _contact_surface_forms(contact: Any) -> set[str]:
    name = (getattr(contact, "name", "") or "").strip()
    if not name:
        return set()

    forms: set[str] = {name}
    base = _PAREN_RE.sub(" ", name).strip()
    if base:
        forms.add(base)

    # "社区王阿姨" should also match "王阿姨": strip a leading descriptor from
    # long display names, but never create a form shorter than two chars.
    if sum(1 for char in base if "\u4e00" <= char <= "\u9fff") >= 4:
        for prefix in ("社区", "公司", "老家", "楼下", "隔壁", "单位", "小区"):
            if base.startswith(prefix):
                remainder = base[len(prefix) :].strip()
                if len(remainder) >= 2:
                    forms.add(remainder)
                break

    parts = [part for part in _NAME_PART_RE.split(base) if part]
    if len(parts) > 1:
        generic = set(RELATION_WORDS) | {"室友", "店员", "老家", "社区"}
        parts = [part for part in parts if part not in generic]
    for part in parts:
        if len(part) >= 2:
            forms.add(part)

    for phrase in _ASCII_PHRASE_RE.findall(base):
        if len(phrase) >= 2:
            forms.add(phrase)

    # Parenthesised aliases are part of how the user names the contact,
    # e.g. "阿姨 (nanny)" must also match "nanny".
    for inner in re.findall(r"[（(](.*?)[）)]", name):
        inner = inner.strip()
        if len(inner) >= 2:
            forms.add(inner)

    return forms


def _select_matches(
    candidates: list[tuple[int, int, Any]],
) -> list[tuple[int, int, list[Any]]]:
    """Choose the longest non-overlapping matches, grouping identical spans."""

    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0]))

    chosen: list[tuple[int, int, Any]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, entity in candidates:
        conflict = False
        for left, right in occupied:
            if end <= left or start >= right:
                continue
            if start == left and end == right:
                # Same surface form for several entities (e.g. two 张伟).
                continue
            conflict = True
            break
        if conflict:
            continue
        chosen.append((start, end, entity))
        if (start, end) not in occupied:
            occupied.append((start, end))

    grouped: OrderedDict[tuple[int, int], list[Any]] = OrderedDict()
    for start, end, entity in chosen:
        bucket = grouped.setdefault((start, end), [])
        if entity not in bucket:
            bucket.append(entity)

    return [
        (start, end, group)
        for (start, end), group in sorted(grouped.items(), key=lambda item: item[0][0])
    ]


def _contact_label_text(contact: Any) -> str:
    """Everything the bank uses to display a contact, lowercased."""

    values = [getattr(contact, "name", "")]
    for attribute in ("label", "display_name", "display_label"):
        values.append(getattr(contact, attribute, ""))
    return " ".join(str(value) for value in values if value).lower()


def _mentions_any(text: str, words: set[str]) -> bool:
    lowered = text.lower()
    for word in words:
        if word.isascii():
            # Whole words only: "son" must not match "Jason" or "Alison".
            if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered):
                return True
        elif word in lowered:
            return True
    return False


def _find_name_matches(
    text: str, contacts: list[Any]
) -> list[tuple[int, int, list[Any]]]:
    """Return non-overlapping (start, end, contacts) matches, longest first."""

    candidates: list[tuple[int, int, Any]] = []

    for contact in contacts:
        for form in _contact_surface_forms(contact):
            for match in _surface_matches(text, form):
                candidates.append((match.start(), match.end(), contact))

    for alias, canonical_name in PINYIN_ALIASES.items():
        matching = [
            contact for contact in contacts if contact.name == canonical_name
        ]
        if not matching:
            continue
        for match in _surface_matches(text, alias):
            for contact in matching:
                candidates.append((match.start(), match.end(), contact))

    # Explicit spoken-name variant for the demo contact. An exact existing
    # contact wins; never use general fuzzy matching to choose a recipient.
    spoken_targets = [c for c in contacts if c.name == "李娜 Lina"]
    if len(spoken_targets) == 1:
        for match in _surface_matches(text, "Lena"):
            if not any(left < match.end() and right > match.start()
                       for left, right, _ in candidates):
                candidates.append((match.start(), match.end(), spoken_targets[0]))

    for word, relation in RELATION_WORDS.items():
        matching = [
            contact for contact in contacts if getattr(contact, "relation", "") == relation
        ]
        if not matching:
            continue
        lowered_word = word.lower()
        if lowered_word in _FEMALE_WORDS:
            same, excluded = _FEMALE_WORDS, _MALE_WORDS
        elif lowered_word in _MALE_WORDS:
            same, excluded = _MALE_WORDS, _FEMALE_WORDS
        else:
            same = excluded = None
        if excluded is not None:
            # Skip a contact only when its label names the other gender alone;
            # "老家 爸妈" (both parents) still matches 爸爸 and 妈妈.
            matching = [
                contact
                for contact in matching
                if not _mentions_any(_contact_label_text(contact), excluded)
                or _mentions_any(_contact_label_text(contact), same)
            ]
            if not matching:
                # Same as a relation word that matches no contact at all.
                continue
        # "女儿" with a daughter AND a son (both relation "child"): prefer the
        # contact whose own name carries the word, e.g. "女儿 陈静".
        named = [contact for contact in matching if word in str(getattr(contact, "name", ""))]
        if named:
            matching = named
        for match in _surface_matches(text, word):
            for contact in matching:
                candidates.append((match.start(), match.end(), contact))

    return _merge_adjacent_matches(text, _select_matches(candidates))


_JOIN_RE = re.compile(r"[\s的]{0,2}")


def _merge_adjacent_matches(
    text: str, matches: list[tuple[int, int, list[Any]]]
) -> list[tuple[int, int, list[Any]]]:
    """"my friend Tom" / "同事张伟": a relation word right before a name narrows
    it down (friend ∩ Tom = Tom, colleague ∩ 张伟 = the colleague) instead of
    minting two ambiguous payees (eval F55, F05)."""

    ordered = sorted(matches, key=lambda item: item[0])
    merged: list[tuple[int, int, list[Any]]] = []
    for start, end, contacts in ordered:
        if merged:
            prev_start, prev_end, prev_contacts = merged[-1]
            gap = text[prev_end:start]
            common = [
                contact
                for contact in contacts
                if any(contact is other or contact == other for other in prev_contacts)
            ]
            if _JOIN_RE.fullmatch(gap) and common:
                merged[-1] = (prev_start, end, common)
                continue
        merged.append((start, end, contacts))
    return merged


def _all_subscriptions(bank: Any) -> list[Any]:
    lister = getattr(bank, "list_subscriptions", None)
    if lister is None:
        return []
    return list(lister())


def _brand_surface_forms(name: str) -> set[str]:
    name = (name or "").strip()
    if not name:
        return set()

    forms: set[str] = {name}
    base = _PAREN_RE.sub(" ", name).strip()
    if base:
        forms.add(base)

    for part in _NAME_PART_RE.split(base):
        if len(part) >= 2 and part.lower() not in _GENERIC_BRAND_PARTS:
            forms.add(part)

    for phrase in _ASCII_PHRASE_RE.findall(base):
        if len(phrase) >= 2 and phrase.lower() not in _GENERIC_BRAND_PARTS:
            forms.add(phrase)

    return forms


def _find_subscription_matches(
    text: str, subscriptions: list[Any]
) -> list[tuple[int, int, list[Any]]]:
    candidates: list[tuple[int, int, Any]] = []

    for subscription in subscriptions:
        for form in _brand_surface_forms(subscription.merchant):
            for match in _surface_matches(text, form):
                candidates.append((match.start(), match.end(), subscription))

    for alias, canonical_name in SUB_ALIASES.items():
        matching = [
            subscription
            for subscription in subscriptions
            if subscription.merchant == canonical_name
        ]
        if not matching:
            continue
        for match in _surface_matches(text, alias):
            for subscription in matching:
                candidates.append((match.start(), match.end(), subscription))

    return _select_matches(candidates)


def _find_merchant_matches(
    text: str,
) -> list[tuple[int, int, list[str]]]:
    candidates: list[tuple[int, int, Any]] = []

    for real_name, _label in MERCHANT_SPECS:
        for form in _brand_surface_forms(real_name):
            for match in _surface_matches(text, form):
                candidates.append((match.start(), match.end(), real_name))

    for alias, canonical_name in MERCHANT_ALIASES.items():
        if not any(real_name == canonical_name for real_name, _ in MERCHANT_SPECS):
            continue
        for match in _surface_matches(text, alias):
            candidates.append((match.start(), match.end(), canonical_name))

    return _select_matches(candidates)


def _find_product_matches(
    text: str, products: list[Any]
) -> list[tuple[int, int, list[Any]]]:
    """Official product names, plus an unambiguous duration-stripped stem."""

    full_names = {product.name for product in products}
    stems: dict[str, list[Any]] = {}
    for product in products:
        stem = re.sub(r"\s*\d+\s*天\s*$", "", product.name).strip()
        if stem != product.name and len(stem) >= 4:
            stems.setdefault(stem, []).append(product)

    surfaces: dict[str, list[Any]] = {}
    for product in products:
        surfaces.setdefault(product.name, []).append(product)
    for stem, holders in stems.items():
        # A stem is only usable when it names exactly one product and is not
        # already another product's official name.
        if len(holders) == 1 and stem not in surfaces and stem not in full_names:
            surfaces[stem] = holders

    candidates: list[tuple[int, int, Any]] = []
    for surface, holders in surfaces.items():
        if len(holders) != 1:
            continue
        for match in _surface_matches(text, surface):
            candidates.append((match.start(), match.end(), holders[0]))
    return _select_matches(candidates)


def _find_dates(text: str, now: date) -> list[tuple[int, int, date]]:
    found: list[tuple[int, int, date]] = []
    claimed: list[tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return all(end <= left or start >= right for left, right in claimed)

    def add(start: int, end: int, target: date) -> None:
        if free(start, end):
            found.append((start, end, target))
            claimed.append((start, end))

    # Relative words first: they are unambiguous.
    relative_words = [
        (r"大后天", 3),
        (r"后天", 2),
        (r"明天", 1),
        (r"今天", 0),
        (r"昨天", -1),
        (r"tomorrow", 1),
        (r"today", 0),
        (r"yesterday", -1),
    ]
    for word, offset in relative_words:
        pattern = re.compile(
            rf"(?<![A-Za-z]){word}(?![A-Za-z])" if word.isascii() else word,
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            add(match.start(), match.end(), now + timedelta(days=offset))

    weekdays = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
    for match in re.finditer(r"下(?:个)?(?:周|星期)([一二三四五六日天])", text):
        weekday = weekdays[match.group(1)]
        next_monday = now - timedelta(days=now.weekday()) + timedelta(days=7)
        add(match.start(), match.end(), next_monday + timedelta(days=weekday))

    english_weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for name, weekday in english_weekdays.items():
        pattern = re.compile(rf"\bnext\s+{name}\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            next_monday = now - timedelta(days=now.weekday()) + timedelta(days=7)
            add(match.start(), match.end(), next_monday + timedelta(days=weekday))

    # Explicit Chinese month/day.
    for match in re.finditer(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?", text):
        month, day = int(match.group(1)), int(match.group(2))
        try:
            target = date(now.year, month, day)
        except ValueError:
            continue
        if target < now:
            try:
                target = date(now.year + 1, month, day)
            except ValueError:
                continue
        add(match.start(), match.end(), target)

    # Day of month: this month, or next month once the day has passed.
    for match in re.finditer(r"(?<!\d)(\d{1,2})\s*[号日]", text):
        day = int(match.group(1))
        if not 1 <= day <= 31:
            continue
        year, month = now.year, now.month
        if day < now.day:
            month += 1
            if month > 12:
                month = 1
                year += 1
        last_day = calendar.monthrange(year, month)[1]
        target = date(year, month, min(day, last_day))
        add(match.start(), match.end(), target)

    # English ordinal: "the 29th".
    for match in re.finditer(
        r"(?i)\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b", text
    ):
        day = int(match.group(1))
        if not 1 <= day <= 31:
            continue
        year, month = now.year, now.month
        if day < now.day:
            month += 1
            if month > 12:
                month = 1
                year += 1
        last_day = calendar.monthrange(year, month)[1]
        target = date(year, month, min(day, last_day))
        add(match.start(), match.end(), target)

    found.sort(key=lambda item: item[0])
    return found


def _find_amounts(
    text: str, claimed: list[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    """Return (start, end, fen) for money-like expressions in ``text``."""

    found: list[tuple[int, int, int]] = []

    def free(start: int, end: int) -> bool:
        return all(end <= left or start >= right for left, right in claimed)

    def negative(start: int) -> bool:
        index = start - 1
        while index >= 0 and text[index] in " \t":
            index -= 1
        return index >= 0 and text[index] in ("-", "−", "－", "负")

    def add(start: int, end: int, fen: int) -> None:
        if fen <= 0 or fen > _MAX_AMOUNT_FEN:
            return
        if negative(start):
            return
        if not free(start, end):
            return
        if _has_foreign_currency(text, start, end):
            return
        found.append((start, end, fen))
        claimed.append((start, end))

    for start, end, fen in find_english_rmb(text):
        add(start, end, fen)

    # Arabic digits with Chinese units: 5千, 3千5百, 1万2千3百, 2万3千, 1.5万.
    # Without this pass "5千" became ¥5 plus a lone 千 = ¥1,000.
    for match in _DIGIT_UNIT_RE.finditer(text):
        yuan = _digit_unit_value(match.group(0))
        end = match.end()
        if yuan is not None and end < len(text) and text[end] == "万":
            yuan *= 10_000  # "3百万", "5千万"
            end += 1
        if yuan is not None:
            add(match.start(), end, int(yuan * 100))

    # Shorthand multipliers with an Arabic base: 3k5 == 3500, 1万5 == 15000.
    for match in _KB_SHORT_RE.finditer(text):
        whole, fraction = int(match.group(1)), int(match.group(2))
        add(
            match.start(),
            match.end(),
            (whole * 1000 + fraction * 100) * 100,
        )
    for match in _WAN_SHORT_RE.finditer(text):
        whole, fraction = int(match.group(1)), int(match.group(2))
        add(
            match.start(),
            match.end(),
            (whole * 10_000 + fraction * 1000) * 100,
        )

    # Arabic numerals.
    for match in _ARABIC_NUMBER_RE.finditer(text):
        start, end = match.start(), match.end()
        if not free(start, end):
            continue

        raw_number = match.group(1)
        if _DOT_THREE_RE.fullmatch(raw_number) is not None:
            continue
        has_decimal = "." in raw_number
        value = Decimal(raw_number.replace(",", ""))

        # A currency symbol immediately before the number.
        prefix_start = start
        while prefix_start > 0 and text[prefix_start - 1] in " \t":
            prefix_start -= 1
        has_currency = False
        if prefix_start > 0 and text[prefix_start - 1] in _CURRENCY_PREFIXES:
            has_currency = True
            start = prefix_start - 1
        elif text[max(0, prefix_start - 3) : prefix_start].lower() == "rmb":
            has_currency = True
            start = max(0, prefix_start - 3)

        # A unit/multiplier immediately after the number.
        suffix_end = end
        while suffix_end < len(text) and text[suffix_end] in " \t":
            suffix_end += 1
        multiplier = Decimal(1)
        has_unit = has_currency
        if suffix_end < len(text):
            suffix_match = _MONEY_SUFFIX_RE.match(text, suffix_end)
            if suffix_match is not None:
                suffix = suffix_match.group(1)
                has_unit = True
                end = suffix_match.end()
                if suffix in {"k", "K"}:
                    multiplier = Decimal(1000)
                elif suffix in {"w", "W", "万"}:
                    multiplier = Decimal(10_000)
                elif suffix == "¥" or (suffix == "￥"):
                    pass

        if not free(start, end):
            continue

        if has_unit or has_decimal:
            add(start, end, _to_fen(value * multiplier))
            continue

        # A bare integer: reject dates, ordinals, counts and other classifiers.
        plain = value.to_integral_value()
        if plain != value:
            continue
        if plain <= 0:
            continue
        if start > 0 and text[start - 1] == "第":
            continue
        after = text[end:].lstrip()
        if after:
            if after[0] in _NON_MONEY_NEXT:
                continue
            if after[0] in "%‰":
                continue
            if re.match(r"(?i)(st|nd|rd|th)\b", after):
                continue
            if _ENGLISH_COUNT_RE.match(after):
                continue
        add(start, end, _to_fen(value))

    # Chinese numerals such as 三百、两千、一千五、1万.
    for match in _CN_NUMERAL_RE.finditer(text):
        start, end = match.start(), match.end()
        if not free(start, end):
            continue

        numeral = match.group(0).translate(_CN_BANKER_FOLD)
        if numeral in {"千万", "万分", "十分", "万一", "千千万万"}:
            continue
        if all(char in "万千百十" for char in numeral) and (
            start > 0 and text[start - 1].isdigit()
        ):
            # A unit left over after an Arabic digit ("-5千", "3百万") is not an
            # amount of its own: minting it made a lone 千 into ¥1,000.
            continue
        has_magnitude = any(char in _CN_UNITS for char in numeral)

        value = _parse_chinese_number(numeral)
        if value <= 0:
            continue

        suffix_end = end
        while suffix_end < len(text) and text[suffix_end] in " \t":
            suffix_end += 1
        suffix_match = (
            _MONEY_SUFFIX_RE.match(text, suffix_end)
            if suffix_end < len(text)
            else None
        )
        if suffix_match is not None:
            suffix = suffix_match.group(1)
            end = suffix_match.end()
            if suffix in {"k", "K"}:
                value *= 1000
            elif suffix in {"w", "W", "万"}:
                value *= 10_000

        # A bare Chinese numeral without a magnitude/unit ("两", "一") is too
        # ambiguous to be money on its own.
        if not has_magnitude and suffix_match is None:
            continue
        if not free(start, end):
            continue

        after = text[end:].lstrip()
        if after and after[0] in _NON_MONEY_NEXT:
            continue
        if after and after[0] in "%‰":
            continue

        add(start, end, value * 100)

    found.sort(key=lambda item: item[0])
    return found


def _apply_replacements(
    text: str, replacements: list[tuple[int, int, str]]
) -> str:
    for start, end, replacement in sorted(
        replacements, key=lambda item: item[0], reverse=True
    ):
        text = text[:start] + replacement + text[end:]
    return text


def _tokenize(
    text: str,
    bank: Any,
    vault: TokenVault,
    session: Session,
    now: date | datetime | str,
    origin: str,
) -> tuple[str, list[TokenEntry]]:
    if not isinstance(text, str):
        text = str(text)
    text = _normalise(text)
    now_date = _as_date(now)
    contacts = _all_contacts(bank)

    replacements: list[tuple[int, int, str]] = []
    entries: list[TokenEntry] = []
    seen_labels: set[str] = set()
    claimed: list[tuple[int, int]] = []

    def remember(entry: TokenEntry) -> None:
        if entry.label not in seen_labels:
            seen_labels.add(entry.label)
            entries.append(entry)

    def is_free(start: int, end: int) -> bool:
        return all(end <= left or start >= right for left, right in claimed)

    # 1. Phone / account numbers typed by the user are always RAW, never PAYEE.
    # They run first so no amount can be minted from digits inside a number.
    raw_spans: list[tuple[int, int, str]] = []
    for match in _SENSITIVE_VALUE_RE.finditer(text):
        for group in range(1, 5):
            if match.group(group) is not None:
                raw_spans.append((*match.span(group), match.group(group)))
    for match in _RAW_PHONE_RE.finditer(text):
        raw_spans.append((match.start(), match.end(), match.group(0)))
    for match in _RAW_LONG_RE.finditer(text):
        raw_spans.append((match.start(), match.end(), match.group(0)))
    for match in _RAW_CN_RE.finditer(text):
        raw_spans.append((match.start(), match.end(), match.group(0)))
    for match in _RAW_EMAIL_RE.finditer(text):
        raw_spans.append((match.start(), match.end(), match.group(0)))
    for match in _RAW_ID_RE.finditer(text):
        raw_spans.append((match.start(), match.end(), match.group(0)))

    for start, end, raw in sorted(
        raw_spans, key=lambda item: (-(item[1] - item[0]), item[0])
    ):
        if not is_free(start, end):
            continue
        entry = vault.mint(
            session,
            "RAW",
            raw,
            OrderedDict([("note", "unverified number")]),
            origin,
        )
        remember(entry)
        replacements.append((start, end, entry.label))
        claimed.append((start, end))

    # 2. Contact names, including aliases and relation words.
    for start, end, matched_contacts in _find_name_matches(text, contacts):
        if not is_free(start, end):
            continue
        labels: list[str] = []
        for contact in matched_contacts:
            entry = vault.mint(
                session,
                "PAYEE",
                contact.id,
                _payee_tags(bank, contact, now_date),
                origin,
            )
            remember(entry)
            labels.append(entry.label)
        if labels:
            replacements.append((start, end, "/".join(labels)))
            claimed.append((start, end))

    # 3. Subscription brand names -> SUB tokens.
    subscriptions = _all_subscriptions(bank)
    for start, end, matched_subscriptions in _find_subscription_matches(
        text, subscriptions
    ):
        if not is_free(start, end):
            continue
        labels = []
        for subscription in matched_subscriptions:
            entry = vault.mint(
                session,
                "SUB",
                subscription.id,
                _subscription_tags(subscription, now_date),
                origin,
            )
            remember(entry)
            labels.append(entry.label)
        if labels:
            replacements.append((start, end, "/".join(labels)))
            claimed.append((start, end))

    # 4. Merchant names -> MERCHANT tokens (rendered generically).
    for start, end, matched_merchants in _find_merchant_matches(text):
        if not is_free(start, end):
            continue
        labels = []
        for real_name in matched_merchants:
            entry = vault.mint(
                session,
                "MERCHANT",
                real_name,
                _merchant_tags(real_name),
                origin,
            )
            remember(entry)
            labels.append(entry.label)
        if labels:
            replacements.append((start, end, "/".join(labels)))
            claimed.append((start, end))

    # 5. Product names, before dates/amounts so "90 天" stays part of the name.
    products = _all_products(bank)
    held_product_ids = _held_product_ids(bank)
    # Keep the canonical context ordering even when the user names p3 first.
    for product in products:
        vault.mint(session, "PROD", product.id,
                   _product_tags(product, held_product_ids), "bank_context")
    for start, end, matched_products in _find_product_matches(text, products):
        if not is_free(start, end):
            continue
        labels = []
        for product in matched_products:
            entry = vault.mint(
                session,
                "PROD",
                product.id,
                _product_tags(product, held_product_ids),
                origin,
            )
            remember(entry)
            labels.append(entry.label)
        if labels:
            replacements.append((start, end, "/".join(labels)))
            claimed.append((start, end))

    # 6. Dates.
    for start, end, target in _find_dates(text, now_date):
        if any(not (end <= left or start >= right) for left, right in claimed):
            continue
        entry = vault.mint(
            session,
            "DATE",
            target,
            OrderedDict([("relative", relative_date_text(target, now_date))]),
            origin,
        )
        remember(entry)
        replacements.append((start, end, entry.label))
        claimed.append((start, end))

    # 7. Amounts.
    for start, end, fen in _find_amounts(text, claimed):
        entry = vault.mint(
            session,
            "AMT",
            fen,
            OrderedDict([("bucket", amount_bucket(fen))]),
            origin,
        )
        remember(entry)
        replacements.append((start, end, entry.label))
        claimed.append((start, end))

    return _apply_replacements(text, replacements), entries


def tokenize_user_text(
    text: str,
    bank: Any,
    vault: TokenVault,
    session: Session,
    now: date | datetime | str,
) -> tuple[str, list[TokenEntry]]:
    """Replace PII in a user's message with semantic tokens."""

    return _tokenize(text, bank, vault, session, now, "user_text")


def tokenize_untrusted(
    text: str,
    bank: Any,
    vault: TokenVault,
    session: Session,
    now: date | datetime | str,
) -> tuple[str, list[TokenEntry]]:
    """Tokenize untrusted text (memos, merchant names, reader output).

    Any existing token for the same real value is downgraded to origin
    ``untrusted``; trust is never upgraded by untrusted text.
    """

    return _tokenize(text, bank, vault, session, now, "untrusted")


# ---------------------------------------------------------------------------
# Conversation context
# ---------------------------------------------------------------------------


def _next_birthday(birthday: date, now: date) -> date:
    try:
        candidate = birthday.replace(year=now.year)
    except ValueError:  # 29 February in a non-leap year
        candidate = date(now.year, 2, 28)
    if candidate < now:
        try:
            candidate = birthday.replace(year=now.year + 1)
        except ValueError:
            candidate = date(now.year + 1, 2, 28)
    return candidate


def build_context(
    bank: Any,
    vault: TokenVault,
    session: Session,
    now: date | datetime | str,
) -> list[str]:
    """Render the bank-side context as semantic-token lines."""

    now_date = _as_date(now)
    lines: list[str] = []

    amounts = [entry for entry in vault.entries(session) if entry.kind == "AMT"]

    # Accounts, with a can_cover answer for every amount minted so far.
    for account_id, account_type in _account_rows(bank):
        tags: OrderedDict[str, Any] = OrderedDict()
        tags["type"] = account_type
        can_cover: list[str] = []
        for amount in amounts:
            try:
                covered = bool(bank.can_cover(amount.real_ref))
            except Exception:  # pragma: no cover - defensive
                covered = False
            can_cover.append(
                f"can_cover({amount.label}): {'yes' if covered else 'no'}"
            )
        if can_cover:
            tags["can_cover"] = can_cover
        entry = vault.mint(session, "ACCT", account_id, tags, "bank_context")
        lines.append(render(entry))

    contacts = _all_contacts(bank)
    contact_by_id = {contact.id: contact for contact in contacts}
    mentioned_ids = [
        entry.real_ref
        for entry in vault.entries(session)
        if entry.kind == "PAYEE"
    ]

    selected: list[Any] = []
    selected_ids: set[str] = set()
    for contact in contacts:
        if contact.relation in _CONTEXT_RELATIONS or contact.id in mentioned_ids:
            selected.append(contact)
            selected_ids.add(contact.id)
    for contact_id in mentioned_ids:
        contact = contact_by_id.get(contact_id)
        if contact is not None and contact.id not in selected_ids:
            selected.append(contact)
            selected_ids.add(contact.id)

    for contact in selected:
        entry = vault.mint(
            session,
            "PAYEE",
            contact.id,
            _payee_tags(bank, contact, now_date),
            "bank_context",
        )
        lines.append(render(entry))

    for card in bank.list_cards():
        tags = OrderedDict([("kind", card.kind), ("status", card.status)])
        entry = vault.mint(session, "CARD", card.id, tags, "bank_context")
        lines.append(render(entry))

    for subscription in bank.list_subscriptions():
        if subscription.status != "active":
            continue
        entry = vault.mint(
            session,
            "SUB",
            subscription.id,
            _subscription_tags(subscription, now_date),
            "bank_context",
        )
        lines.append(render(entry))

    # A holding is "held" only when its bank-computed value is positive.
    held_product_ids = _held_product_ids(bank)
    for product in bank.list_products():
        entry = vault.mint(
            session,
            "PROD",
            product.id,
            _product_tags(product, held_product_ids),
            "bank_context",
        )
        lines.append(render(entry))

    for real_name, _label in MERCHANT_SPECS:
        entry = vault.mint(
            session,
            "MERCHANT",
            real_name,
            _merchant_tags(real_name),
            "bank_context",
        )
        lines.append(render(entry))

    birthday_labels: set[str] = set()
    for contact in contacts:
        if contact.birthday is None:
            continue
        target = _next_birthday(contact.birthday, now_date)
        if not 0 <= (target - now_date).days <= 60:
            continue
        entry = vault.mint(
            session,
            "DATE",
            target,
            OrderedDict([("relative", relative_date_text(target, now_date))]),
            "bank_context",
        )
        if entry.label not in birthday_labels:
            birthday_labels.add(entry.label)
            lines.append(render(entry))

    return lines


# ---------------------------------------------------------------------------
# Detokenization (final replies back to the user)
# ---------------------------------------------------------------------------


def _last4(value: Any) -> str:
    digits = "".join(char for char in str(value) if char.isdigit())
    return (digits or "0000").rjust(4, "0")[-4:]


def _bank_row(bank: Any, sql: str, params: tuple[Any, ...]) -> Any:
    conn = getattr(bank, "conn", None)
    if conn is None:
        return None
    return conn.execute(sql, params).fetchone()


def _display_value(entry: TokenEntry, bank: Any) -> str:
    customer_id = entry.customer_id or getattr(bank, "customer_id", "")

    if entry.kind == "PAYEE":
        row = _bank_row(
            bank,
            "SELECT name, account_no FROM contacts "
            "WHERE customer_id = ? AND id = ?",
            (customer_id, entry.real_ref),
        )
        if row is not None:
            return f"{row['name']} ···{_last4(row['account_no'])}"
        return str(entry.real_ref)

    if entry.kind == "AMT":
        return bank_money.fmt_fen(int(entry.real_ref))

    if entry.kind == "DATE":
        target = date.fromisoformat(str(entry.real_ref))
        return f"{target.month}月{target.day}日"

    if entry.kind == "RAW":
        return f"···{_last4(entry.real_ref)}"

    if entry.kind == "ACCT":
        row = _bank_row(
            bank,
            "SELECT type FROM accounts WHERE customer_id = ? AND id = ?",
            (customer_id, entry.real_ref),
        )
        if row is not None:
            return {"current": "活期账户", "savings": "储蓄账户"}.get(
                row["type"], row["type"]
            )
        return str(entry.real_ref)

    if entry.kind == "CARD":
        row = _bank_row(
            bank,
            "SELECT last4 FROM cards WHERE customer_id = ? AND id = ?",
            (customer_id, entry.real_ref),
        )
        return f"···{row['last4']}" if row is not None else str(entry.real_ref)

    if entry.kind == "SUB":
        row = _bank_row(
            bank,
            "SELECT merchant FROM subscriptions WHERE customer_id = ? AND id = ?",
            (customer_id, entry.real_ref),
        )
        return row["merchant"] if row is not None else str(entry.real_ref)

    if entry.kind == "PROD":
        row = _bank_row(
            bank,
            "SELECT name FROM products WHERE id = ?",
            (entry.real_ref,),
        )
        return row["name"] if row is not None else str(entry.real_ref)

    if entry.kind == "POT":
        row = _bank_row(
            bank,
            "SELECT name FROM pots WHERE customer_id = ? AND id = ?",
            (customer_id, entry.real_ref),
        )
        return row["name"] if row is not None else str(entry.real_ref)

    if entry.kind == "MERCHANT":
        return str(entry.real_ref)

    return str(entry.real_ref)


def detokenize(
    text: str,
    vault: TokenVault,
    session: Session,
    bank: Any,
) -> str:
    """Replace token labels in a final reply with user-facing values."""

    def replace(match: re.Match[str]) -> str:
        label = match.group(0)
        try:
            entry = vault.resolve(label, session)
        except TokenError:
            vault.unknown_labels.append(label)
            return label
        return _display_value(entry, bank)

    return _TOKEN_LABEL_RE.sub(replace, text)
