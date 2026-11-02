"""Scenario 4 (card management) intent rules. See agent/intents/__init__.py."""

from __future__ import annotations

import re
import json
from typing import Any

from security.policy import DEFAULT_ELDER_PERSONAS
from security.tokens import Session, amount_bucket

_SAFE_VIRTUAL_LIMIT_FEN = 20_000

_CARD_HINT = re.compile(r"\bcard\b|卡片|银行卡|卡", re.IGNORECASE)
_OTHER_WRITE = re.compile(
    r"\b(?:transfer|send|pay|schedule|withdraw|deposit|buy|purchase|order|"
    r"wire|move|remit|give|tip|e-?transfer|lend|repay|split|"
    r"subscribe|subscriptions?|cancel|refund|recharge|top\s*up|bill|spending|spend|spent|transactions?|statements?|income|expenses?)\b|"
    r"转账|汇款|转给|打钱|缴费|账单|花销|花费|购买|下单|充值|提现|存款|退款|"
    r"订阅|会员|收支|支出|进账|收入|报告|流水|(?:给|向)(?!谁)[^\s，。,.!?]{1,20}(?:转|汇|打钱)|"
    r"(?:转|汇|打钱)\s*(?:<AMT_|\d|¥|￥)|"
    r"打\s*(?:\d|[一二两三四五六七八九十百千万]+\s*(?:块|元))|还(?!原|有|是|要|在|没|能|可|剩)|汇",
    re.IGNORECASE,
)
_LOSS = re.compile(
    r"(?:卡|卡片|银行卡).{0,8}(?:丢|不见|找不到|遗失|被偷|被盗|挂失)|"
    r"(?:丢|不见|找不到|遗失|被偷|被盗|挂失).{0,8}(?:卡|卡片|银行卡)|"
    r"\b(?:lost|stolen|missing|misplaced)\b.{0,30}\bcard\b|"
    r"\bcard\b.{0,30}\b(?:lost|stolen|missing|misplaced)\b",
    re.IGNORECASE,
)
_VIRTUAL = re.compile(r"虚拟(?:银行)?卡|virtual\s+card", re.IGNORECASE)
_QUERY = re.compile(
    r"多少|什么|状态|哪些|查询|查看|看看|show|list|what|which|status|how\s+much",
    re.IGNORECASE,
)
_RESTORE = re.compile(
    r"(?:恢复|还原|调回|改回|退回|撤销).{0,10}(?:限额|额度)|"
    r"(?:限额|额度).{0,10}(?:恢复|还原|调回|改回|退回)|"
    r"\b(?:restore|revert|reset|undo|put\s+back)\b.{0,30}\b(?:card\s+)?limit\b|"
    r"\blimit\b.{0,30}\b(?:back|restore|revert)\b",
    re.IGNORECASE,
)
_LIMIT = re.compile(r"限额|额度|\blimit\b", re.IGNORECASE)
_SET_LIMIT = re.compile(
    r"改|调|设|变为|变成|提高|降低|提升|下调|"
    r"\b(?:set|change|adjust|raise|lower|increase|decrease|make|update)\b",
    re.IGNORECASE,
)
_BLOCK_ONLINE = re.compile(
    r"屏蔽.{0,8}网购|"
    r"(?:不能|禁止|关闭|关掉|停用|禁用|别|不要).{0,8}"
    r"(?:网购|网上支付|线上支付|网上消费|线上消费|网络支付)|"
    r"(?:网购|网上支付|线上支付|网上消费|线上消费|网络支付).{0,8}"
    r"(?:关闭|关掉|停用|禁止|不能|别|不要)|"
    r"\b(?:block|disable|turn\s+off|stop)\b.{0,30}\b(?:online|internet)\b|"
    r"\b(?:can'?t|cannot|can\s+not)\b.{0,20}\b(?:shop|pay)\b.{0,15}\bonline\b",
    re.IGNORECASE,
)
_UNBLOCK_ONLINE = re.compile(
    r"(?:恢复|开通|解除|取消|允许|重新开启|打开).{0,8}"
    r"(?:网购|网上支付|线上支付|网上消费|线上消费|网络支付)|"
    r"(?:网购|网上支付|线上支付|网上消费|线上消费|网络支付).{0,8}"
    r"(?:恢复|开通|解除|取消|允许)|"
    r"\b(?:unblock|enable|allow|restore|reactivate)\b.{0,30}\b(?:online|internet)\b|"
    r"\b(?:turn|switch)\b.{0,20}\bonline\b.{0,20}\b(?:on|back)\b",
    re.IGNORECASE,
)
_WORK_ONLINE = re.compile(
    r"\b(?:work|shop|pay)\b.{0,20}\bonline\b.{0,20}\bagain\b|"
    r"\bonline\b.{0,20}\bagain\b",
    re.IGNORECASE,
)
_FREEZE = re.compile(
    r"(?:冻结|冻住|锁定|挂起|停用).{0,12}(?:卡|卡片|银行卡)|"
    r"(?:卡|卡片|银行卡).{0,12}(?:冻结|冻住|锁定|挂起|停用)|"
    r"\b(?:freeze|lock|suspend|block)\b.{0,20}\b(?:card|卡)\b|"
    r"\b(?:card|卡)\b.{0,20}\b(?:freeze|lock|suspend|block)\b",
    re.IGNORECASE,
)
_UNFREEZE = re.compile(
    r"(?:解冻|解锁|解封|恢复使用|解除冻结).{0,12}(?:卡|卡片|银行卡)|"
    r"(?:卡|卡片|银行卡).{0,12}(?:解冻|解锁|解封|恢复使用|解除冻结)|"
    r"\b(?:unfreeze|unlock|reactivate|thaw)\b.{0,20}\b(?:card|卡)\b|"
    r"\b(?:card|卡)\b.{0,20}\b(?:unfreeze|unlock|reactivate|thaw)\b",
    re.IGNORECASE,
)
_LIST = re.compile(
    r"(?:我的|看看|查看|查一下|查询|显示|有哪些|列出).{0,8}(?:卡|卡片|银行卡)|"
    r"(?:卡|卡片|银行卡).{0,8}(?:列表|有哪些|什么状态|状态)|"
    r"\b(?:show|list|view|see|check)\b.{0,20}\b(?:my\s+)?cards?\b|"
    r"\bmy\s+cards?\b|\bcard\s+(?:status|list)\b|\bwhat\s+cards\b",
    re.IGNORECASE,
)
# A card clause plus a spending question must reach the planner so it can plan
# both actions instead of the card rule (or the bill rule) answering only one.
_SPENDING_QUESTION = re.compile(
    r"花多少|还能花|能花|剩多少|"
    r"\bhow\s+much\s+can\s+i\s+spend\b|"
    r"\bspending\s+left\b|\bbudget\s+left\b",
    re.IGNORECASE,
)
# "先冻结" / "just freeze": a lost card is frozen only, not reported lost.
_FREEZE_ONLY = re.compile(
    r"先(?:把[^，。,.!?]{0,6})?冻|只冻|冻一下|冻结就行|"
    r"\b(?:just|only)\s+freeze\b|"
    r"\bfreeze\s+(?:it|this|the\s+card|my\s+card)\s+(?:first|for\s+now|only)\b|"
    r"\bfreeze\s+for\s+now\b",
    re.IGNORECASE,
)

_CARD_WRITE_TOOLS = frozenset({
    "freeze_card", "unfreeze_card", "report_card_lost", "create_virtual_card",
    "set_card_limit", "restore_card_limit", "set_card_control", "close_virtual_card",
})
_NEGATED_ACTION = re.compile(
    r"\b(?:don['’]t|do\s+not|never)(?:\s+want\s+to)?\s+"
    r"(?:freeze|unfreeze|report|create|open|restore|revert|reset|undo|disable|block|enable|"
    r"unlock|lock|set|change|raise|lower|increase|decrease|turn|switch)\b|"
    r"(?:不要|不想|先别|别|请勿|不).{0,12}(?:恢复|还原|调回|改回|关闭|关掉|"
    r"禁用|停用|开通|打开|冻结|解冻|挂失|开卡|设置|调整|提高|降低)", re.I,
)


def _ambiguous_action(raw: str) -> bool:
    relevant = bool(_CARD_HINT.search(raw) or _RESTORE.search(raw)
                    or _BLOCK_ONLINE.search(raw) or _UNBLOCK_ONLINE.search(raw))
    if not relevant:
        return False
    if _NEGATED_ACTION.search(raw):
        return True
    if re.search(r"\bnot\s+(?:lost|stolen|missing)\b|没丢|没有丢", raw, re.I):
        return True
    # A single action shortcut cannot represent contradictory or multiple
    # independent card writes. Lost-card freeze + report remains one flow.
    actions = [bool(pattern.search(raw)) for pattern in
               (_FREEZE, _UNFREEZE, _RESTORE, _BLOCK_ONLINE, _UNBLOCK_ONLINE)]
    if _LIMIT.search(raw) and _SET_LIMIT.search(raw) and not _RESTORE.search(raw):
        actions.append(True)
    return sum(actions) > 1


def guard_model_plan(ctx: Any, raw: str) -> str | None:
    """A model retry cannot turn negated/ambiguous card wording into a write."""
    if not _ambiguous_action(str(ctx.raw_text or "")):
        return None
    try:
        output = json.loads(raw)
        steps = output.get("steps", []) if isinstance(output, dict) else []
    except (ValueError, TypeError):
        return None
    if not isinstance(steps, list) or not any(isinstance(step, dict) and step.get("tool") in _CARD_WRITE_TOOLS for step in steps):
        return None
    question = ("请确认你要对卡片做什么；目前没有更改卡片。" if ctx.lang == "zh"
                else "Please confirm which card action you want. The card has not been changed.")
    from agent.intents import remember_question
    remember_question(ctx.customer_id, ctx.session_id, question)
    return json.dumps({"kind": "clarify", "question": question, "options": []}, ensure_ascii=False)


def _step(step_id: str, tool: str, args: dict[str, Any] | None = None,
          after: list[str] | None = None) -> dict[str, Any]:
    return {"id": step_id, "tool": tool, "args": args or {}, "after": after or []}


def _plan(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "plan", "steps": list(steps)}


def _clarify(lang: str) -> dict[str, Any]:
    en = str(lang).lower().startswith("en")
    question = "What card limit would you like?" if en else "请告诉我卡片限额"
    return {"kind": "clarify", "question": question, "options": []}


def _session(ctx: Any) -> Session:
    return Session(
        session_id=ctx.session_id, plan_id=ctx.plan_id,
        customer_id=ctx.customer_id,
    )


def _current_user_text(tokenized: str) -> str:
    """Only the final ``User:`` line, never an earlier context amount."""

    for line in reversed(str(tokenized).splitlines()):
        match = re.match(r"\s*User:\s*(.*)$", line)
        if match:
            return match.group(1)
    return str(tokenized)


def _cards(ctx: Any) -> list[dict[str, Any]]:
    try:
        rows = ctx.conn.execute(
            "SELECT id, kind, status, online_limit_fen FROM cards "
            "WHERE customer_id = ? ORDER BY id",
            (ctx.customer_id,),
        ).fetchall()
    except Exception:  # noqa: BLE001 - no readable cards means no card rule
        return []
    return [
        {"id": row["id"], "kind": row["kind"], "status": row["status"],
         "limit": int(row["online_limit_fen"])}
        for row in rows
    ]


def _default_physical(cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    physical = [
        card for card in cards
        if card["kind"] == "physical" and card["status"] not in {"closed", "lost"}
    ]
    return physical[0] if len(physical) == 1 else None


def _select_card(
    ctx: Any, text: str, cards: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    labels = re.findall(r"<CARD_[0-9]+>", text)
    if labels:
        if len(set(labels)) != 1:
            return None, None
        by_id = {card["id"]: card for card in cards}
        try:
            entry = ctx.vault.resolve(labels[0], _session(ctx))
        except Exception:
            return None, None
        return (by_id[entry.real_ref], entry.label) if entry.kind == "CARD" and entry.real_ref in by_id else (None, None)
    if _VIRTUAL.search(ctx.raw_text):
        candidates = [card for card in cards if card["kind"] == "virtual" and card["status"] not in {"closed", "lost"}]
        card = candidates[0] if len(candidates) == 1 else None
    else:
        card = _default_physical(cards)
    if card is None:
        return None, None
    entry = ctx.vault.mint(
        _session(ctx),
        "CARD",
        card["id"],
        {"kind": card["kind"], "status": card["status"]},
        "bank_context",
    )
    return card, entry.label


def _amount_token(ctx: Any, text: str, default: int | None = None) -> str | None:
    labels = re.findall(r"<AMT_[0-9]+>", text)
    if labels:
        return labels[-1]
    if default is None:
        return None
    entry = ctx.vault.mint(
        _session(ctx), "AMT", default,
        {"bucket": amount_bucket(default)}, "bank_context",
    )
    return entry.label


def _previous_limit(ctx: Any, card_id: str) -> int | None:
    try:
        row = ctx.conn.execute(
            "SELECT previous_limit_fen FROM card_limit_history "
            "WHERE customer_id = ? AND card_id = ? ORDER BY rowid DESC LIMIT 1",
            (ctx.customer_id, card_id),
        ).fetchone()
    except Exception:  # noqa: BLE001 - the root creates this table
        return None
    if row is None or row["previous_limit_fen"] is None:
        return None
    return int(row["previous_limit_fen"])


def _is_elder(ctx: Any) -> bool:
    from security import compliance
    try:
        row = ctx.conn.execute(
            "SELECT persona_id FROM customers WHERE id = ?", (ctx.customer_id,)
        ).fetchone()
    except Exception:  # noqa: BLE001 - unknown persona is not elder
        return False
    return (row is not None and row["persona_id"] in DEFAULT_ELDER_PERSONAS) or compliance.elder_mode(ctx.conn, ctx.customer_id)


def resolve(ctx: Any) -> dict[str, Any] | None:
    raw = ctx.raw_text or ""
    if not raw.strip():
        return None
    if _SPENDING_QUESTION.search(raw):
        return None
    if _ambiguous_action(raw):
        return None
    if re.search(r"\b(?:don't|do not|never)(?:\s+want\s+to)?\s+(?:freeze|unfreeze|report|create|open)\b|\bnot\s+(?:lost|stolen|missing)\b|(?:不要|不想|先别|别)(?:冻结|解冻|挂失|开卡)|没丢|没有丢", raw, re.IGNORECASE):
        return None
    if re.search(r"\b(?:what if|how do|how can|suppose)\b|如果|假如|如何", raw, re.IGNORECASE):
        return None
    # A card clause plus an unrelated money write must go to the normal planner,
    # otherwise the card rule would silently drop the transfer/bill/subscription.
    if _OTHER_WRITE.search(raw) and not re.fullmatch(r"\s*(?:unblock|enable|restore)\s+online\s+shopping\s+category[.!?]?\s*", raw, re.IGNORECASE):
        return None
    # A named contact means the message also asks for a payment: leave the
    # whole request to the planner so nothing is silently dropped.
    if "<PAYEE_" in _current_user_text(ctx.tokenized_text):
        return None

    text = _current_user_text(ctx.tokenized_text)
    cards = _cards(ctx)

    if _LOSS.search(raw):
        card, token = _select_card(ctx, text, cards)
        if card is None or token is None:
            return None
        freeze_only = bool(_FREEZE_ONLY.search(raw)) and not re.search(
            r"挂失|\breport\b", raw, re.IGNORECASE
        )
        if freeze_only:
            return _plan(_step("s1", "freeze_card", {"card": token}))
        if card["status"] == "frozen":
            return _plan(_step("s1", "report_card_lost", {"card": token}))
        if card["status"] != "active":
            return None
        return _plan(
            _step("s1", "freeze_card", {"card": token}),
            _step("s2", "report_card_lost", {"card": token}, ["s1"]),
        )

    if (_VIRTUAL.search(raw) or re.search(r"新开.{0,12}网购.{0,8}卡", raw)) and (
        re.search(r"申请|新开|开一|创建|\b(?:apply|open|create|new)\b", raw, re.IGNORECASE)
        or (_LIMIT.search(raw) and not _SET_LIMIT.search(raw) and not _FREEZE.search(raw) and not _UNFREEZE.search(raw))
    ) and not _QUERY.search(raw):
        amount = _amount_token(ctx, text, _SAFE_VIRTUAL_LIMIT_FEN)
        if amount is not None and not re.search(r"<AMT_[0-9]+>", text):
            ctx.vault.mint(_session(ctx), "AMT", _SAFE_VIRTUAL_LIMIT_FEN, {"default_virtual_limit": True}, "bank_context")
        return None if amount is None else _plan(
            _step("s1", "create_virtual_card", {"amount": amount})
        )

    if _RESTORE.search(raw):
        default = _default_physical(cards)
        if default is None:
            return None
        card, token = _select_card(ctx, text, cards)
        if card is None or token is None or card["id"] != default["id"]:
            return None
        previous = _previous_limit(ctx, card["id"])
        if previous is None:
            return _plan(_step("s1", "restore_card_limit"))
        if previous > card["limit"] or (previous > 50_000 and _is_elder(ctx)):
            amount = _amount_token(ctx, "", previous)
            if amount is None:
                return None
            ctx.vault.mint(_session(ctx), "AMT", previous, {"restored_limit": True}, "bank_context")
            return _plan(
                _step("s1", "set_card_limit", {"card": token, "amount": amount})
            )
        return _plan(_step("s1", "restore_card_limit"))

    if _LIMIT.search(raw) and _SET_LIMIT.search(raw):
        amount = _amount_token(ctx, text)
        if amount is None:
            return _clarify(ctx.lang)
        card, token = _select_card(ctx, text, cards)
        if card is None or token is None:
            return None
        return _plan(_step("s1", "set_card_limit", {"card": token, "amount": amount}))

    if _BLOCK_ONLINE.search(raw) or _UNBLOCK_ONLINE.search(raw) or _WORK_ONLINE.search(raw):
        card, token = _select_card(ctx, text, cards)
        if card is None or token is None:
            return None
        enabled = bool(_UNBLOCK_ONLINE.search(raw) or _WORK_ONLINE.search(raw))
        args = {"card": token, "control": "online", "enabled": enabled}
        # The existing graph requires explicit unfreeze wording. Preserve that
        # protection: changing the setting never unfreezes a card implicitly.
        # The result explains that the frozen card still cannot make payments.
        return _plan(_step("s1", "set_card_control", args))

    if _FREEZE.search(raw) and not _UNFREEZE.search(raw):
        card, token = _select_card(ctx, text, cards)
        if card is None or token is None:
            return None
        return _plan(_step("s1", "freeze_card", {"card": token}))

    if _UNFREEZE.search(raw):
        card, token = _select_card(ctx, text, cards)
        if card is None or token is None:
            return None
        return _plan(_step("s1", "unfreeze_card", {"card": token}))

    if _LIST.search(raw):
        return _plan(_step("s1", "list_cards"))

    return None
