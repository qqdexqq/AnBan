"""Receipt selection only; early settlement still needs the separate OTP gate."""
import hmac
import json
import re
from agent.executor import _undo_sig
from agent.recent_actions import read_recent_transfer

_REQUEST = re.compile(r"^(?:(?:please|can you|could you|just|i want to)\s+)*(?:send|release|transfer)\s+(?:(?:it|(?:that|this)(?:\s+(?:transfer|money))?|the money|the transfer)\s+)?(?:now|immediately)(?:\s+(?:please|(?:instead of|without) waiting(?: (?:2|two) hours)?))*[.!?\s]*$|^(?:请|帮我)?(?:立即|马上)(?:转出|转账|汇出)(?:这笔|这笔钱|吧)?[。！\s]*$", re.I)

def send_now_request(text):
    return bool(_REQUEST.fullmatch(text.strip()))

def resolve_send_now(services, customer_id, session_id, lang):
    result = {"status": "clarify", "reply": "Which held transfer? Open its receipt and choose Send now." if lang == "en" else "请选择本对话中尚未转出的转账，点击「立即转出」。", "plan_id": "", "plan_view": [], "actions": []}
    recent = read_recent_transfer(services.conn, services.secret_key, customer_id, session_id)
    if recent is None or recent.status != "pending":
        return result
    matches = []
    for row in services.conn.execute("SELECT * FROM exec_undo WHERE customer_id = ? AND tool = 'cancel_transfer' AND used = 0", (customer_id,)):
        if not re.fullmatch(re.escape(session_id) + r":p[0-9]+", row["plan_id"]):
            continue
        expected = _undo_sig(services.secret_key, customer_id, row["plan_id"], row["step_id"], row["tool"], row["args_json"], row["created_at"])
        if not hmac.compare_digest(expected, row["sig"]):
            continue
        args = json.loads(row["args_json"])
        if (args.get("transfer_id") or args.get("id")) == recent.transfer_id:
            matches.append(row)
    if len(matches) != 1:
        return result
    row = matches[0]
    return {"status": "done", "reply": "Send now needs fresh phone verification. Review this transfer and request a code." if lang == "en" else "立即转出需要新的手机验证。请核对这笔转账并获取验证码。", "plan_id": row["plan_id"], "plan_view": [{"step_id": row["step_id"], "tool": "transfer", "status": "done", "tier": "red", "after": []}], "actions": ["send_now"]}
