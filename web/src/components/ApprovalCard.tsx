// The confirmation card under an assistant reply: amount, payee, tier and the
// checks a red step needs (SMS code, retyped amount, face, family code).
import { useState } from "react";
import { api } from "../api";
import { useApp } from "../state";
import type { Approval, Turn } from "../types";
import Icon from "./Icon";

export function yuan(fen: number | null | undefined): string {
  if (fen === null || fen === undefined) return "";
  return `¥${(fen / 100).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

const TOOL_EN: Record<string, string> = {
  transfer: "Transfer",
  schedule_transfer: "Standing order",
  request_payment: "Request money",
  merchant_order: "Order",
  move_to_pot: "Move to savings pot",
  cancel_subscription: "Cancel subscription",
  freeze_card: "Freeze card",
  unfreeze_card: "Unfreeze card",
  report_card_lost: "Report card lost",
  set_card_limit: "Change card limit",
  create_virtual_card: "New virtual card",
  subscribe_product: "Buy product",
  change_password: "Change password",
};
const TOOL_ZH: Record<string, string> = {
  transfer: "转账",
  schedule_transfer: "定时转账",
  request_payment: "发起收款",
  merchant_order: "下单",
  move_to_pot: "存入储蓄罐",
  cancel_subscription: "取消订阅",
  freeze_card: "冻结卡片",
  unfreeze_card: "解冻卡片",
  report_card_lost: "挂失卡片",
  set_card_limit: "调整卡片限额",
  create_virtual_card: "开通虚拟卡",
  subscribe_product: "购买理财",
  change_password: "修改密码",
};

interface Checks {
  otp: string;
  retype: string;
  face: boolean;
  coOtp: string;
}

export default function ApprovalCard({
  turn,
  onResult,
}: {
  turn: Turn;
  onResult: (next: Turn) => void;
}) {
  const { t, settings } = useApp();
  const zh = settings.lang === "zh";
  const [busy, setBusy] = useState(false);
  const [checks, setChecks] = useState<Record<string, Checks>>({});

  const get = (a: Approval): Checks =>
    checks[a.step_id] ?? { otp: "", retype: "", face: false, coOtp: "" };
  const set = (a: Approval, patch: Partial<Checks>) =>
    setChecks((current) => ({ ...current, [a.step_id]: { ...get(a), ...patch } }));

  const ready = turn.approvals.every((a) => {
    if (!a.needs_mfa) return true;
    const c = get(a);
    return c.otp.length > 0 && c.retype.length > 0 && c.face && (!a.needs_co_approval || c.coOtp.length > 0);
  });

  async function answer(reject: boolean) {
    setBusy(true);
    const approvals: Record<string, unknown> = {};
    for (const a of turn.approvals) {
      if (reject) {
        approvals[a.step_id] = { reject: true };
        continue;
      }
      const c = get(a);
      approvals[a.step_id] = a.needs_mfa
        ? {
            confirm: true,
            otp: c.otp.trim(),
            retyped_amount_fen: Math.round(parseFloat(c.retype.replace(/[,¥\s]/g, "")) * 100) || 0,
            face_ok: c.face,
            ...(a.needs_co_approval ? { co_otp: c.coOtp.trim() } : {}),
          }
        : { confirm: true };
    }
    try {
      onResult(await api.approve(turn.session_id, approvals));
    } finally {
      setBusy(false);
    }
  }

  const anyRed = turn.approvals.some((a) => a.tier === "red");

  return (
    <section
      aria-label={t("confirmTitle")}
      className="card"
      style={{ display: "flex", flexDirection: "column", gap: "0.85rem", boxShadow: "0 -4px 18px rgba(20,33,61,0.08)" }}
    >
      {turn.approvals.map((a) => {
        const c = get(a);
        const name = (zh ? TOOL_ZH : TOOL_EN)[a.tool] ?? a.tool;
        const target = a.payee_display ?? a.target_display ?? "";
        const reasons = a.reasons.map((r) => (zh ? r.message_zh : r.message_en)).filter(Boolean);
        return (
          <div key={a.step_id} style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "0.5rem" }}>
              <div>
                <div className="muted small">
                  {name}
                  {target ? ` · ${target}` : ""}
                  {a.day_of_month ? ` · ${t("everyMonthDay", { day: a.day_of_month })}` : ""}
                </div>
                {a.amount_fen ? (
                  <div className="tabular" style={{ fontSize: "1.75rem", fontWeight: 800 }}>
                    {yuan(a.amount_fen)}
                  </div>
                ) : null}
              </div>
              <span className={`badge ${a.tier === "red" ? "badge-red" : a.tier === "yellow" ? "badge-yellow" : "badge-green"}`}>
                {a.tier === "red" ? t("highRisk") : t("needsConfirm")}
              </span>
            </div>
            {reasons.length > 0 && <div className="small muted">{reasons.join(" ")}</div>}
            {a.error && (
              <div className="small" role="alert" style={{ color: "var(--danger-dark)" }}>
                {a.error}
              </div>
            )}

            {a.needs_mfa && (
              <ol style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: "0.45rem" }}>
                <li className={`step-row ${c.otp ? "done" : ""}`}>
                  <span className="step-dot">{c.otp ? <Icon name="check" size={14} strokeWidth={3} /> : 1}</span>
                  <label htmlFor={`otp-${a.step_id}`} style={{ fontWeight: 600, flex: 1 }}>
                    {t("stepSms")}
                    {a.demo_otp && <span className="small muted" style={{ display: "block", fontWeight: 400 }}>{t("smsHint", { code: a.demo_otp })}</span>}
                  </label>
                  <input
                    id={`otp-${a.step_id}`}
                    className="field tabular"
                    inputMode="numeric"
                    value={c.otp}
                    onChange={(e) => set(a, { otp: e.target.value })}
                    style={{ width: "6.5rem" }}
                  />
                </li>
                <li className={`step-row ${c.retype ? "done" : ""}`}>
                  <span className="step-dot">{c.retype ? <Icon name="check" size={14} strokeWidth={3} /> : 2}</span>
                  <label htmlFor={`amt-${a.step_id}`} style={{ fontWeight: 600, flex: 1 }}>
                    {t("stepRetype")}
                  </label>
                  <input
                    id={`amt-${a.step_id}`}
                    className="field tabular"
                    inputMode="decimal"
                    value={c.retype}
                    onChange={(e) => set(a, { retype: e.target.value })}
                    style={{ width: "6.5rem", textAlign: "right" }}
                  />
                </li>
                <li className={`step-row ${c.face ? "done" : ""}`}>
                  <span className="step-dot">{c.face ? <Icon name="check" size={14} strokeWidth={3} /> : 3}</span>
                  <span style={{ fontWeight: 600, flex: 1 }}>{t("stepFace")}</span>
                  <button className="pill-btn" onClick={() => set(a, { face: true })} disabled={c.face}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
                      <Icon name="face" size={16} />
                      {c.face ? t("faceOk") : t("doFace")}
                    </span>
                  </button>
                </li>
                {a.needs_co_approval && (
                  <li className={`step-row ${c.coOtp ? "done" : ""}`}>
                    <span className="step-dot">{c.coOtp ? <Icon name="check" size={14} strokeWidth={3} /> : 4}</span>
                    <label htmlFor={`co-${a.step_id}`} style={{ fontWeight: 600, flex: 1 }}>
                      {t("stepFamily")}
                      {a.demo_co_otp && <span className="small muted" style={{ display: "block", fontWeight: 400 }}>{t("familyHint", { code: a.demo_co_otp })}</span>}
                    </label>
                    <input
                      id={`co-${a.step_id}`}
                      className="field tabular"
                      inputMode="numeric"
                      value={c.coOtp}
                      onChange={(e) => set(a, { coOtp: e.target.value })}
                      style={{ width: "6.5rem" }}
                    />
                  </li>
                )}
              </ol>
            )}
          </div>
        );
      })}

      <div style={{ display: "flex", gap: "0.6rem" }}>
        <button className="secondary-btn" style={{ flex: 1 }} disabled={busy} onClick={() => answer(true)}>
          {t("cancel")}
        </button>
        <button className="primary-btn" style={{ flex: 2 }} disabled={busy || !ready} onClick={() => answer(false)}>
          {t("confirmAll")}
        </button>
      </div>
      {anyRed || turn.approvals.some((a) => a.tool === "transfer") ? (
        <div className="small muted" style={{ textAlign: "center" }}>{t("undoNote")}</div>
      ) : null}
    </section>
  );
}
