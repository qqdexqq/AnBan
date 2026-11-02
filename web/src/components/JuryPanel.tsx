// Desktop "behind the scenes" panel for the jury: user text, the plan the
// model proposed, the tier code assigned, regulator-side records and the audit
// chain. AssistantSheet publishes each completed turn on juryBus.
import { Fragment, useEffect, useState } from "react";
import Icon from "./Icon";
import { api, login } from "../api";
import { useApp } from "../state";
import type { ComplianceReport, Tier, Turn } from "../types";

export const juryBus = {
  lastTurn: null as Turn | null,
  lastUserText: "",
  listeners: new Set<() => void>(),
  publish(turn: Turn, userText: string) {
    this.lastTurn = turn;
    this.lastUserText = userText;
    this.listeners.forEach((listener) => listener());
  },
};

const TEXT = "#e8ecf1";
const MUTED = "#9aa4b2";
const BORDER = "#222833";

const TIER_STYLE: Record<Tier, { color: string; background: string }> = {
  green: { color: "#3dd68c", background: "rgba(61,214,140,0.08)" },
  yellow: { color: "#f5b841", background: "rgba(245,184,65,0.08)" },
  red: { color: "#ff6b5b", background: "rgba(255,107,91,0.08)" },
};

const COPY = {
  en: {
    whatSaid: "1 · What the user said → what the model saw",
    privacy:
      "names, account numbers and exact amounts are replaced by tokens like <PAYEE_1>, <AMT_1> before the model sees them",
    compliance: "3 · Compliance (simulated regulator side)",
    empty: "Send a message in the app to see the pipeline.",
    chainBroken: "chain broken",
    nextDay: "Next day ⏭",
    nextMonthDay: "Next month day",
  },
  zh: {
    whatSaid: "1 · 用户说了什么 → 模型看到了什么",
    privacy:
      "姓名、账号和真实金额在模型看到之前都被替换成 <PAYEE_1>、<AMT_1> 这样的代号",
    compliance: "3 · 合规（模拟监管侧）",
    empty: "在 App 里发一条消息，就能看到整个流程。",
    chainBroken: "链已损坏",
    nextDay: "下一天 ⏭",
    nextMonthDay: "下个月同日",
  },
} as const;

function prettyTool(tool: string): string {
  return tool.replace(/_/g, " ");
}

function clockTime(createdAt: string): string {
  const date = new Date(createdAt);
  if (Number.isNaN(date.getTime())) return createdAt;
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  return `${hours}:${minutes}`;
}

interface AuditState {
  lines: string[];
  chain_ok: boolean;
}

export default function JuryPanel() {
  const { t, settings, activity, persona, assistantChatId, refresh } = useApp();
  const copy = COPY[settings.lang];
  const [turn, setTurn] = useState<Turn | null>(juryBus.lastTurn);
  const [userText, setUserText] = useState(juryBus.lastUserText);
  const [reports, setReports] = useState<ComplianceReport[] | null>(null);
  const [audit, setAudit] = useState<AuditState | null>(null);
  const [clockBusy, setClockBusy] = useState(false);

  async function advanceClock(minutes: number) {
    if (clockBusy) return;
    setClockBusy(true);
    try {
      await api.demoClock(minutes);
      // Login tokens follow the demo clock, so jumping days ahead expires
      // them: sign in again as the same customer before refreshing.
      await login(persona);
      // Refresh so newly due standing orders and reminders notify.
      await refresh();
    } catch (error) {
      console.error(error);
    } finally {
      setClockBusy(false);
    }
  }

  useEffect(() => {
    // The inspection panel belongs to the selected conversation too.
    juryBus.lastTurn = null;
    juryBus.lastUserText = "";
    setTurn(null);
    setUserText("");
    setReports(null);
    setAudit(null);
  }, [persona, assistantChatId]);

  useEffect(() => {
    const listener = () => {
      setTurn(juryBus.lastTurn);
      setUserText(juryBus.lastUserText);
    };
    juryBus.listeners.add(listener);
    return () => {
      juryBus.listeners.delete(listener);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .compliance()
      .then((status) => {
        if (!cancelled) setReports(status.reports);
      })
      .catch((error) => {
        console.error(error);
        if (!cancelled) setReports([]);
      });
    return () => {
      cancelled = true;
    };
  }, [activity, persona, assistantChatId]);

  useEffect(() => {
    let cancelled = false;
    api
      .audit(settings.lang)
      .then((next) => {
        if (!cancelled) setAudit(next);
      })
      .catch((error) => {
        console.error(error);
        if (!cancelled) setAudit(null);
      });
    return () => {
      cancelled = true;
    };
  }, [activity, settings.lang, turn, persona, assistantChatId]);

  const counts = new Map<string, number>();
  for (const report of reports ?? []) {
    counts.set(report.kind, (counts.get(report.kind) ?? 0) + 1);
  }
  const reasons = Array.from(
    new Set(
      (turn?.approvals ?? []).flatMap((approval) =>
        approval.reasons.map((reason) =>
          settings.lang === "zh" ? reason.message_zh : reason.message_en,
        ),
      ),
    ),
  );

  return (
    <aside className="jury" aria-label={t("behind")}>
      <header
        className="jury-card"
        style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px" }}
      >
        <span
          aria-hidden="true"
          style={{
            width: 32,
            height: 32,
            borderRadius: 9,
            background: "rgba(61,214,140,0.10)",
            color: "#3dd68c",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <Icon name="shield" size={18} />
        </span>
        <span style={{ fontSize: 15, fontWeight: 700, color: TEXT, letterSpacing: 0.2 }}>
          AnBan · <span style={{ color: MUTED }}>{t("behind")}</span>
        </span>
      </header>

      <section className="jury-card">
        <h2>{t("demo")}</h2>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          <button
            type="button"
            className="jury-demo-btn"
            disabled={clockBusy}
            onClick={() => advanceClock(1440)}
          >
            {copy.nextDay}
          </button>
          <button
            type="button"
            className="jury-demo-btn"
            disabled={clockBusy}
            onClick={() => advanceClock(1440 * 7)}
          >
            {copy.nextMonthDay}
          </button>
        </div>
      </section>

      <section className="jury-card">
        <h2>{copy.whatSaid}</h2>
        {userText && (
          <div
            style={{
              color: TEXT,
              fontSize: 14,
              lineHeight: 1.5,
              whiteSpace: "pre-wrap",
              overflowWrap: "anywhere",
            }}
          >
            {userText}
          </div>
        )}
        <div
          className="mono"
          style={{ marginTop: userText ? 10 : 0, color: MUTED, fontSize: 11.5, lineHeight: 1.55 }}
        >
          {copy.privacy}
        </div>
      </section>

      <section className="jury-card">
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
          <h2 style={{ margin: 0 }}>{`2 · ${t("planTitle")}`}</h2>
          {turn && (
            <span
              className="mono"
              style={{
                flexShrink: 0,
                border: `1px solid ${BORDER}`,
                background: "#171d26",
                color: MUTED,
                borderRadius: 999,
                padding: "2px 7px",
                fontSize: 10,
                lineHeight: 1.6,
              }}
            >
              {turn.status}
            </span>
          )}
        </div>

        {!turn ? (
          <div style={{ marginTop: 10, color: MUTED, fontSize: 12.5 }}>{copy.empty}</div>
        ) : (
          <>
            {turn.plan_view.length > 0 && (
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "center",
                  gap: 8,
                  marginTop: 12,
                }}
              >
                {turn.plan_view.map((step, index) => {
                  const tier = TIER_STYLE[step.tier] ?? { color: MUTED, background: "#171d26" };
                  return (
                    <Fragment key={`${step.step_id}-${index}`}>
                      {index > 0 && (
                        <span aria-hidden="true" style={{ color: MUTED, fontSize: 14 }}>
                          →
                        </span>
                      )}
                      <div
                        style={{
                          border: `1px solid ${tier.color}`,
                          background: tier.background,
                          borderRadius: 10,
                          padding: "8px 10px",
                          flex: "0 0 auto",
                        }}
                      >
                        <div className="mono" style={{ color: tier.color, fontSize: 10.5 }}>
                          {step.step_id} · {(step.tier ?? "unverified").toUpperCase()} · {step.status}
                        </div>
                        <div style={{ marginTop: 4, color: TEXT, fontSize: 12 }}>
                          {prettyTool(step.tool)}
                        </div>
                      </div>
                    </Fragment>
                  );
                })}
              </div>
            )}

            {reasons.length > 0 && (
              <ul
                style={{
                  margin: "10px 0 0",
                  padding: "0 0 0 16px",
                  color: MUTED,
                  fontSize: 12,
                  lineHeight: 1.55,
                }}
              >
                {reasons.map((reason, index) => (
                  <li key={`${reason}-${index}`}>{reason}</li>
                ))}
              </ul>
            )}
          </>
        )}
      </section>

      <section className="jury-card">
        <h2>{copy.compliance}</h2>
        {counts.size > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
            {Array.from(counts.entries()).map(([kind, count]) => (
              <span
                key={kind}
                className="mono"
                style={{
                  border: `1px solid ${BORDER}`,
                  background: "#171d26",
                  borderRadius: 999,
                  color: TEXT,
                  fontSize: 10.5,
                  lineHeight: 1.6,
                  padding: "2px 7px",
                }}
              >
                {kind} <span style={{ color: MUTED }}>{count}</span>
              </span>
            ))}
          </div>
        )}
        {(reports ?? []).slice(0, 4).map((report, index) => (
          <div
            key={`${report.kind}-${report.created_at}-${index}`}
            className="mono"
            style={{
              display: "flex",
              gap: 8,
              borderTop: index === 0 ? "none" : `1px solid ${BORDER}`,
              padding: "5px 0",
              fontSize: 11.5,
            }}
          >
            <span style={{ color: MUTED }}>{clockTime(report.created_at)}</span>
            <span style={{ color: TEXT }}>{report.kind}</span>
          </div>
        ))}
      </section>

      <section className="jury-card">
        <h2>{`4 · ${t("auditTitle")}`}</h2>
        <div
          className="mono"
          style={{
            color: "#cfd6df",
            fontSize: 12,
            lineHeight: 1.55,
            whiteSpace: "pre-wrap",
            overflowWrap: "anywhere",
          }}
        >
          {(audit?.lines ?? []).slice(-8).map((line, index) => (
            <div key={`${line}-${index}`}>{line}</div>
          ))}
        </div>
        {audit && (
          <div
            className="mono"
            style={{
              marginTop: 8,
              color: audit.chain_ok ? "#3dd68c" : "#ff6b5b",
              fontSize: 12,
            }}
          >
            {audit.chain_ok ? `✓ ${t("chainOk")}` : copy.chainBroken}
          </div>
        )}
      </section>
    </aside>
  );
}
