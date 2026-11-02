// Home screen: greeting, balance card, quick actions, recent activity and the
// iOS tab bar. Every visible string comes from t().
import Icon from "./Icon";
import { useApp } from "../state";
import type { Lang, Txn } from "../types";

const QUICK_ACTIONS: { icon: string; labelKey: "transfer" | "spending" | "cards" | "savingsAction"; promptEn: string; promptZh: string }[] = [
  { icon: "transfer", labelKey: "transfer", promptEn: "I want to send money", promptZh: "我要转账" },
  { icon: "chart", labelKey: "spending", promptEn: "Where did my money go this month?", promptZh: "这个月钱都花哪了" },
  { icon: "card", labelKey: "cards", promptEn: "Show my cards", promptZh: "我的卡怎么样" },
  { icon: "piggy", labelKey: "savingsAction", promptEn: "I want to save money", promptZh: "我想存钱" },
];

const CARDS_ACTION = QUICK_ACTIONS.find((action) => action.labelKey === "cards");

function formatDay(ts: string, lang: Lang): string {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return ts;
  if (lang === "zh") return `${date.getMonth() + 1}月${date.getDate()}日`;
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" }).format(date);
}

function initials(txn: Txn, lang: Lang): string {
  if (lang === "zh") return txn.counterparty.trim().slice(0, 1);
  return txn.counterparty_en.trim().slice(0, 2).toUpperCase();
}

/** Backend amounts look like "-¥12.00"; show a real minus sign, never a hyphen. */
function formatAmount(txn: Txn): string {
  if (txn.amount_fen >= 0) return txn.display;
  const unsigned = txn.display.replace(/[-−]/g, "").trim();
  return `−${unsigned.startsWith("¥") ? "" : "¥"}${unsigned}`;
}

export default function Home() {
  const { t, personaName, settings, summary, transactions, setScreen, openAssistant } = useApp();
  const lang = settings.lang;
  const elderMode = settings.elder_mode;
  const cardLast4 = (summary?.cards?.[0]?.last4 ?? "").replace(/\*/g, "");
  const actions = QUICK_ACTIONS.slice(0, elderMode ? 2 : QUICK_ACTIONS.length);

  return (
    <>
      <div className="screen home-screen">
        <header
          className="home-header"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "1rem",
            padding: "1.25rem 1rem 0.75rem",
          }}
        >
          <div style={{ minWidth: 0 }}>
            <div className="small muted">{t("greeting")}</div>
            <div style={{ fontSize: "1.2rem", fontWeight: 700, lineHeight: 1.3 }}>{personaName}</div>
          </div>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button
              className="icon-btn"
              type="button"
              aria-label={t("settings")}
              onClick={() => setScreen("settings")}
            >
              <Icon name="gear" />
            </button>
          </div>
        </header>

        <section
          className="balance-card home-balance"
          style={{
            position: "relative",
            overflow: "hidden",
            margin: "0 1rem",
            padding: "1.15rem 1.25rem 1.25rem",
            borderRadius: "1.25rem",
            background: "var(--surface)",
            color: "var(--ink)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "0.5rem",
            }}
          >
            <span style={{ fontSize: "1rem", fontWeight: 700 }}>AnBan 安办</span>
            <span className="small muted">
              {t("debit")} · ···· {cardLast4}
            </span>
          </div>

          <div className="small muted" style={{ marginTop: "1.1rem" }}>
            {t("currentAccount")}
          </div>
          <div className="tabular" style={{ fontSize: "2.1rem", fontWeight: 700, lineHeight: 1.2 }}>
            {summary?.display}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "0.5rem",
              marginTop: "0.9rem",
              paddingTop: "0.75rem",
              borderTop: "1px solid var(--line-soft)",
              fontSize: "0.8rem",
            }}
          >
            <span className="muted">{`${t("limitLeft")} ${elderMode ? "¥500" : "¥1,000"}`}</span>
          </div>
        </section>

        <div
          className="home-quick-actions"
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${elderMode ? 2 : 4}, minmax(0, 1fr))`,
            gap: "0.5rem",
            margin: "1rem",
          }}
        >
          {actions.map((action) => (
            <button
              key={action.icon}
              className="home-quick-action"
              type="button"
              onClick={() => openAssistant(lang === "zh" ? action.promptZh : action.promptEn)}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "0.4rem",
                padding: "0.75rem 0.25rem",
                border: "none",
                borderRadius: "1rem",
                background: "var(--surface)",
                color: "var(--ink)",
                fontSize: "0.8rem",
                fontWeight: 600,
              }}
            >
              <span
                style={{
                  width: "2.4rem",
                  height: "2.4rem",
                  borderRadius: "0.85rem",
                  background: "var(--primary-soft)",
                  color: "var(--primary)",
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Icon name={action.icon} />
              </span>
              {t(action.labelKey)}
            </button>
          ))}
        </div>

        <section className="card home-activity" style={{ margin: "0 1rem", padding: "0.5rem 1rem 0.75rem" }}>
          <div
            className="home-activity-head"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "0.5rem",
              padding: "0.65rem 0",
            }}
          >
            <h2 className="home-activity-title" style={{ margin: 0, fontSize: "1rem" }}>
              {t("recentActivity")}
            </h2>
            <button
              className="home-see-all"
              type="button"
              onClick={() => {}}
              style={{
                border: "none",
                background: "transparent",
                color: "var(--primary)",
                fontSize: "0.85rem",
                fontWeight: 600,
                padding: 0,
              }}
            >
              {t("seeAll")}
            </button>
          </div>

          {transactions.map((txn) => {
            const income = txn.amount_fen > 0;
            return (
              <div
                key={txn.id}
                className="home-txn"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.75rem",
                  padding: "0.65rem 0",
                  borderTop: "1px solid var(--line-soft)",
                }}
              >
                <div
                  aria-hidden="true"
                  style={{
                    flexShrink: 0,
                    width: "2.5rem",
                    height: "2.5rem",
                    borderRadius: "0.8rem",
                    background: "var(--primary-soft)",
                    color: "var(--primary-dark)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "0.8rem",
                    fontWeight: 700,
                  }}
                >
                  {initials(txn, lang)}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: "0.9rem",
                      fontWeight: 600,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {lang === "zh" ? txn.counterparty : txn.counterparty_en}
                  </div>
                  <div
                    className="small muted"
                    style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                  >
                    {txn.hold_reason
                      ? (lang === "zh" ? "安全审核中，资金仍暂存，可申请撤销" : "Held for security review; undo available")
                      : txn.held_until
                      ? `${t("onHoldUntil")} ${new Date(txn.held_until).toLocaleTimeString(lang === "zh" ? "zh-CN" : "en-GB", { hour: "2-digit", minute: "2-digit" })}`
                      : `${formatDay(txn.ts, lang)} · ${txn.category}`}
                  </div>
                </div>
                <div
                  className="tabular"
                  style={{
                    flexShrink: 0,
                    fontSize: "0.9rem",
                    fontWeight: 600,
                    color: income ? "var(--success)" : "var(--ink)",
                  }}
                >
                  {formatAmount(txn)}
                </div>
              </div>
            );
          })}
        </section>
      </div>

      <nav className="tabbar" aria-label={t("tabHome")}>
        <button className="tab active" type="button" aria-current="page">
          <Icon name="house" />
          {t("tabHome")}
        </button>
        <button
          className="tab"
          type="button"
          onClick={() => {
            if (CARDS_ACTION) openAssistant(lang === "zh" ? CARDS_ACTION.promptZh : CARDS_ACTION.promptEn);
          }}
        >
          <Icon name="card" />
          {t("tabCards")}
        </button>
        <button className="tab" type="button" onClick={() => openAssistant()}>
          <Icon name="chat" />
          {t("tabAssistant")}
        </button>
        <button className="tab" type="button" onClick={() => setScreen("settings")}>
          <Icon name="person" />
          {t("tabMe")}
        </button>
      </nav>
    </>
  );
}
