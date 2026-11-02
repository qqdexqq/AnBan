// The AI assistant: a sheet over the home screen with the chat, suggestions,
// confirmation cards, scam alerts and undo.
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { PersonaId } from "../api";
import { SUGGESTIONS } from "../i18n";
import { useApp } from "../state";
import type { ChatMessage, Turn } from "../types";
import ApprovalCard from "./ApprovalCard";
import Icon from "./Icon";
import { juryBus } from "./JuryPanel";
import SendNowCard from "./SendNowCard";
import VoiceInput from "./VoiceInput";
import UndoConfirm from "./UndoConfirm";

const CONSENT_KEY = "anban.ai-consent.voice-v1";
const OPTION_RE = /^\s*(\d+)\.\s+(.+)$/;

function readConsent(): boolean {
  try {
    return window.localStorage.getItem(CONSENT_KEY) === "yes";
  } catch {
    return false;
  }
}

/** A compact date for the history list. */
function formatChatDate(value: string, zh: boolean): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(zh ? "zh-CN" : "en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Split a clarify reply into the question and numbered options. */
function splitOptions(reply: string): { text: string; options: { n: string; label: string }[] } {
  const lines = reply.split("\n");
  const options = lines.map((line) => OPTION_RE.exec(line)).filter((m): m is RegExpExecArray => m !== null);
  return {
    text: lines.filter((line) => !OPTION_RE.test(line)).join("\n").trim(),
    options: options.map((m) => ({ n: m[1], label: m[2] })),
  };
}

function ScamAlert({ reply }: { reply: string }) {
  const { t } = useApp();
  return (
    <section
      aria-label={t("scamTitle")}
      className="card"
      style={{ border: "1.5px solid var(--danger)", display: "flex", flexDirection: "column", gap: "0.75rem" }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
        <span style={{ width: "3rem", height: "3rem", borderRadius: "999px", background: "var(--danger)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
          <Icon name="alert" size={26} />
        </span>
        <div>
          <h3 style={{ margin: 0, fontSize: "1.15rem", fontWeight: 800, color: "var(--danger-dark)" }}>{t("scamTitle")}</h3>
          <div className="small muted">{t("scamPaused")}</div>
        </div>
      </div>
      <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>{reply}</p>
      <p className="small muted" style={{ margin: 0 }}>{t("scamSafe")}</p>
      <a className="danger-btn" href="tel:96110">
        <Icon name="phone" />
        {t("call96110")}
      </a>
    </section>
  );
}

export default function AssistantSheet() {
  const app = useApp();
  const {
    t,
    settings,
    closeAssistant,
    assistantPrefill,
    personaName,
    persona,
    assistantChat,
    assistantChatId,
    chatHistory,
    updateAssistantChat,
    createAssistantChat,
    selectAssistantChat,
    refreshChatHistory,
    assistantEpoch,
    isAssistantEpochCurrent,
  } = app;
  const { messages, sessionId, undone, busy } = assistantChat;
  const [input, setInput] = useState(assistantPrefill);
  const [consented, setConsented] = useState(readConsent);
  const [voiceBusy, setVoiceBusy] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [chatQuery, setChatQuery] = useState("");
  const [pendingUndo, setPendingUndo] = useState<{
    turn: Turn;
    stepId: string;
    owner: PersonaId;
    chatId?: string;
    epoch: number;
  } | null>(null);
  const [pendingSendNow, setPendingSendNow] = useState<{
    turn: Turn;
    stepId: string;
    owner: PersonaId;
    chatId?: string;
    epoch: number;
  } | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const lastMessageId = messages[messages.length - 1]?.id;

  // A chat switch is a fresh conversation surface: drop transient UI state so
  // nothing from the previous thread can leak into the new one.
  useEffect(() => {
    setInput(assistantPrefill);
    setPendingUndo(null);
    setPendingSendNow(null);
    setVoiceBusy(false);
  }, [assistantPrefill, persona, assistantChatId]);
  useEffect(() => {
    setHistoryOpen(false);
    setChatQuery("");
  }, [persona]);
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [lastMessageId, busy]);

  // Escape closes the history drawer. Verification (consent, OTP approval,
  // Undo, Send now) is never dismissed from the keyboard.
  useEffect(() => {
    if (!historyOpen) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        if (!consented || pendingUndo || pendingSendNow) return;
        setHistoryOpen(false);
        return;
      }
      if (event.key !== "Tab") return;
      const root = sidebarRef.current;
      if (!root) return;
      const focusable = root.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [historyOpen, consented, pendingUndo, pendingSendNow]);
  useEffect(() => {
    if (!historyOpen) return;
    sidebarRef.current?.querySelector<HTMLElement>("button:not([disabled]), input:not([disabled])")?.focus();
  }, [historyOpen]);

  function addTurn(
    owner: PersonaId,
    epoch: number,
    originChatId: string | undefined,
    turn: Turn,
    userText: string,
    recall?: ChatMessage["recall"],
  ) {
    // The reply always lands in the chat that sent the request, even when the
    // user has already switched away. Undo/confirm results are never dropped.
    updateAssistantChat(
      owner,
      (current) => ({
        ...current,
        sessionId: turn.session_id || current.sessionId,
        messages: [
          ...current.messages,
          { id: `${turn.plan_id}-${current.messages.length}`, role: "assistant", text: turn.reply, turn, recall },
        ],
      }),
      originChatId,
    );
    // Only side effects for the chat still on screen.
    if (!isAssistantEpochCurrent(owner, epoch)) return;
    if (turn.actions?.includes("send_now")) {
      const step = turn.plan_view.find((item) => item.tool === "transfer");
      if (step) setPendingSendNow({ turn, stepId: step.step_id, owner, epoch, chatId: originChatId });
    }
    juryBus.publish(turn, userText);
    app.noteActivity();
    app.refresh().catch(() => undefined);
  }

  async function send(text: string, shown?: string) {
    const clean = text.trim();
    if (!clean || !sessionId || busy || voiceBusy || pendingUndo || pendingSendNow) return;
    const owner = persona;
    const originChatId = assistantChatId || sessionId || undefined;
    const epoch = assistantEpoch(owner);
    setInput("");
    updateAssistantChat(
      owner,
      (current) => ({
        ...current,
        busy: true,
        messages: [...current.messages, { id: `u-${current.messages.length}`, role: "user", text: shown ?? clean }],
      }),
      originChatId,
    );
    try {
      addTurn(owner, epoch, originChatId, await api.chat(clean, sessionId, settings.lang), shown ?? clean);
    } catch (error) {
      updateAssistantChat(
        owner,
        (current) => ({
          ...current,
          messages: [...current.messages, { id: `e-${current.messages.length}`, role: "assistant", text: String(error) }],
        }),
        originChatId,
      );
    } finally {
      // Always clear the origin's busy flag, never some newer active chat.
      updateAssistantChat(owner, (current) => ({ ...current, busy: false }), originChatId);
    }
  }

  async function undo(
    turn: Turn,
    stepId: string,
    owner: PersonaId,
    epoch: number,
    originChatId: string | undefined,
  ) {
    const result = await api.undo(turn.session_id, turn.plan_id, stepId);
    // A closed window is an error reply, not an undo: keep the button honest.
    if (result.status === "done") {
      updateAssistantChat(
        owner,
        (current) => ({
          ...current,
          undone: { ...current.undone, [`${turn.plan_id}:${stepId}`]: true },
        }),
        originChatId,
      );
    }
    const recall = result.actions?.includes("recall")
      ? { sessionId: turn.session_id, planId: turn.plan_id, stepId }
      : undefined;
    addTurn(owner, epoch, originChatId, result, t("undo"), recall);
  }

  async function requestRecall(message: ChatMessage) {
    if (!message.recall) return;
    const owner = persona;
    const originChatId = assistantChatId || sessionId || undefined;
    const epoch = assistantEpoch(owner);
    const { sessionId: sid, planId, stepId } = message.recall;
    const result = await api.recall(sid, planId, stepId, settings.lang);
    updateAssistantChat(
      owner,
      (current) => ({
        ...current,
        undone: { ...current.undone, [`recall:${planId}:${stepId}`]: true },
      }),
      originChatId,
    );
    addTurn(owner, epoch, originChatId, result, t("requestRecall"));
  }

  const cycleTextSize = () => {
    const order = ["standard", "large", "xlarge"] as const;
    const next = order[(order.indexOf(settings.text_size) + 1) % order.length];
    app.saveSettings({ text_size: next }).catch(() => undefined);
  };

  // Switching chats closes the history drawer at every size.
  const afterChatSwitch = () => {
    setHistoryOpen(false);
  };

  const lastIndex = messages.length - 1;
  const approvalEpoch = assistantEpoch(persona);
  const approvalChatId = assistantChatId || sessionId || undefined;
  const zh = settings.lang === "zh";

  // A released transfer is final: never offer Undo or Send now for it again.
  const releasedKeys = new Set<string>();
  for (const message of messages) {
    const turn = message.turn;
    if (!turn) continue;
    if (turn.actions?.includes("send_now_released")) {
      for (const step of turn.plan_view) releasedKeys.add(`${turn.plan_id}:${step.step_id}`);
    }
    for (const step of turn.plan_view) {
      if (step.status === "released") releasedKeys.add(`${turn.plan_id}:${step.step_id}`);
    }
  }

  return (
    <div className="sheet-backdrop assistant-backdrop" role="dialog" aria-modal="true" aria-label={t("assistant")}>
      <div className={`sheet assistant-shell${historyOpen ? " history-open" : ""}`}>
        <aside
          ref={sidebarRef}
          id="chat-history"
          className="assistant-sidebar"
          aria-label={zh ? "对话历史" : "Chat history"}
          tabIndex={-1}
        >
          <div className="assistant-sidebar-head">
            <span className="assistant-sidebar-title">{zh ? "对话" : "Chats"}</span>
            <button
              type="button"
              className="icon-btn assistant-sidebar-close"
              aria-label={zh ? "关闭历史" : "Close history"}
              onClick={() => setHistoryOpen(false)}
            >
              <Icon name="close" size={16} strokeWidth={2.5} />
            </button>
          </div>
          <button
            type="button"
            className="primary-btn assistant-newchat"
            onClick={() => {
              setChatQuery("");
              createAssistantChat().then(() => refreshChatHistory()).catch(() => undefined);
              afterChatSwitch();
            }}
          >
            {zh ? "新对话" : "New chat"}
          </button>
          <form
            className="assistant-search"
            onSubmit={(event) => {
              event.preventDefault();
              refreshChatHistory(chatQuery.trim() || undefined).catch(() => undefined);
            }}
          >
            <input
              type="search"
              className="field"
              value={chatQuery}
              onChange={(event) => setChatQuery(event.target.value)}
              placeholder={zh ? "搜索对话…" : "Search chats…"}
              aria-label={zh ? "搜索对话" : "Search chats"}
            />
            <button type="submit" className="pill-btn">{zh ? "搜索" : "Search"}</button>
          </form>
          {chatHistory.length === 0 ? (
            <div className="small muted assistant-history-empty">{zh ? "还没有对话。" : "No chats yet."}</div>
          ) : (
            <div className="assistant-history-list">
              {chatHistory.map((chat) => {
                const current = chat.id === assistantChatId;
                return (
                  <button
                    key={chat.id}
                    data-chat-id={chat.id}
                    type="button"
                    className="assistant-history-item"
                    aria-current={current ? "true" : undefined}
                    onClick={() => {
                      selectAssistantChat(chat.id).catch(() => undefined);
                      afterChatSwitch();
                    }}
                  >
                    <span className="assistant-history-item-title">
                      {chat.title || (zh ? "未命名对话" : "Untitled chat")}
                    </span>
                    <span className="assistant-history-date">
                      {formatChatDate(chat.updated_at || chat.created_at, zh)}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </aside>

        {historyOpen && (
          <button
            type="button"
            className="assistant-history-backdrop"
            aria-label={zh ? "关闭历史" : "Close history"}
            onClick={() => setHistoryOpen(false)}
          />
        )}

        <div className="assistant-main">
          <header className="assistant-header">
            <span className="assistant-mark" aria-hidden="true">A</span>
            <div className="assistant-heading">
              <h2>{t("assistant")}</h2>
              <div className="small muted">{t("aiAssisted")}</div>
            </div>
            <div className="assistant-actions">
              <button
                type="button"
                className="pill-btn"
                aria-expanded={historyOpen}
                aria-controls="chat-history"
                onClick={() => {
                  const next = !historyOpen;
                  setHistoryOpen(next);
                  if (next) refreshChatHistory().catch(() => undefined);
                }}
              >
                {zh ? "历史" : "History"}
              </button>
              <button type="button" className="pill-btn" onClick={cycleTextSize} aria-label={t("textSize")} style={{ fontWeight: 700 }}>
                Aa
              </button>
              <button type="button" className="icon-btn" onClick={closeAssistant} aria-label={t("close")}>
                <Icon name="close" size={16} strokeWidth={2.5} />
              </button>
            </div>
          </header>

        <div ref={listRef} className="assistant-messages">
          {assistantChat.hasMore && (
            <button type="button" className="pill-btn" onClick={() => app.loadEarlierAssistantMessages().catch(() => undefined)}>
              {zh ? "加载更早的消息" : "Load earlier messages"}
            </button>
          )}
          {messages.length === 0 && (
            <>
              <div className="bubble-ai assistant-greeting">
                {t("hello", { name: personaName })}
              </div>
              <div className="assistant-empty-label">{t("tryAsking")}</div>
              <div className="assistant-suggestions">
                {SUGGESTIONS[settings.lang].slice(0, 4).map((s) => (
                  <button key={s} type="button" className="pill-btn" onClick={() => send(s)}>{s}</button>
                ))}
              </div>
              <p className="small muted assistant-empty-note">{t("reassure")}</p>
            </>
          )}

          {messages.map((message, index) => {
            if (message.role === "user") {
              return <div key={message.id} className="bubble-user">{message.text}</div>;
            }
            const turn = message.turn;
            if (turn?.status === "scam") {
              return <ScamAlert key={message.id} reply={turn.reply} />;
            }
            const latest = index === lastIndex;
            const clarify = turn?.status === "clarify" ? splitOptions(turn.reply) : null;
            const doneTransfers =
              turn?.status === "done" && !turn.actions?.includes("send_now")
                ? turn.plan_view.filter((s) => s.tool === "transfer" && (s.status === "done" || (s.status === "released" && turn.actions?.includes("send_now_released"))))
                : [];
            return (
              <div key={message.id} style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                <div className="bubble-ai">
                  {clarify ? clarify.text : message.text}
                </div>
                {clarify && clarify.options.length > 0 && latest && (
                  <div role="group" aria-label={t("chooseOne")} style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                    {clarify.options.map((option) => (
                      <button key={option.n} type="button" className="secondary-btn" style={{ textAlign: "left", minHeight: "2.75rem" }} onClick={() => send(option.n, option.label)}>
                        {option.label}
                      </button>
                    ))}
                  </div>
                )}
                {turn?.status === "needs_approval" && latest && turn.approvals.length > 0 && (
                  <ApprovalCard
                    turn={turn}
                    onResult={(next) => addTurn(persona, approvalEpoch, approvalChatId, next, t("confirm"))}
                  />
                )}
                {message.recall && (
                  <button
                    type="button"
                    className="pill-btn"
                    style={{ alignSelf: "flex-start" }}
                    disabled={undone[`recall:${message.recall.planId}:${message.recall.stepId}`]}
                    onClick={() => requestRecall(message)}
                  >
                    {undone[`recall:${message.recall.planId}:${message.recall.stepId}`] ? t("recallSent") : t("requestRecall")}
                  </button>
                )}
                {doneTransfers.map((step) => {
                  const key = `${turn!.plan_id}:${step.step_id}`;
                  const released = releasedKeys.has(key);
                  const isUndone = Boolean(undone[key]);
                  if (released) {
                    if (!turn!.actions?.includes("send_now_released")) return null;
                    return (
                      <button
                        key={key}
                        type="button"
                        className="pill-btn"
                        disabled
                        style={{ alignSelf: "flex-start", display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
                      >
                        <Icon name="check" size={16} />
                        {zh ? "已转出" : "Sent now"}
                      </button>
                    );
                  }
                  return (
                    <div key={key} style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignSelf: "flex-start", alignItems: "center" }}>
                      <button
                        type="button"
                        className="pill-btn"
                        style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
                        disabled={isUndone || busy || Boolean(pendingUndo) || Boolean(pendingSendNow)}
                        onClick={() => setPendingUndo({
                          turn: turn!,
                          stepId: step.step_id,
                          owner: persona,
                          chatId: assistantChatId || sessionId || undefined,
                          epoch: assistantEpoch(persona),
                        })}
                      >
                        <Icon name="undo" size={16} />
                        {isUndone ? t("undone") : t("undo")}
                      </button>
                      {!isUndone && (
                        <button
                          type="button"
                          className="pill-btn"
                          style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
                          disabled={busy || Boolean(pendingUndo) || Boolean(pendingSendNow)}
                          onClick={() => setPendingSendNow({
                            turn: turn!,
                            stepId: step.step_id,
                            owner: persona,
                            chatId: assistantChatId || sessionId || undefined,
                            epoch: assistantEpoch(persona),
                          })}
                        >
                          <Icon name="send" size={16} />
                          {zh ? "立即转出" : "Send now"}
                        </button>
                      )}
                    </div>
                  );
                })}
                {doneTransfers.length > 0 && (
                  <div className="small muted">
                    {zh
                      ? "立即转出为最终操作，将结束撤销；等待 2 小时仍可撤销。"
                      : "Send now is final and ends Undo. Waiting 2 hours keeps Undo available."}
                  </div>
                )}
              </div>
            );
          })}
          {busy && <div className="bubble-ai muted" aria-live="polite">{t("thinking")}</div>}
        </div>

        <form
          className="assistant-composer"
          onSubmit={(event) => {
            event.preventDefault();
            send(input);
          }}
          style={{ display: "flex", gap: "0.5rem", alignItems: "center", padding: "0.6rem 1.1rem 1rem", borderTop: "1px solid var(--line-soft)" }}
        >
          <input
            aria-label={t("placeholder")}
            placeholder={t("placeholder")}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            maxLength={500}
            disabled={voiceBusy || !sessionId}
            style={{ flex: 1, minWidth: 0, minHeight: "3.1rem", borderRadius: "12px", border: "1px solid #c9d3e3", background: "#fff", padding: "0 1.1rem" }}
          />
          <div title={input.trim() ? (settings.lang === "zh" ? "清空输入框后可录音" : "Clear the message box to record") : undefined} style={{ flexShrink: 0 }}>
            <VoiceInput key={`${persona}:${assistantChatId ?? "none"}:${assistantEpoch(persona)}`} lang={settings.lang}
              disabled={!sessionId || busy || !consented || Boolean(input.trim()) || Boolean(pendingUndo) || Boolean(pendingSendNow)}
              onTranscript={setInput} onBusyChange={setVoiceBusy} />
          </div>
          <button type="submit" className="icon-btn" aria-label={t("send")} disabled={!sessionId || busy || voiceBusy} style={{ width: "3.1rem", height: "3.1rem", background: "var(--primary)", color: "#fff", border: "none" }}>
            <Icon name="send" />
          </button>
        </form>
        <p className="small muted assistant-footnote">
          {settings.lang === "zh" ? "语音最长30秒，发送前请核对姓名和金额。" : "Check names and amounts before sending. Voice clips: up to 30 seconds."}
        </p>
        </div>
      </div>

      {pendingUndo && (
        <UndoConfirm lang={settings.lang} receipt={(() => {
          const card = messages.flatMap(m => m.turn?.plan_id === pendingUndo.turn.plan_id ? m.turn.approvals : [])
            .find(a => a.step_id === pendingUndo.stepId);
          return card?.amount_fen != null
            ? `${new Intl.NumberFormat(settings.lang === "zh" ? "zh-CN" : "en-US", { style: "currency", currency: "CNY" }).format(card.amount_fen / 100)} → ${card.payee_display || ""}`
            : pendingUndo.turn.reply;
        })()} onCancel={() => setPendingUndo(null)} onConfirm={async () => {
          // The result belongs to the chat that opened the dialog, even if the
          // active chat or epoch has changed since.
          await undo(pendingUndo.turn, pendingUndo.stepId, pendingUndo.owner, pendingUndo.epoch, pendingUndo.chatId);
          setPendingUndo(null);
        }} />
      )}

      {pendingSendNow && (
        <SendNowCard
          key={`${pendingSendNow.owner}:${pendingSendNow.chatId ?? "local"}:${pendingSendNow.turn.plan_id}:${pendingSendNow.stepId}`}
          turn={pendingSendNow.turn}
          stepId={pendingSendNow.stepId}
          onResult={(next) => addTurn(
            pendingSendNow.owner,
            pendingSendNow.epoch,
            pendingSendNow.chatId,
            next,
            zh ? "立即转出" : "Send now",
          )}
          onClose={() => setPendingSendNow(null)}
        />
      )}

      {!consented && (
        <div className="sheet-backdrop" style={{ justifyContent: "center", padding: "1.25rem" }}>
          <div className="card" style={{ display: "flex", flexDirection: "column", gap: "0.8rem" }}>
            <h2 style={{ margin: 0, fontSize: "1.15rem" }}>{t("consentTitle")}</h2>
            <p style={{ margin: 0 }}>{t("consentBody")}</p>
            <button
              type="button"
              className="primary-btn"
              onClick={() => {
                try {
                  window.localStorage.setItem(CONSENT_KEY, "yes");
                } catch {
                  /* private mode: ask again next time */
                }
                setConsented(true);
              }}
            >
              {t("consentOk")}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
