// App-wide state: who is logged in, settings (language, text size, elder
// mode…), balance and recent activity, and which screen is shown.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { api, login, PERSONAS } from "./api";
import type { ChatMeta, PersonaId } from "./api";
import { yuan } from "./components/ApprovalCard";
import { makeT } from "./i18n";
import type { TFn } from "./i18n";
import type { AppNotification, ChatMessage, Settings, Summary, Txn } from "./types";

type NotificationDraft = Omit<AppNotification, "id">;

function notification(
  titleEn: string,
  titleZh: string,
  textEn: string,
  textZh: string,
): NotificationDraft {
  return {
    title: { en: titleEn, zh: titleZh },
    text: { en: textEn, zh: textZh },
  };
}

const SKIP_REASON: Record<string, { en: string; zh: string }> = {
  over_daily_limit: { en: "daily limit reached", zh: "已达当日限额" },
  insufficient_funds: { en: "balance too low", zh: "余额不足" },
  payments_locked: { en: "payments are paused", zh: "付款已暂停" },
  scam_freeze: { en: "scam protection is on", zh: "防诈保护已开启" },
};

const SKIP_FALLBACK = { en: "check your standing orders", zh: "请查看定时转账" };

/** Local calendar date, so "today" matches the day the user is looking at. */
function todayIso(): string {
  const now = new Date();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${month}-${day}`;
}

/**
 * Turn two summary snapshots into banners. The first snapshot after login only
 * seeds the baseline, except reminders that are already due.
 */
function collectNotifications(
  previous: Summary | null,
  next: Summary,
  notifiedReminders: Set<string>,
): NotificationDraft[] {
  const drafts: NotificationDraft[] = [];

  if (previous) {
    const previousOrders = new Map(
      (previous.standing_orders ?? []).map((order) => [order.scheduled_id, order]),
    );
    for (const order of next.standing_orders ?? []) {
      const run = order.last_run;
      if (!run) continue;
      const before = previousOrders.get(order.scheduled_id)?.last_run ?? null;
      if (
        before &&
        before.occurrence === run.occurrence &&
        before.status === run.status &&
        before.reason === run.reason &&
        before.amount_fen === run.amount_fen
      ) {
        continue;
      }

      if (run.status === "paid") {
        const amount = yuan(run.amount_fen ?? order.amount_fen);
        drafts.push(
          notification(
            "Standing order sent",
            "定时转账已转出",
            `${amount} to ${order.payee}`,
            `${amount} 转给 ${order.payee}`,
          ),
        );
      } else if (run.status === "skipped") {
        const reason = SKIP_REASON[run.reason ?? ""] ?? SKIP_FALLBACK;
        drafts.push(
          notification(
            "Standing order skipped",
            "定时转账未执行",
            reason.en,
            reason.zh,
          ),
        );
      }
    }

    if ((next.anomalies ?? 0) > (previous.anomalies ?? 0)) {
      drafts.push(
        notification(
          "Unusual activity",
          "异常交易提醒",
          "Please review your recent transactions.",
          "请查看最近的交易。",
        ),
      );
    }

    const previousCards = new Map(
      (previous.cards ?? []).map((card) => [card.last4, card.status.toLowerCase()]),
    );
    for (const card of next.cards ?? []) {
      const status = card.status.toLowerCase();
      if (status !== "frozen" && status !== "lost") continue;
      if (previousCards.get(card.last4) === status) continue;
      drafts.push(
        status === "lost"
          ? notification("Card reported lost", "卡片已挂失", `···${card.last4}`, `···${card.last4}`)
          : notification("Card frozen", "卡片已冻结", `···${card.last4}`, `···${card.last4}`),
      );
    }
  }

  // Prefer the injected demo clock so reminders fire on the app's "today",
  // not the browser's local date.
  const today = next.now ? next.now.slice(0, 10) : todayIso();
  for (const reminder of next.reminders ?? []) {
    if (!reminder.due_on || reminder.due_on > today) continue;
    const key = `${reminder.due_on}|${reminder.note}`;
    if (notifiedReminders.has(key)) continue;
    notifiedReminders.add(key);
    const note = reminder.note.trim();
    drafts.push(
      notification(
        "Reminder",
        "提醒",
        note || "renewal reminder",
        note || "续费提醒",
      ),
    );
  }

  return drafts;
}

export type Screen = "home" | "settings" | "compliance";

/** Chat state that survives closing and reopening the assistant sheet. */
export interface AssistantChatState {
  hasMore?: boolean;
  oldestId?: string | null;
  messages: ChatMessage[];
  sessionId: string | null;
  undone: Record<string, boolean>;
  busy: boolean;
}

const EMPTY_ASSISTANT_CHAT: AssistantChatState = {
  messages: [],
  sessionId: null,
  undone: {},
  busy: false,
};

/** Named conversations for one persona. Threads are kept by server chat id. */
interface PersonaChats {
  activeId: string | null;
  chats: Record<string, AssistantChatState>;
}

function isPersonaChats(value: unknown): value is PersonaChats {
  return typeof value === "object" && value !== null && "chats" in value && "activeId" in value;
}

function isAssistantChatState(value: unknown): value is AssistantChatState {
  return typeof value === "object" && value !== null && "messages" in value && "sessionId" in value;
}

/**
 * Accept the previous single-chat-per-persona shape and keep it as a thread, so
 * an in-memory chat is never discarded when this state structure changes.
 */
function asPersonaChats(value: unknown, owner: PersonaId): PersonaChats {
  if (isPersonaChats(value)) return value;
  if (isAssistantChatState(value)) {
    const id = value.sessionId ?? `local:${owner}`;
    return { activeId: id, chats: { [id]: value } };
  }
  return { activeId: null, chats: {} };
}

const DEFAULT_SETTINGS: Settings = {
  lang: "en",
  text_size: "standard",
  elder_mode: false,
  arrival_delay: "2h",
  guardian: "",
};

interface AppState {
  ready: boolean;
  persona: PersonaId;
  personaName: string;
  settings: Settings;
  t: TFn;
  summary: Summary | null;
  transactions: Txn[];
  screen: Screen;
  assistantOpen: boolean;
  assistantPrefill: string;
  /** In-memory chat for the active thread of the current persona. Never persisted. */
  assistantChat: AssistantChatState;
  /** Id of the active named chat, or null before one has been chosen/created. */
  assistantChatId: string | null;
  /** Named chats for the current persona (server metadata, newest first). */
  chatHistory: ChatMeta[];
  updateAssistantChat: (
    persona: PersonaId,
    update: (current: AssistantChatState) => AssistantChatState,
    chatId?: string,
  ) => void;
  createAssistantChat: () => Promise<void>;
  selectAssistantChat: (id: string) => Promise<void>;
  refreshChatHistory: (query?: string) => Promise<void>;
  loadEarlierAssistantMessages: () => Promise<void>;
  /** Bumped on persona switch and chat selection so old callbacks are ignored. */
  assistantEpoch: (persona: PersonaId) => number;
  isAssistantEpochCurrent: (persona: PersonaId, epoch: number) => boolean;
  notifications: AppNotification[];
  pushNotification: (notification: NotificationDraft) => void;
  dismissNotification: (id: string) => void;
  juryPanel: boolean;
  /** Bumped after anything that may change the audit log or reports. */
  activity: number;
  setScreen: (screen: Screen) => void;
  openAssistant: (prefill?: string) => void;
  closeAssistant: () => void;
  saveSettings: (changes: Partial<Settings>) => Promise<void>;
  switchPersona: (persona: PersonaId) => Promise<void>;
  refresh: () => Promise<void>;
  setJuryPanel: (on: boolean) => void;
  noteActivity: () => void;
}

const AppContext = createContext<AppState | null>(null);

export function useApp(): AppState {
  const value = useContext(AppContext);
  if (!value) throw new Error("useApp outside AppProvider");
  return value;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [persona, setPersona] = useState<PersonaId>("young_pro");
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [transactions, setTransactions] = useState<Txn[]>([]);
  const [screen, setScreen] = useState<Screen>("home");
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [assistantPrefill, setAssistantPrefill] = useState("");
  const [assistantChats, setAssistantChats] = useState<Partial<Record<PersonaId, PersonaChats>>>({});
  const [chatHistoryByPersona, setChatHistoryByPersona] = useState<Partial<Record<PersonaId, ChatMeta[]>>>({});
  const chatsRef = useRef<Partial<Record<PersonaId, PersonaChats>>>({});
  const personaRef = useRef<PersonaId>("young_pro");
  const initialLoginStarted = useRef(false);
  const personaSwitchGen = useRef(0);
  const refreshGen = useRef(0);
  const assistantEpochs = useRef<Partial<Record<PersonaId, number>>>({});
  const selectionGen = useRef(0);
  const historyLoading = useRef<Partial<Record<PersonaId, boolean>>>({});
  const [juryPanel, setJuryPanel] = useState(() => window.innerWidth > 1000);
  const [activity, setActivity] = useState(0);
  const [notifications, setNotifications] = useState<AppNotification[]>([]);
  const notificationSeq = useRef(0);
  const previousSummary = useRef<Summary | null>(null);
  const notifiedReminders = useRef<Set<string>>(new Set());

  const pushNotification = useCallback((draft: NotificationDraft) => {
    const item: AppNotification = { ...draft, id: `n-${++notificationSeq.current}` };
    setNotifications((current) => [...current, item].slice(-3));
  }, []);

  const dismissNotification = useCallback((id: string) => {
    setNotifications((current) => current.filter((item) => item.id !== id));
  }, []);

  const refresh = useCallback(async () => {
    const owner = personaRef.current;
    const generation = personaSwitchGen.current;
    const request = ++refreshGen.current;
    const [nextSummary, nextTxns] = await Promise.all([api.summary(), api.transactions(6)]);
    if (owner !== personaRef.current || generation !== personaSwitchGen.current || request !== refreshGen.current) return;
    setSummary(nextSummary);
    setTransactions(nextTxns);
    const previous = previousSummary.current;
    previousSummary.current = nextSummary;
    for (const draft of collectNotifications(previous, nextSummary, notifiedReminders.current)) {
      pushNotification(draft);
    }
  }, [pushNotification]);

  const commitChats = useCallback((next: Partial<Record<PersonaId, PersonaChats>>) => {
    chatsRef.current = next;
    setAssistantChats(next);
  }, []);

  const mutateChats = useCallback(
    (owner: PersonaId, update: (current: PersonaChats) => PersonaChats) => {
      const current = asPersonaChats(chatsRef.current[owner], owner);
      const next = update(current);
      if (next === current) return;
      commitChats({ ...chatsRef.current, [owner]: next });
    },
    [commitChats],
  );

  const bumpEpoch = useCallback((owner: PersonaId) => {
    assistantEpochs.current[owner] = (assistantEpochs.current[owner] ?? 0) + 1;
  }, []);

  const updateAssistantChat = useCallback(
    (
      owner: PersonaId,
      update: (current: AssistantChatState) => AssistantChatState,
      chatId?: string,
    ) => {
      const store = asPersonaChats(chatsRef.current[owner], owner);
      const target = chatId ?? store.activeId ?? `local:${owner}`;
      const activate = chatId === undefined && !store.activeId;
      const existing = store.chats[target] ?? {
        ...EMPTY_ASSISTANT_CHAT,
        sessionId: chatId ?? (activate ? null : store.activeId),
      };
      const next = update(existing);
      if (next === existing && !activate) return;
      // An explicit chatId always lands in that thread, even when it is no
      // longer active, so late financial results are never routed elsewhere.
      mutateChats(owner, (current) => ({
        activeId: activate ? target : current.activeId,
        chats: { ...current.chats, [target]: next },
      }));
    },
    [mutateChats],
  );

  const assistantEpoch = useCallback(
    (owner: PersonaId) => assistantEpochs.current[owner] ?? 0,
    [],
  );

  const isAssistantEpochCurrent = useCallback(
    (owner: PersonaId, epoch: number) => (assistantEpochs.current[owner] ?? 0) === epoch,
    [],
  );

  const createAssistantChat = useCallback(async () => {
    const owner = personaRef.current;
    const generation = ++selectionGen.current;
    const chat = await api.createChat();
    // Ignore a response for a persona the user has already left.
    if (personaRef.current !== owner || generation !== selectionGen.current) return;
    bumpEpoch(owner);
    mutateChats(owner, (store) => ({
      activeId: chat.id,
      // The minted server id is the session id from the very first message.
      chats: { ...store.chats, [chat.id]: { ...EMPTY_ASSISTANT_CHAT, sessionId: chat.id } },
    }));
    setChatHistoryByPersona((current) => ({
      ...current,
      [owner]: [chat, ...(current[owner] ?? []).filter((item) => item.id !== chat.id)],
    }));
  }, [bumpEpoch, mutateChats]);

  const selectAssistantChat = useCallback(
    async (id: string) => {
      const owner = personaRef.current;
      const generation = ++selectionGen.current;
      bumpEpoch(owner);
      mutateChats(owner, (store) => ({
        activeId: id,
        chats: store.chats[id]
          ? store.chats
          : { ...store.chats, [id]: { ...EMPTY_ASSISTANT_CHAT, sessionId: id } },
      }));
      try {
        const page = await api.chatMessages(id);
        // A quick A-then-B click must never let a late A response win.
        if (generation !== selectionGen.current || personaRef.current !== owner) return;
        mutateChats(owner, (store) => {
          const existing = store.chats[id];
          // Never mix histories or overwrite live replies; only fill fresh threads.
          if (existing && (existing.messages.length > 0 || existing.busy)) return store;
          return {
            ...store,
            chats: {
              ...store.chats,
              [id]: {
                ...(existing ?? EMPTY_ASSISTANT_CHAT),
                sessionId: existing?.sessionId ?? id,
                messages: page.messages.map((message) => ({
                  id: message.id,
                  role: message.role,
                  text: message.text,
                  ...(message.turn ? { turn: message.turn } : {}),
                })),
                hasMore: page.has_more,
                oldestId: page.oldest_id,
              },
            },
          };
        });
      } catch (error) {
        // History unavailable (older demo API): keep the in-memory chat.
        console.error(error);
      }
    },
    [bumpEpoch, mutateChats],
  );

  const refreshChatHistory = useCallback(async (query?: string) => {
    const owner = personaRef.current;
    try {
      const chats = await api.listChats(query);
      if (personaRef.current !== owner) return;
      setChatHistoryByPersona((current) => ({ ...current, [owner]: chats }));
    } catch (error) {
      // History unavailable: keep whatever is loaded in memory.
      console.error(error);
    }
  }, []);

  const loadEarlierAssistantMessages = useCallback(async () => {
    const owner = personaRef.current;
    const store = chatsRef.current[owner];
    const id = store?.activeId;
    const cursor = id && store?.chats[id]?.oldestId;
    if (!id || !cursor) return;
    const page = await api.chatMessages(id, cursor);
    updateAssistantChat(owner, (current) => {
      if (current.oldestId !== cursor) return current;
      const existing = new Set(current.messages.map((m) => m.id));
      return { ...current, hasMore: page.has_more, oldestId: page.oldest_id,
        messages: [...page.messages.filter((m) => !existing.has(m.id)).map((m) => ({
          id: m.id, role: m.role, text: m.text, ...(m.turn ? { turn: m.turn } : {}),
        })), ...current.messages] };
    }, id);
  }, [updateAssistantChat]);

  const ensureChats = useCallback(
    async (owner: PersonaId) => {
      if (historyLoading.current[owner]) return;
      historyLoading.current[owner] = true;
      try {
        const chats = await api.listChats();
        if (personaRef.current !== owner) return;
        setChatHistoryByPersona((current) => ({ ...current, [owner]: chats }));
        if (asPersonaChats(chatsRef.current[owner], owner).activeId) return;
        if (chats.length > 0) {
          const newest = chats.reduce((best, chat) =>
            chat.updated_at > best.updated_at ? chat : best,
          );
          await selectAssistantChat(newest.id);
        } else {
          await createAssistantChat();
        }
      } catch (error) {
        // Prior demo API without /chats: do not wipe the current chat.
        console.error(error);
      } finally {
        historyLoading.current[owner] = false;
      }
    },
    [createAssistantChat, selectAssistantChat],
  );

  const switchPersona = useCallback(
    async (next: PersonaId) => {
      const generation = ++personaSwitchGen.current;
      // Invalidate in-flight assistant callbacks, chat selections and stop
      // showing "thinking" for requests whose chat was abandoned.
      selectionGen.current += 1;
      for (const p of PERSONAS) bumpEpoch(p.id);
      const cleared: Partial<Record<PersonaId, PersonaChats>> = { ...chatsRef.current };
      let changed = false;
      for (const p of PERSONAS) {
        const store = asPersonaChats(chatsRef.current[p.id], p.id);
        const chats: Record<string, AssistantChatState> = {};
        let clearedBusy = false;
        for (const [id, chat] of Object.entries(store.chats)) {
          if (chat.busy) {
            chats[id] = { ...chat, busy: false };
            clearedBusy = true;
          } else {
            chats[id] = chat;
          }
        }
        if (clearedBusy) {
          cleared[p.id] = { activeId: store.activeId, chats };
          changed = true;
        }
      }
      if (changed) commitChats(cleared);
      setReady(false);
      // A new login starts a fresh baseline; only due reminders may fire now.
      previousSummary.current = null;
      notifiedReminders.current = new Set();
      setNotifications([]);
      await login(next);
      if (generation !== personaSwitchGen.current) return;
      personaRef.current = next;
      setPersona(next);
      const stored = await api.settings();
      if (generation !== personaSwitchGen.current) return;
      setSettings(stored);
      await refresh();
      if (generation !== personaSwitchGen.current) return;
      setAssistantOpen(false);
      setScreen("home");
      setActivity((n) => n + 1);
      setReady(true);
    },
    [bumpEpoch, commitChats, refresh],
  );

  // Load the persona's named chats, creating or selecting one only if none is
  // active yet. Kept in an effect so no network work happens during render.
  useEffect(() => {
    if (!ready) return;
    ensureChats(persona).catch(() => undefined);
  }, [ensureChats, persona, ready]);

  // On first open, retry if the earlier history load was unavailable.
  useEffect(() => {
    if (!ready || !assistantOpen) return;
    ensureChats(personaRef.current).catch(() => undefined);
  }, [assistantOpen, ensureChats, ready]);

  // Keep the drawer list fresh after activity without touching loaded messages.
  useEffect(() => {
    if (!ready) return;
    const owner = personaRef.current;
    if (!chatsRef.current[owner]?.activeId) return;
    refreshChatHistory().catch(() => undefined);
  }, [activity, ready, refreshChatHistory]);

  useEffect(() => {
    if (initialLoginStarted.current) return;
    initialLoginStarted.current = true;
    switchPersona("young_pro").catch((error) => console.error(error));
  }, [switchPersona]);

  // Text size and elder mode scale the whole UI through CSS variables.
  useEffect(() => {
    document.documentElement.dataset.text = settings.text_size;
    document.documentElement.dataset.elder = String(settings.elder_mode);
    document.documentElement.lang = settings.lang === "zh" ? "zh-CN" : "en";
  }, [settings]);

  const saveSettings = useCallback(async (changes: Partial<Settings>) => {
    setSettings((current) => ({ ...current, ...changes }));
    const stored = await api.saveSettings(changes);
    setSettings(stored);
  }, []);

  // Poll for real account events (standing orders, cards, anomalies, reminders).
  // Chat replies and approvals also refresh immediately through refresh().
  useEffect(() => {
    if (!ready) return;
    const timer = window.setInterval(() => {
      refresh().catch(() => undefined);
    }, 15000);
    return () => window.clearInterval(timer);
  }, [ready, refresh]);

  const value = useMemo<AppState>(() => {
    const meta = PERSONAS.find((p) => p.id === persona) ?? PERSONAS[0];
    const active = asPersonaChats(assistantChats[persona], persona);
    const activeChat = active.activeId ? active.chats[active.activeId] : undefined;
    return {
      ready,
      persona,
      personaName: settings.lang === "zh" ? meta.nameZh : meta.name,
      settings,
      t: makeT(settings.lang),
      summary,
      transactions,
      screen,
      assistantOpen,
      assistantPrefill,
      assistantChat: activeChat ?? EMPTY_ASSISTANT_CHAT,
      assistantChatId: active.activeId,
      chatHistory: chatHistoryByPersona[persona] ?? [],
      updateAssistantChat,
      createAssistantChat,
      selectAssistantChat,
      refreshChatHistory,
      loadEarlierAssistantMessages,
      assistantEpoch,
      isAssistantEpochCurrent,
      notifications,
      pushNotification,
      dismissNotification,
      juryPanel,
      activity,
      setScreen,
      openAssistant: (prefill = "") => {
        setAssistantPrefill(prefill);
        setAssistantOpen(true);
      },
      closeAssistant: () => setAssistantOpen(false),
      saveSettings,
      switchPersona,
      refresh,
      setJuryPanel,
      noteActivity: () => setActivity((n) => n + 1),
    };
  }, [ready, persona, settings, summary, transactions, screen, assistantOpen, assistantPrefill, assistantChats, chatHistoryByPersona, updateAssistantChat, createAssistantChat, selectAssistantChat, refreshChatHistory, loadEarlierAssistantMessages, assistantEpoch, isAssistantEpochCurrent, notifications, pushNotification, dismissNotification, juryPanel, activity, saveSettings, switchPersona, refresh]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
