// Thin client for the AnBan demo API. Vite proxies /api to the backend on :8000.
import type {
  ComplianceStatus,
  Lang,
  Settings,
  Summary,
  Turn,
  Txn,
} from "./types";

let token: string | null = null;
let loginGeneration = 0;

export async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init.headers ?? {}),
    },
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }
  return (await response.json()) as T;
}

export const PERSONAS = [
  { id: "young_pro", name: "Chen Hao", nameZh: "陈浩" },
  { id: "retiree", name: "Zhao Guilan", nameZh: "赵桂兰" },
  { id: "micro_business", name: "Huang Lei", nameZh: "黄磊" },
  { id: "expat", name: "Daniel", nameZh: "Daniel" },
] as const;

export type PersonaId = (typeof PERSONAS)[number]["id"];

/** One named assistant conversation stored on the server. */
export interface ChatMeta {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

/** A stored message row returned by GET /chats/{id}/messages. */
export interface ChatHistoryMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  turn?: Turn;
  created_at?: string;
}

export interface ChatMessagesPage {
  messages: ChatHistoryMessage[];
  has_more: boolean;
  oldest_id: string | null;
}

export async function login(persona: PersonaId): Promise<void> {
  const generation = ++loginGeneration;
  const result = await call<{ token: string }>("/auth/demo-login", {
    method: "POST",
    body: JSON.stringify({ persona }),
  });
  if (generation === loginGeneration) token = result.token;
}

/** Upload a recorded clip as a raw blob (no JSON) for server-side transcription. */
export async function transcribeAudio(
  blob: Blob,
  signal?: AbortSignal,
): Promise<{ text: string; language: string; model: string }> {
  const response = await fetch("/api/voice/transcribe", {
    method: "POST",
    headers: {
      "Content-Type": blob.type || "application/octet-stream",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: blob,
    signal,
  });
  if (!response.ok) throw await transcribeError(response);
  return (await response.json()) as { text: string; language: string; model: string };
}

/** Turn a backend error body into a minimal Error, keeping a safe code if given. */
async function transcribeError(response: Response): Promise<Error & { code?: string }> {
  let code: string | undefined;
  let message = `voice request failed (${response.status})`;
  try {
    const body = (await response.json()) as Record<string, unknown>;
    const detail = body.detail ?? body.error;
    if (typeof detail === "string" && detail.trim()) {
      message = detail.trim();
    } else if (detail && typeof detail === "object") {
      const nested = detail as Record<string, unknown>;
      if (typeof nested.code === "string") code = nested.code;
      if (typeof nested.message === "string" && nested.message.trim()) message = nested.message.trim();
    }
    if (!code && typeof body.code === "string") code = body.code;
  } catch {
    // Non-JSON body: keep only the status message.
  }
  const error: Error & { code?: string } = new Error(message);
  if (code) error.code = code;
  return error;
}

export const api = {
  summary: () => call<Summary>("/me/summary"),
  transactions: (limit = 6) =>
    call<{ transactions: Txn[] }>(`/me/transactions?limit=${limit}`).then((r) => r.transactions),
  settings: () => call<Settings>("/me/settings"),
  saveSettings: (changes: Partial<Settings>) =>
    call<Settings>("/me/settings", { method: "PUT", body: JSON.stringify(changes) }),
  listChats: (query?: string) =>
    call<{ chats: ChatMeta[] }>(`/chats${query ? `?query=${encodeURIComponent(query)}` : ""}`).then(
      (r) => r.chats,
    ),
  createChat: (title?: string) =>
    call<{ chat: ChatMeta }>("/chats", {
      method: "POST",
      body: JSON.stringify(title ? { title } : {}),
    }).then((r) => r.chat),
  chatMessages: (id: string, before?: string) =>
    call<ChatMessagesPage>(`/chats/${encodeURIComponent(id)}/messages${before ? `?before=${encodeURIComponent(before)}` : ""}`),
  chat: (text: string, sessionId: string | null, lang: Lang) =>
    call<Turn>("/chat", {
      method: "POST",
      body: JSON.stringify({ text, session_id: sessionId, lang }),
    }),
  approve: (sessionId: string, approvals: Record<string, unknown>) =>
    call<Turn>(`/chat/${sessionId}/approve`, {
      method: "POST",
      body: JSON.stringify({ approvals }),
    }),
  undo: (sessionId: string, planId: string, stepId: string) =>
    call<Turn>("/undo", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, plan_id: planId, step_id: stepId }),
    }),
  recall: (sessionId: string, planId: string, stepId: string, lang: Lang) =>
    call<Turn>("/recall", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, plan_id: planId, step_id: stepId, lang }),
    }),
  audit: (lang: Lang) =>
    call<{ lines: string[]; chain_ok: boolean }>(`/me/audit?lang=${lang}`),
  compliance: () => call<ComplianceStatus>("/compliance/status"),
  advanceClock: (minutes: number) =>
    call<{ now: string }>("/demo/clock", {
      method: "POST",
      body: JSON.stringify({ advance_minutes: minutes }),
    }),
  demoClock: (minutes: number) =>
    call<{ now: string }>("/demo/clock", {
      method: "POST",
      body: JSON.stringify({ advance_minutes: minutes }),
    }),
  reset: () => call<{ ok: boolean }>("/demo/reset", { method: "POST" }),
};
