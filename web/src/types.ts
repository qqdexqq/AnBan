// Shapes returned by the AnBan API (api/app.py, agent/graph.py TurnResult).

export type Lang = "en" | "zh";
export type TextSize = "standard" | "large" | "xlarge";
export type ArrivalDelay = "2h" | "24h";
export type Tier = "green" | "yellow" | "red";

export type TurnStatus =
  | "done"
  | "needs_approval"
  | "clarify"
  | "refused"
  | "scam"
  | "scam_warning"
  | "frozen"
  | "vetoed"
  | "paused"
  | "error";

export interface Approval {
  step_id: string;
  tool: string;
  tier: Tier;
  amount_fen: number | null;
  payee_label: string | null;
  payee_display: string | null;
  target_display: string | null;
  day_of_month: number | null;
  reasons: { message_zh: string; message_en: string }[];
  needs_mfa: boolean;
  needs_co_approval: boolean;
  demo_otp: string | null;
  demo_co_otp: string | null;
  error: string | null;
}

export interface PlanStep {
  step_id: string;
  tool: string;
  tier: Tier;
  status: string;
  after: string[];
}

export interface Turn {
  status: TurnStatus;
  reply: string;
  approvals: Approval[];
  plan_view: PlanStep[];
  session_id: string;
  plan_id: string;
  ai_generated?: boolean;
  actions?: string[];
}

export interface StandingOrderRun {
  occurrence: string;
  status: string;
  reason: string | null;
  amount_fen: number | null;
}

export interface StandingOrder {
  scheduled_id: string;
  payee: string;
  amount_fen: number;
  day_of_month: number;
  status: string;
  note: string;
  next_due: string | null;
  last_run: StandingOrderRun | null;
}

export interface Reminder {
  due_on: string;
  note: string;
}

export interface Summary {
  balance_fen: number;
  display: string;
  cards: { last4: string; status: string }[];
  subscriptions: { merchant: string; amount_fen: number; unused_60d: boolean }[];
  anomalies: number;
  standing_orders?: StandingOrder[];
  reminders?: Reminder[];
  // Injected demo clock (ISO), used as the reminder "today" when present.
  now?: string;
}

export interface AppNotification {
  id: string;
  title: { en: string; zh: string };
  text: { en: string; zh: string };
}

export interface Txn {
  id: string;
  ts: string;
  amount_fen: number;
  display: string;
  counterparty: string;
  counterparty_en: string;
  category: string;
  // Transfers still on hold (undoable until this time).
  held_until?: string;
  hold_reason?: string | null;
}

export interface Settings {
  lang: Lang;
  text_size: TextSize;
  elder_mode: boolean;
  arrival_delay: ArrivalDelay;
  guardian: string;
  // Lowered protections wait 24 h before they apply.
  pending?: { key: string; value: unknown; effective_at: string }[];
}

export interface ComplianceReport {
  kind: string;
  created_at: string;
  payload: Record<string, unknown>;
  simulated: boolean;
}

export interface ComplianceStatus {
  reports: ComplianceReport[];
  fraud_list_size: number;
  large_transfer_threshold_fen: number;
  settings: Settings;
}

/** One chat bubble in the assistant. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  turn?: Turn;
  // Set on a closed-window undo reply: the transfer a recall request is for.
  recall?: { sessionId: string; planId: string; stepId: string };
}
