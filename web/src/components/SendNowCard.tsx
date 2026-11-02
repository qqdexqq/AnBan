// Modal for releasing a held transfer immediately. This is the final action:
// it ends Undo for the transfer, so the wait option stays next to it.
import { useEffect, useId, useRef, useState } from "react";
import type { FormEvent } from "react";
import { call } from "../api";
import { useApp } from "../state";
import type { Turn } from "../types";
import { yuan } from "./ApprovalCard";
import Icon from "./Icon";

interface PrepareResult {
  challenge_id: string;
  amount_fen: number;
  payee_label: string;
  expires_at: string;
  needs_co_approval: boolean;
}

interface CodeResult {
  sent: boolean;
  demo_otp?: string;
  demo_co_otp?: string;
}

export interface SendNowCardProps {
  turn: Turn;
  stepId: string;
  onResult: (next: Turn) => void;
  onClose: () => void;
}

const STRINGS = {
  en: {
    title: "Send now",
    finalWarning:
      "Send now is final. The held funds are released immediately and Undo ends for this transfer.",
    waitNote: "Wait 2 hours instead to keep Undo available.",
    preparing: "Preparing this transfer…",
    prepareError: "Could not prepare this transfer. Check its current status in transaction history.",
    amount: "Amount",
    payee: "To",
    expires: "Request expires",
    requestCode: "Request phone code",
    requestingCode: "Requesting code…",
    codeSent: "Verification code requested.",
    requestError: "Could not request a phone code. Try again.",
    simulatedSms: "Simulated SMS",
    simulatedNote: "Demo only. The code is shown here, not sent to a real phone.",
    smsCode: "SMS code",
    guardianCode: "Guardian code",
    otp: "Six-digit SMS code",
    retype: "Retype the amount",
    retypeHint: "Enter the amount in yuan, for example 50.00",
    mismatch: "The amount does not match this transfer.",
    face: "Demo face verification",
    faceHint: "Simulated check, the same as the approval card.",
    guardianOtp: "Guardian code",
    confirm: "Send now",
    cancel: "Wait 2 hours (keeps Undo)",
    failure: "Could not confirm the result. Check transaction history before trying again.",
    need: "Complete the checks to continue.",
  },
  zh: {
    title: "立即转出",
    finalWarning: "立即转出为最终操作。冻结的资金将马上划出，此笔转账将不再可以撤销。",
    waitNote: "改为等待 2 小时，仍可以撤销。",
    preparing: "正在准备这笔转账…",
    prepareError: "暂时无法准备这笔转账，请查看交易记录中的最新状态。",
    amount: "金额",
    payee: "收款方",
    expires: "请求有效期至",
    requestCode: "获取手机验证码",
    requestingCode: "正在获取验证码…",
    codeSent: "验证码已请求。",
    requestError: "暂时无法获取手机验证码，请重试。",
    simulatedSms: "模拟短信",
    simulatedNote: "仅用于演示。验证码显示在这里，不会发送到真实手机。",
    smsCode: "短信验证码",
    guardianCode: "监护人验证码",
    otp: "六位短信验证码",
    retype: "重新输入金额",
    retypeHint: "请输入以元为单位的金额，例如 50.00",
    mismatch: "金额与这笔转账不一致。",
    face: "模拟人脸核验",
    faceHint: "模拟核验，与确认卡片一致。",
    guardianOtp: "监护人验证码",
    confirm: "立即转出",
    cancel: "等待 2 小时（保留撤销）",
    failure: "暂时无法确认结果，请先查看交易记录，再决定是否重试。",
    need: "请完成各项核对后继续。",
  },
} as const;

function parseFen(value: string): number {
  const clean = value.replace(/[,¥\s]/g, "");
  if (!/^\d+(\.\d{0,2})?$/.test(clean)) return Number.NaN;
  return Math.round(Number(clean) * 100);
}

export default function SendNowCard({ turn, stepId, onResult, onClose }: SendNowCardProps) {
  const { settings } = useApp();
  const text = STRINGS[settings.lang];
  const titleId = useId();
  const descId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const preparedRef = useRef(false);
  const mounted = useRef(true);
  const busyRef = useRef(false);
  const inFlight = useRef(false);
  const requestingRef = useRef(false);

  const [challenge, setChallenge] = useState<PrepareResult | null>(null);
  const [prepareError, setPrepareError] = useState<string | null>(null);
  const [requesting, setRequesting] = useState(false);
  const [codeSent, setCodeSent] = useState(false);
  const [sms, setSms] = useState<CodeResult | null>(null);
  const [otp, setOtp] = useState("");
  const [retype, setRetype] = useState("");
  const [face, setFace] = useState(false);
  const [coOtp, setCoOtp] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    mounted.current = true;
    if (!preparedRef.current) {
      preparedRef.current = true;
      call<PrepareResult>("/send-now/prepare", {
        method: "POST",
        body: JSON.stringify({
          session_id: turn.session_id,
          plan_id: turn.plan_id,
          step_id: stepId,
        }),
      })
        .then((result) => {
          if (!mounted.current) return;
          setChallenge(result);
        })
        .catch(() => {
          if (!mounted.current) return;
          setPrepareError(text.prepareError);
        });
    }
    cancelRef.current?.focus();
    return () => {
      mounted.current = false;
    };
    // Prepared once per mounted receipt; the language can change without refetching.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stepId, turn.plan_id, turn.session_id]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.stopPropagation();
        if (!busyRef.current) onClose();
      }
      if (event.key === "Tab") {
        const focusable = Array.from(
          dialogRef.current?.querySelectorAll<HTMLElement>(
            "button:not(:disabled), input:not(:disabled)",
          ) || [],
        );
        if (!focusable.length) return;
        event.preventDefault();
        const index = focusable.indexOf(document.activeElement as HTMLElement);
        focusable[(index + (event.shiftKey ? -1 : 1) + focusable.length) % focusable.length].focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const retypedFen = parseFen(retype);
  const amountMatches =
    challenge !== null && Number.isFinite(retypedFen) && retypedFen === challenge.amount_fen;
  const ready =
    challenge !== null &&
    otp.length === 6 &&
    amountMatches &&
    face &&
    (!challenge.needs_co_approval || coOtp.trim().length > 0);

  async function requestCode() {
    if (!challenge || requestingRef.current) return;
    requestingRef.current = true;
    setRequesting(true);
    setError(null);
    try {
      const result = await call<CodeResult>("/send-now/request-code", {
        method: "POST",
        body: JSON.stringify({ challenge_id: challenge.challenge_id }),
      });
      if (!mounted.current) return;
      // The simulated card is only shown when the server returns a demo code.
      setSms(result.demo_otp ? result : null);
      setCodeSent(result.sent);
    } catch {
      if (mounted.current) setError(text.requestError);
    } finally {
      requestingRef.current = false;
      if (mounted.current) setRequesting(false);
    }
  }

  async function confirm(event: FormEvent) {
    event.preventDefault();
    if (!challenge || !ready || inFlight.current || busyRef.current) return;
    inFlight.current = true;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      const next = await call<Turn>("/send-now/confirm", {
        method: "POST",
        body: JSON.stringify({
          challenge_id: challenge.challenge_id,
          otp: otp.trim(),
          retyped_amount_fen: retypedFen,
          face_ok: face,
          co_otp: challenge.needs_co_approval ? coOtp.trim() : "",
        }),
      });
      onResult(next);
      if (mounted.current) onClose();
    } catch {
      if (!mounted.current) return;
      inFlight.current = false;
      busyRef.current = false;
      setBusy(false);
      // A lost response is not proof of failure; the bank remains authoritative.
      setError(text.failure);
    }
  }

  return (
    <div
      ref={dialogRef}
      className="sheet-backdrop"
      role="alertdialog"
      aria-modal="true"
      aria-busy={busy}
      aria-labelledby={titleId}
      aria-describedby={descId}
      style={{ justifyContent: "center", alignItems: "center", padding: "1.25rem" }}
    >
      <div
        className="card"
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "0.8rem",
          width: "100%",
          maxWidth: "27rem",
          maxHeight: "90vh",
          overflowY: "auto",
        }}
      >
        <h2 id={titleId} style={{ margin: 0, fontSize: "1.2rem" }}>
          {text.title}
        </h2>
        <div id={descId} style={{ display: "flex", flexDirection: "column", gap: "0.55rem" }}>
          <p style={{ margin: 0, fontWeight: 700, color: "var(--danger-dark)" }}>{text.finalWarning}</p>
          <p className="small muted" style={{ margin: 0 }}>{text.waitNote}</p>
        </div>

        {!challenge && !prepareError && (
          <div className="small muted" aria-live="polite">{text.preparing}</div>
        )}
        {prepareError && (
          <div role="alert" className="small" style={{ color: "var(--danger-dark)" }}>
            {prepareError}
          </div>
        )}

        {challenge && (
          <>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
              <div className="small muted">{text.payee}</div>
              <div style={{ fontWeight: 700, overflowWrap: "anywhere" }}>{challenge.payee_label}</div>
              <div className="small muted">{text.amount}</div>
              <div className="tabular" style={{ fontSize: "1.6rem", fontWeight: 800 }}>
                {yuan(challenge.amount_fen)}
              </div>
              {challenge.expires_at && (
                <div className="small muted">
                  {text.expires}: {formatExpiry(challenge.expires_at, settings.lang === "zh")}
                </div>
              )}
            </div>

            <button
              type="button"
              className="secondary-btn"
              disabled={requesting || busy}
              onClick={() => void requestCode()}
            >
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}>
                <Icon name="phone" size={17} />
                {requesting ? text.requestingCode : text.requestCode}
              </span>
            </button>
            {codeSent && !sms && (
              <div className="small muted" role="status">{text.codeSent}</div>
            )}
            {sms && sms.demo_otp && (
              <div
                className="card"
                role="status"
                style={{ background: "var(--primary-soft)", border: "1px solid var(--primary)", padding: "0.65rem 0.8rem" }}
              >
                <div className="small" style={{ fontWeight: 800 }}>{text.simulatedSms}</div>
                <div className="small muted" style={{ marginBottom: "0.35rem" }}>{text.simulatedNote}</div>
                <div className="tabular">
                  {text.smsCode}: <strong>{sms.demo_otp}</strong>
                </div>
                {sms.demo_co_otp && (
                  <div className="tabular">
                    {text.guardianCode}: <strong>{sms.demo_co_otp}</strong>
                  </div>
                )}
              </div>
            )}

            <form onSubmit={(event) => void confirm(event)} style={{ display: "flex", flexDirection: "column", gap: "0.65rem" }}>
              <label className="small" style={{ display: "flex", flexDirection: "column", gap: "0.3rem", fontWeight: 600 }}>
                {text.otp}
                <input
                  className="field tabular"
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  maxLength={6}
                  value={otp}
                  onChange={(event) => setOtp(event.target.value.replace(/\D/g, "").slice(0, 6))}
                  disabled={busy}
                />
              </label>

              <label className="small" style={{ display: "flex", flexDirection: "column", gap: "0.3rem", fontWeight: 600 }}>
                {text.retype}
                <input
                  className="field tabular"
                  inputMode="decimal"
                  aria-label={text.retype}
                  value={retype}
                  onChange={(event) => setRetype(event.target.value)}
                  disabled={busy}
                  aria-describedby={retype && !amountMatches ? `${titleId}-mismatch` : undefined}
                />
                <span className="small muted" style={{ fontWeight: 400 }}>{text.retypeHint}</span>
              </label>
              {retype && !amountMatches && (
                <div id={`${titleId}-mismatch`} className="small" style={{ color: "var(--danger-dark)" }}>
                  {text.mismatch}
                </div>
              )}

              <label className="small" style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontWeight: 600 }}>
                <input
                  type="checkbox"
                  checked={face}
                  onChange={(event) => setFace(event.target.checked)}
                  disabled={busy}
                />
                {text.face}
              </label>
              <span className="small muted" style={{ marginTop: "-0.35rem" }}>{text.faceHint}</span>

              {challenge.needs_co_approval && (
                <label className="small" style={{ display: "flex", flexDirection: "column", gap: "0.3rem", fontWeight: 600 }}>
                  {text.guardianOtp}
                  <input
                    className="field tabular"
                    inputMode="numeric"
                    value={coOtp}
                    onChange={(event) => setCoOtp(event.target.value.replace(/\D/g, "").slice(0, 6))}
                    disabled={busy}
                  />
                </label>
              )}

              {error && (
                <div role="alert" className="small" style={{ color: "var(--danger-dark)" }}>
                  {error}
                </div>
              )}

              <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
                <button
                  type="submit"
                  className="primary-btn"
                  style={{ flex: 1, minWidth: "8rem" }}
                  disabled={busy || !ready}
                >
                  {text.confirm}
                </button>
              </div>
              {!ready && !busy && <div className="small muted" style={{ textAlign: "center" }}>{text.need}</div>}
            </form>
          </>
        )}

        <button ref={cancelRef} type="button" className="secondary-btn" disabled={busy} onClick={onClose}>
          {text.cancel}
        </button>
      </div>
    </div>
  );
}

function formatExpiry(value: string, zh: boolean): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(zh ? "zh-CN" : "en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
