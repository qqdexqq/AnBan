// Explicit Yes/No confirmation before an undo. It never calls the API itself:
// the parent wires onConfirm to the real undo request and closes on success.
import { useEffect, useId, useRef, useState } from "react";

export interface UndoConfirmProps {
  lang?: "en" | "zh";
  receipt: string;
  onConfirm: () => Promise<void>;
  onCancel: () => void;
}

const STRINGS = {
  en: {
    title: "Undo this transfer?",
    receiptLabel: "Transfer receipt",
    warning:
      "Cancelling a transfer returns the funds to your account only while it is still eligible. A transfer that has already been processed may not be stopped.",
    no: "No, keep transfer",
    yes: "Yes, undo transfer",
  },
  zh: {
    title: "确定撤销这笔转账？",
    receiptLabel: "转账回执",
    warning:
      "只有在转账仍符合条件时，撤销才会将资金退回您的账户。已经处理的转账可能无法撤销。",
    no: "不，保留转账",
    yes: "是，撤销转账",
  },
} as const;

const ERROR =
  "We couldn't confirm the undo result. Check the latest transfer status before retrying. / 暂时无法确认撤销结果，请先查看转账最新状态再重试。";

export default function UndoConfirm({ lang = "en", receipt, onConfirm, onCancel }: UndoConfirmProps) {
  const text = STRINGS[lang];
  const titleId = useId();
  const descId = useId();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busyRef = useRef(false);
  const inFlight = useRef(false);
  const disposed = useRef(false);
  const noRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    disposed.current = false;
    noRef.current?.focus();
    return () => {
      disposed.current = true;
    };
  }, []);

  function cancel() {
    if (busyRef.current) return;
    onCancel();
  }

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") { event.stopPropagation(); cancel(); }
      if (event.key === "Tab") {
        const buttons = Array.from(dialogRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") || []);
        event.preventDefault();
        if (buttons.length) {
          const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
          buttons[(index + (event.shiftKey ? -1 : 1) + buttons.length) % buttons.length].focus();
        }
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onCancel]);

  async function confirm() {
    if (inFlight.current) return;
    inFlight.current = true;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      await onConfirm();
      // Success: the parent closes the dialog. Do not touch state in case it
      // has already unmounted.
    } catch {
      if (disposed.current) return;
      inFlight.current = false;
      busyRef.current = false;
      setBusy(false);
      setError(ERROR);
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
      <div className="card" style={{ display: "flex", flexDirection: "column", gap: "0.85rem", width: "100%", maxWidth: "26rem" }}>
        <h2 id={titleId} style={{ margin: 0, fontSize: "1.2rem" }}>
          {text.title}
        </h2>
        <div id={descId} style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          <div>
            <div className="small muted">{text.receiptLabel}</div>
            <p style={{ margin: 0, whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{receipt}</p>
          </div>
          <p className="small muted" style={{ margin: 0 }}>
            {text.warning}
          </p>
        </div>
        {error && (
          <div role="alert" className="small" style={{ color: "var(--danger-dark)" }}>
            {error}
          </div>
        )}
        <div style={{ display: "flex", gap: "0.6rem" }}>
          <button ref={noRef} type="button" className="secondary-btn" style={{ flex: 1 }} disabled={busy} onClick={cancel}>
            {text.no}
          </button>
          <button type="button" className="primary-btn" style={{ flex: 1 }} disabled={busy} onClick={() => void confirm()}>
            {text.yes}
          </button>
        </div>
      </div>
    </div>
  );
}
