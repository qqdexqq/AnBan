// iOS-style notification banners fed only by real account events (state.tsx
// diffs each /me/summary snapshot and pushes the changes here).
import { useCallback, useEffect, useRef, useState } from "react";
import { useApp } from "../state";
import type { AppNotification, Lang } from "../types";

const AUTO_DISMISS_MS = 5000;
const LEAVE_MS = 250;

function Banner({
  item,
  lang,
  onDismiss,
}: {
  item: AppNotification;
  lang: Lang;
  onDismiss: (id: string) => void;
}) {
  const [leaving, setLeaving] = useState(false);
  const dismissed = useRef(false);
  const leaveTimer = useRef<number | null>(null);

  const beginDismiss = useCallback(() => {
    if (dismissed.current) return;
    dismissed.current = true;
    setLeaving(true);
    leaveTimer.current = window.setTimeout(() => onDismiss(item.id), LEAVE_MS);
  }, [item.id, onDismiss]);

  useEffect(() => {
    const timer = window.setTimeout(beginDismiss, AUTO_DISMISS_MS);
    return () => {
      window.clearTimeout(timer);
      if (leaveTimer.current !== null) window.clearTimeout(leaveTimer.current);
    };
  }, [beginDismiss]);

  return (
    <div
      className={`notif-banner${leaving ? " leaving" : ""}`}
      role="button"
      tabIndex={0}
      onClick={beginDismiss}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          beginDismiss();
        }
      }}
    >
      <div className="notif-icon" aria-hidden="true" />
      <div className="notif-body">
        <div className="notif-app">
          <span>ANBAN</span>
          <span className="notif-time">{lang === "zh" ? "现在" : "now"}</span>
        </div>
        <div className="notif-title">{item.title[lang]}</div>
        <div className="notif-text">{item.text[lang]}</div>
      </div>
    </div>
  );
}

export default function Notifications() {
  const { notifications, dismissNotification, settings } = useApp();
  if (notifications.length === 0) return null;

  return (
    <div className="notif-stack" aria-live="polite">
      {notifications.slice(-3).map((item) => (
        <Banner
          key={item.id}
          item={item}
          lang={settings.lang}
          onDismiss={dismissNotification}
        />
      ))}
    </div>
  );
}
