// Settings screen: language, text size, elder mode, payment protections,
// what the AI can see, and demo-only controls. Every visible string is t().
import { useCallback, useEffect, useRef, useState } from "react";
import Icon from "./Icon";
import { api, PERSONAS } from "../api";
import { useApp } from "../state";
import type { ArrivalDelay, Lang, Settings, TextSize } from "../types";

const LANGUAGES: { value: Lang; label: string }[] = [
  { value: "en", label: "English" },
  { value: "zh", label: "中文" },
];

const TEXT_SIZES: { value: TextSize; labelKey: "textSizeStd" | "textSizeLarge" | "textSizeXL" }[] = [
  { value: "standard", labelKey: "textSizeStd" },
  { value: "large", labelKey: "textSizeLarge" },
  { value: "xlarge", labelKey: "textSizeXL" },
];

function PendingNote({
  settings,
  field,
  t,
}: {
  settings: Settings;
  field: string;
  t: (key: "pendingChange") => string;
}) {
  const item = settings.pending?.find((p) => p.key === field);
  if (!item) return null;
  const when = new Date(item.effective_at).toLocaleString(settings.lang === "zh" ? "zh-CN" : "en-GB", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  return (
    <div className="small" style={{ color: "var(--warning)", marginTop: "0.25rem" }}>
      {t("pendingChange")} {when}
    </div>
  );
}

const ARRIVALS: { value: ArrivalDelay; labelKey: "arrivalStd" | "arrival24" }[] = [
  { value: "2h", labelKey: "arrivalStd" },
  { value: "24h", labelKey: "arrival24" },
];

export default function SettingsScreen() {
  const {
    settings,
    saveSettings,
    t,
    setScreen,
    switchPersona,
    persona,
    refresh,
    juryPanel,
    setJuryPanel,
    noteActivity,
  } = useApp();
  const [guardianDraft, setGuardianDraft] = useState(settings.guardian);
  const [aiPreviewOpen, setAiPreviewOpen] = useState(false);
  const [savedVisible, setSavedVisible] = useState(false);
  const savedTimer = useRef<number | null>(null);

  useEffect(() => {
    setGuardianDraft(settings.guardian);
  }, [settings.guardian]);

  useEffect(
    () => () => {
      if (savedTimer.current !== null) window.clearTimeout(savedTimer.current);
    },
    [],
  );

  const flashSaved = useCallback(() => {
    setSavedVisible(true);
    if (savedTimer.current !== null) window.clearTimeout(savedTimer.current);
    savedTimer.current = window.setTimeout(() => setSavedVisible(false), 2000);
  }, []);

  const persist = useCallback(
    async (changes: Partial<Settings>) => {
      await saveSettings(changes);
      flashSaved();
    },
    [saveSettings, flashSaved],
  );

  const saveGuardian = useCallback(async () => {
    const next = guardianDraft.trim().slice(0, 40);
    if (next === settings.guardian) return;
    await persist({ guardian: next });
  }, [guardianDraft, settings.guardian, persist]);

  const advance = useCallback(
    async (minutes: number) => {
      await api.advanceClock(minutes);
      await refresh();
      noteActivity();
    },
    [refresh, noteActivity],
  );

  const resetDemo = useCallback(async () => {
    await api.reset();
    await switchPersona(persona);
  }, [switchPersona, persona]);

  return (
    <div className="screen" style={{ padding: "1rem 1.25rem 2rem" }}>
      <header style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
        <button className="icon-btn" type="button" aria-label={t("back")} onClick={() => setScreen("home")}>
          <Icon name="back" />
        </button>
        <h1 style={{ flex: 1, margin: 0, fontSize: "1.3rem" }}>{t("settings")}</h1>
        <span
          className="small muted"
          role="status"
          aria-live="polite"
          style={{ minWidth: "3.5rem", textAlign: "right" }}
        >
          {savedVisible ? t("saved") : ""}
        </span>
      </header>

      <section className="settings-group">
        <h2>{t("general")}</h2>
        <div className="settings-card">
          <div className="settings-row">
            <span>{t("language")}</span>
            <div className="segmented" role="group" aria-label={t("language")}>
              {LANGUAGES.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  aria-pressed={settings.lang === option.value}
                  onClick={() => {
                    void persist({ lang: option.value });
                  }}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div className="settings-row" style={{ flexDirection: "column", alignItems: "stretch", gap: "0.4rem" }}>
            <span>{t("textSize")}</span>
            <div className="segmented" role="group" aria-label={t("textSize")} style={{ flexWrap: "wrap", width: "100%" }}>
              {TEXT_SIZES.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  aria-pressed={settings.text_size === option.value}
                  style={{ flex: "1 1 auto" }}
                  onClick={() => {
                    void persist({ text_size: option.value });
                  }}
                >
                  {t(option.labelKey)}
                </button>
              ))}
            </div>
          </div>

          <div className="settings-row">
            <div style={{ minWidth: 0 }}>
              <div>{t("elderMode")}</div>
              <div className="small muted">{t("elderModeHint")}</div>
              <PendingNote settings={settings} field="elder_mode" t={t} />
            </div>
            <button
              className="switch"
              type="button"
              role="switch"
              aria-checked={settings.elder_mode}
              aria-label={t("elderMode")}
              onClick={() => {
                void persist({ elder_mode: !settings.elder_mode });
              }}
            />
          </div>
        </div>
      </section>

      <section className="settings-group">
        <h2>{t("payments")}</h2>
        <div className="settings-card">
          <div className="settings-row" style={{ flexDirection: "column", alignItems: "stretch", gap: "0.4rem" }}>
            <span>{t("arrival")}</span>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
              {ARRIVALS.map((option) => {
                const selected = settings.arrival_delay === option.value;
                return (
                  <button
                    key={option.value}
                    className="secondary-btn"
                    type="button"
                    aria-pressed={selected}
                    style={{
                      width: "100%",
                      display: "flex",
                      alignItems: "center",
                      textAlign: "left",
                      justifyContent: "flex-start",
                      fontWeight: selected ? 700 : 600,
                      borderColor: selected ? "var(--primary)" : "var(--line)",
                      background: selected ? "var(--primary-soft)" : "var(--surface)",
                    }}
                    onClick={() => {
                      void persist({ arrival_delay: option.value });
                    }}
                  >
                    {t(option.labelKey)}
                  </button>
                );
              })}
            </div>
            <span className="small muted">{t("arrivalHint")}</span>
            <PendingNote settings={settings} field="arrival_delay" t={t} />
          </div>

          <div className="settings-row">
            <span>{t("dailyLimit")}</span>
            <span style={{ fontWeight: 700 }}>{settings.elder_mode ? "¥500" : "¥1,000"}</span>
          </div>

          <div className="settings-row" style={{ flexDirection: "column", alignItems: "stretch", gap: "0.35rem" }}>
            <label htmlFor="guardian-input" style={{ fontWeight: 600 }}>
              {t("guardian")}
            </label>
            <input
              id="guardian-input"
              className="field"
              type="text"
              maxLength={40}
              value={guardianDraft}
              onChange={(event) => setGuardianDraft(event.target.value.slice(0, 40))}
              onBlur={() => {
                void saveGuardian();
              }}
            />
            <span className="small muted">{t("guardianHint")}</span>
          </div>
        </div>
      </section>

      <section className="settings-group">
        <h2>{t("privacy")}</h2>
        <div className="settings-card">
          <div className="settings-row">
            <button
              type="button"
              aria-expanded={aiPreviewOpen}
              aria-controls="ai-preview"
              onClick={() => setAiPreviewOpen((open) => !open)}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "0.5rem",
                width: "100%",
                padding: 0,
                border: "none",
                background: "transparent",
                color: "inherit",
                font: "inherit",
                textAlign: "left",
              }}
            >
              <span>{t("whatAiSees")}</span>
              <span
                aria-hidden="true"
                style={{
                  display: "inline-flex",
                  transform: aiPreviewOpen ? "rotate(90deg)" : "none",
                  transition: "transform 0.15s",
                }}
              >
                <Icon name="chevron" />
              </span>
            </button>
          </div>
          {aiPreviewOpen && (
            <div
              id="ai-preview"
              className="settings-row"
              style={{ flexDirection: "column", alignItems: "flex-start", gap: "0.35rem" }}
            >
              <code className="mono small">给&lt;PAYEE_1&gt;转&lt;AMT_1&gt;</code>
              <code className="mono small">Send &lt;PAYEE_1&gt; &lt;AMT_1&gt;</code>
            </div>
          )}

          <div className="settings-row">
            <div style={{ minWidth: 0 }}>
              <div>{t("aiDisclosure")}</div>
              <div className="small muted">{t("aiDisclosureHint")}</div>
            </div>
            <button
              className="switch"
              type="button"
              role="switch"
              aria-checked={true}
              aria-label={t("aiDisclosure")}
              disabled
            />
          </div>

          <div className="settings-row">
            <button
              type="button"
              onClick={() => setScreen("compliance")}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "0.5rem",
                width: "100%",
                padding: 0,
                border: "none",
                background: "transparent",
                color: "inherit",
                font: "inherit",
                textAlign: "left",
              }}
            >
              <span>{t("compliance")}</span>
              <Icon name="chevron" />
            </button>
          </div>
        </div>
      </section>

      <section className="settings-group">
        <h2>{t("demo")}</h2>
        <div className="settings-card">
          <div className="settings-row">
            <label htmlFor="persona-select">{t("persona")}</label>
            <select
              id="persona-select"
              className="field"
              value={persona}
              onChange={(event) => {
                const next = PERSONAS.find((option) => option.id === event.target.value);
                if (next) void switchPersona(next.id);
              }}
            >
              {PERSONAS.map((option) => (
                <option key={option.id} value={option.id}>
                  {settings.lang === "zh" ? option.nameZh : option.name}
                </option>
              ))}
            </select>
          </div>

          <div className="settings-row" style={{ flexWrap: "wrap", justifyContent: "flex-end" }}>
            <button
              className="pill-btn"
              type="button"
              onClick={() => {
                void advance(120);
              }}
            >
              {t("clockPlus2")}
            </button>
            <button
              className="pill-btn"
              type="button"
              onClick={() => {
                void advance(1440);
              }}
            >
              {t("clockPlus24")}
            </button>
          </div>

          <div className="settings-row">
            <button className="secondary-btn" type="button" style={{ width: "100%" }} onClick={() => void resetDemo()}>
              {t("resetDemo")}
            </button>
          </div>

          <div className="settings-row">
            <span>{t("juryPanel")}</span>
            <button
              className="switch"
              type="button"
              role="switch"
              aria-checked={juryPanel}
              aria-label={t("juryPanel")}
              onClick={() => setJuryPanel(!juryPanel)}
            />
          </div>
        </div>
      </section>
    </div>
  );
}
