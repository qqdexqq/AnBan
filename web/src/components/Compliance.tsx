// Compliance center: the Chinese rules implemented in code (or simulated for
// the demo) plus the simulated regulator-side report queue. Strings that name
// regulations stay in their original language; UI chrome goes through t().
import { useEffect, useState } from "react";
import Icon from "./Icon";
import { api } from "../api";
import { useApp } from "../state";
import type { ComplianceReport } from "../types";

type RuleBadge = "code" | "sim";

interface Rule {
  id: string;
  zhTitle: string;
  enTitle: string;
  regulation: string;
  badges: RuleBadge[];
}

const RULES: Rule[] = [
  {
    id: "pboc-261",
    zhTitle: "延时到账（24 小时）与撤销；确认前显示收款人姓名",
    enTitle: "Delayed arrival (24 h) and undo; shows the payee's name before confirm",
    regulation: "PBOC 银发〔2016〕261号",
    badges: ["code"],
  },
  {
    id: "anti-fraud-law",
    zhTitle: "诈骗识别、24 小时冻结、96110",
    enTitle: "Scam detection, 24 h freeze, 96110",
    regulation: "反电信网络诈骗法 (2022)",
    badges: ["code"],
  },
  {
    id: "fraud-list",
    zhTitle: "涉诈账户名单核查（本地名单）",
    enTitle: "Fraud-account list check (local list)",
    regulation: "国家反诈中心",
    badges: ["sim"],
  },
  {
    id: "anti-fraud-report",
    zhTitle: "每次诈骗冻结都向国家反诈中心上报",
    enTitle: "Report to the National Anti-Fraud Center on every scam freeze",
    regulation: "国家反诈中心",
    badges: ["sim"],
  },
  {
    id: "large-suspicious",
    zhTitle: "每日 50 万元大额交易报告与可疑交易报告，从不向客户展示",
    enTitle: "¥500,000/day large-transaction report and suspicious-transaction report, never shown to the customer",
    regulation: "《金融机构大额交易和可疑交易报告管理办法》",
    badges: ["code", "sim"],
  },
  {
    id: "ai-label",
    zhTitle: "每条助手消息都带 AI 生成标识",
    enTitle: "AI-generated label on every assistant message",
    regulation: "《人工智能生成合成内容标识办法》(2025-09-01)",
    badges: ["code"],
  },
  {
    id: "pipl",
    zhTitle: "AI 只看到占位符；首次使用前先征得同意",
    enTitle: "AI sees placeholders only; consent before first use",
    regulation: "《个人信息保护法》 + JR/T 0171-2020",
    badges: ["code"],
  },
  {
    id: "elder-design",
    zhTitle: "字号可调、适老模式、无广告",
    enTitle: "Adjustable text size, elder mode, no ads",
    regulation: "工信部 App 适老化通用设计规范 (2021)",
    badges: ["code"],
  },
];

const REPORT_KINDS: Record<string, string> = {
  fraud_list_hit: "Fraud-list hit / 反诈名单命中",
  anti_fraud_center_report: "国家反诈中心 report",
  suspicious_transaction_report: "Suspicious transaction report (STR)",
  large_transaction_report: "Large transaction report (LTR)",
};

function prettyKind(kind: string): string {
  return REPORT_KINDS[kind] ?? kind;
}

function clockTime(createdAt: string): string {
  const date = new Date(createdAt);
  if (Number.isNaN(date.getTime())) return createdAt;
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  return `${hours}:${minutes}`;
}

export default function Compliance() {
  const { t, setScreen, settings, activity } = useApp();
  const [reports, setReports] = useState<ComplianceReport[] | null>(null);
  const zh = settings.lang === "zh";

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
  }, [activity]);

  return (
    <div className="screen" style={{ padding: "1rem 1.25rem 2rem" }}>
      <header style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
        <button className="icon-btn" type="button" aria-label={t("back")} onClick={() => setScreen("settings")}>
          <Icon name="back" />
        </button>
        <h1 style={{ flex: 1, margin: 0, fontSize: "1.3rem" }}>{t("compliance")}</h1>
      </header>
      <p className="muted" style={{ margin: "0.25rem 0 1rem" }}>
        {t("complianceSub")}
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
        {RULES.map((rule) => (
          <article key={rule.id} className="card" style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.5rem" }}>
              <div style={{ fontWeight: 700 }}>{zh ? rule.zhTitle : rule.enTitle}</div>
              <div style={{ display: "flex", flexShrink: 0, gap: "0.3rem" }}>
                {rule.badges.map((badge) =>
                  badge === "code" ? (
                    <span key={badge} className="badge badge-green">
                      {t("realRule")}
                    </span>
                  ) : (
                    <span key={badge} className="badge badge-sim">
                      {t("simulated")}
                    </span>
                  ),
                )}
              </div>
            </div>
            <div className="small muted">{rule.regulation}</div>
          </article>
        ))}
      </div>

      <section className="card" style={{ marginTop: "1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        <h2 style={{ margin: 0, fontSize: "1rem" }}>{t("reportsTitle")}</h2>
        {reports !== null && reports.length === 0 && <div className="small muted">{t("noReports")}</div>}
        {reports?.map((report, index) => (
          <div
            key={`${report.kind}-${report.created_at}-${index}`}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "0.5rem",
              paddingTop: "0.45rem",
              borderTop: "1px solid var(--line-soft)",
            }}
          >
            <div style={{ minWidth: 0, fontSize: "0.9rem", fontWeight: 600 }}>{prettyKind(report.kind)}</div>
            <div style={{ display: "flex", alignItems: "center", flexShrink: 0, gap: "0.4rem" }}>
              <span className="small muted tabular">{clockTime(report.created_at)}</span>
              <span className="badge badge-sim">{t("simulated")}</span>
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
