# AnBan · Chinese regulations it follows

AnBan is a competition demo, not a licensed bank. It **implements the rules in code**. Where a rule
needs a government system that only licensed banks can reach, AnBan uses a clearly labelled
**SIMULATED** stand-in. Every simulated record carries `simulated = 1`.

| Regulation | What it requires | How AnBan does it | Real / simulated | Code · test |
|---|---|---|---|---|
| PBOC 银发〔2016〕261号 (+2019 adjustment) | Customer can choose delayed arrival; payee shown before confirming; transfer can be withdrawn within the delay | Settings › Transfer arrival: standard (undo within 2 h) or 24 h delayed; the approval card shows the payee name and ···last4; undo until release | Real | `security/cooling.py` `cooling_for(arrival_delay=)`, `security/compliance.py` · `tests/test_compliance.py::test_chosen_24h_arrival_delays_the_transfer_and_it_can_be_undone` |
| 《反电信网络诈骗法》(2022) | Banks monitor fraud risk, warn, delay or suspend payments; 96110 | Deterministic scam detector before the AI; 24 h customer-wide freeze; 96110 button; coached-secrecy, safe-account, OTP-request scripts | Real | `security/scam.py`, `agent/graph.py` `_intake` · `tests/test_scam.py`, eval attacks A01–A22 |
| 国家反诈中心 fraud-account list | No payment to an account on the national list | Policy rule `R_FRAUD_LIST` vetoes transfers and standing orders to listed accounts (numbers normalised: spaces, dashes, full-width) | **Simulated list** (`compliance/fraud_accounts.json`) | `security/policy.py` · `test_payment_to_a_fraud_listed_account_is_blocked_and_reported`, eval A41 |
| 国家反诈中心 reporting | Report fraud cases | Every scam freeze files a report record | **Simulated** | `security/compliance.py` `report_scam_freeze` · `test_scam_freeze_is_reported_to_the_simulated_anti_fraud_centre` |
| 《金融机构大额交易和可疑交易报告管理办法》(in force 2017-07-01) | Natural person: domestic transfers ≥ ¥500,000 a day → large-transaction report; suspicious → STR; no tipping off | One LTR per customer per day at ¥500,000; STR on every scam freeze; never mentioned in a customer reply; the records view exists only in demo mode | Real rule, **simulated** regulator queue | `check_large_transactions` · `test_large_transaction_report_is_filed_but_never_shown`, `test_regulator_records_are_not_a_customer_route` |
| 《人工智能生成合成内容标识办法》(2025-09-01) + GB 45438-2025 | Explicit label on AI-generated content; keep logs | Every assistant message shows "AI-generated / AI 生成" (cannot be switched off); API replies carry `ai_generated: true`; hash-chained audit log | Real | `api/app.py` `as_dict` · `test_chat_replies_are_labelled_ai_generated` |
| 《个人信息保护法》 + JR/T 0171-2020 | Consent, data minimisation, protection of financial data | The AI only sees tokens such as `<PAYEE_1>`, `<AMT_1>`; one-time consent before first use; Settings shows what the AI sees | Real | `security/tokens.py` · `tests/test_tokens.py`, `tests/test_graph.py` (no PII to the planner) |
| 工信部《移动互联网应用(App)适老化通用设计规范》(2021) | Adjustable text size, line spacing ≥ 1.3, no ads, colour not the only signal | Text size standard/large/extra large; elder mode (larger text, simpler home, ¥500 daily limit, family co-approval, 24 h red cooling); risk badges carry text | Real | `web/src/styles.css`, `security/policy.py` `_is_elder` · `test_elder_mode_turns_on_elder_protections` |

## Known limitations

- **Loosening protections takes effect at once** (C3): turning elder mode off or switching back to the
  2 h arrival works immediately, also during a scam freeze, and is not audited. A bank would delay
  loosening by 24 h and refuse it during a freeze.
- **C4 fixed in the October 30 candidate:** delayed release re-reads the fraud list and checks
  active scam/security holds inside its release transaction. Malformed list data fails closed.
  Overdue held transfers remain visible and can still be cancelled before release.
- **Requesting money from a listed account** is not blocked (C6).
- **Guardian** is free text and not yet linked to the co-approval contact (C7).
- Not implemented: account tiers (I/II/III 类账户), real-name verification, 算法备案 (algorithm filing),
  cross-border reporting.
- October 30 scope: password changes use secure self-service instructions; human-support
  requests are audited but there is no live staff connection. Reminders are persisted locally
  without notification delivery. Card-control settings do not establish a complete card-network
  authorization simulator. Scheduled-transfer records still require execution-path closure.
- Semantic masking covers known bank entities and explicit sensitive-value patterns; it is
  not a general guarantee that arbitrary free text contains no previously unknown personal data.
