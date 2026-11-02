# AnBan security assessment

This is an engineering assessment of a mock-bank competition entry, not a production banking authorization or a legal compliance certification.

| Control | Implemented boundary |
| --- | --- |
| Permissions | Code assigns green/yellow/red tiers, cumulative limits, signed action mandates, MFA and required guardian approval. The planner cannot lower them (`security/policy.py`, `security/mandate.py`). |
| Immediate release | An eligible two-hour held receipt requires a fresh red mandate. It binds the authenticated customer, chat, source plan and step, payee, amount and transfer timing. Request code is explicit. Protected longer holds remain intact (`api/send_now.py`). |
| Settlement | Current fraud, scam and lock protections are checked again. Mandate consumption, selected-transfer release, one ledger row, Undo invalidation and audit entries commit together. Reserved funds are not debited again. |
| Chat ownership | Persistent archives and search are scoped to the authenticated customer. Each chat has independent masked context and financial references. Reload never restores OTPs, vault mappings or pending execution state (`api/chat_store.py`, `api/chat_history.py`). |
| Shared protection | Account freezes, lockouts and investment warnings apply across chats. Incomplete scam evidence can be combined across chats without entering model context; old evidence alone does not renew a freeze. |
| Grounding | Bank results supply balances, prices, amounts and transaction status. A lost response is treated as an uncertain result requiring a history check. |
| Execution boundary | The model emits a closed typed DAG. Arbitrary generated code is not executed. This is not an OS sandbox for running generated programs. |
| Voice | Local transcription produces an editable draft. It never sends a chat, approves an action or moves funds by itself. Switching chats invalidates transient recording state. |
| Audit | Decision and execution events form an HMAC-verified hash chain. Verification codes are excluded from stored chat turns and decision logs. |

Tests: `tests/test_send_now.py`, `tests/test_early_release.py`, plus policy, mandate, graph, history, recent-action, bank and API rollback, replay, hold,
expiry, wrong-code, account-wide scam, responder, ledger, DAG, executor, voice, audit, and browser or browser-to-ASR rehearsal suites.

## Limitations

- The bank, phone and guardian delivery, and face verification are simulated.
  Demo persona login and reset/clock routes are not safe to expose publicly.
- Semantic masking covers recognized bank entities and explicit sensitive
  patterns; it is not complete anonymization of arbitrary free text.
- Model understanding is imperfect. Natural conversation and finance concepts
  are constrained; ambiguous instructions require clarification.
- Context is bounded. The archive is longer than the portion supplied to the
  model, and another chat's financial instructions never authorize this chat.
- Funds released with Send now cannot be undone. Recall is a separate, simulated
  process with no recovery guarantee.
- No live banking, card or subscription provider, production biometrics, external
  staff connection or regulator integration is established by this demo.
- Local database and key compromise and infrastructure hardening remain
  production concerns; an in-database audit anchor alone is not external
  immutable storage.
- This assessment does not replace the full Python suite, frontend build, voice and Undo tests, or live browser checks for a given build.
