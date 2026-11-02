# AnBan core contracts (v1, Oct 19)

Shared formats between tokenizer, planner (LLM), DAG check, policy, mandates and executor.
The planner's training data uses exactly these formats, so changing them later means regenerating data.

## 1. Semantic tokens

The LLM never sees names, phones, account numbers, exact balances, exact past amounts or real dates.

Token label format: `<KIND_n>` where KIND ∈

| KIND | Refers to | Rendered for the LLM (examples) |
|---|---|---|
| `PAYEE` | a contact | `<PAYEE_1 \| spouse \| weekly \| last: 100-500 \| verified>` |
| `ACCT` | own account | `<ACCT_A \| current>` (letters, not numbers) |
| `POT` | own savings pot | `<POT_1 \| birthday>` |
| `CARD` | own card | `<CARD_1 \| physical \| active>` |
| `SUB` | subscription | `<SUB_1 \| entertainment \| <100/mo \| unused 60d+>` |
| `PROD` | wealth product | `<PROD_1 \| risk 2 \| rate 2-3%>` |
| `MERCHANT` | order merchant (florist, cake) | `<MERCHANT_1 \| florist>` |
| `AMT` | an amount the USER stated | `<AMT_1 \| 100-500>` |
| `DATE` | a date the user stated or a contact date | `<DATE_1 \| in 12 days>` |
| `RAW` | account/card/phone number typed in text or found in untrusted data | `<RAW_1 \| unverified number>` |

Tag vocabulary (v1, minimal):
- relation: self, spouse, parent, child, sibling, friend, colleague, landlord, merchant, other
- frequency: new, rare, monthly, weekly (from last 90 days of payments)
- amount bucket: `<100`, `100-500`, `500-1000`, `1000-5000`, `5000+` (1,000 = tier line, always an edge)
- trust: verified, new_payee, flagged
- dates: relative only (`today`, `in N days`, `N days ago`)
- balance: NEVER rendered. Only via `can_cover` tool / `can_cover(<AMT_n>): yes|no` tag on ACCT.

Rules:
- Tokens are minted only by code (`security/tokens.py`), bound to `(session_id, plan_id)`. Text cannot create one.
- Every token has an `origin`: `user_text` | `bank_context` | `untrusted` (memos, merchant names in incoming txns, reader output).
- `RAW` tokens can never be a payee. Money-moving args only accept PAYEE/ACCT/POT/AMT tokens.
- A write step whose payee or amount token has origin `untrusted` is vetoed by policy.

## 2. Planner output (one JSON object)

```json
{"kind": "plan", "steps": [
  {"id": "s1", "tool": "get_contact_date", "args": {"payee": "<PAYEE_1>", "what": "birthday"}, "after": []},
  {"id": "s2", "tool": "can_cover", "args": {"amount": "<AMT_1>"}, "after": []},
  {"id": "s3", "tool": "move_to_pot", "args": {"pot_name": "birthday", "amount": "<AMT_1>"}, "after": ["s2"]},
  {"id": "s4", "tool": "merchant_order", "args": {"merchant": "<MERCHANT_1>", "item": "bouquet_m", "deliver": {"date": "$s1.date", "offset_days": -2}}, "after": ["s1", "s3"]},
  {"id": "s5", "tool": "merchant_order", "args": {"merchant": "<MERCHANT_2>", "item": "cake_8in", "deliver": {"date": "$s1.date", "offset_days": -2}}, "after": ["s1", "s3"]}
]}
{"kind": "clarify", "question": "你有两个联系人叫张伟，是哪一位？", "options": ["<PAYEE_2>", "<PAYEE_3>"]}
{"kind": "refuse", "reason": "out_of_scope | not_account_owner | not_supported"}
{"kind": "scam", "signals": ["safe_account", "police_claim"]}
```

Argument value forms (nothing else is accepted):
- token: `"<PAYEE_1>"`
- step reference: `"$s2.amount"`, `"$s1.date"`, `"$s1.payee"` (field must be one the source tool declares as output)
- literal: small enums/ints/short strings defined in the tool's schema (e.g. `divide_by: 3`, `item: "cake_8in"` (catalogue code; price comes from bank/merchants.py), `category: "food"`)
- relative date object: `{"date": <token|ref>, "offset_days": int}`
- **never** a raw number for an amount or account. Amounts are always `<AMT_n>` or a step ref.

The planner never outputs tiers, undo actions, or mandates. Code adds those.

## 3. Pipeline objects

- `ValidatedPlan` (from `security/dag.py`): plan whose tools are allowlisted, args typed, refs resolvable, acyclic, ≤ 12 steps.
- `PlanDecision` (from `security/policy.py`): per step `tier ∈ green|yellow|red`, `reasons[]`, or `veto` with reasons.
- `Mandate` (from `security/mandate.py`): signed `{customer, session, plan, step, action, payee_ref, amount_fen, expires_at, nonce}`.
- The executor (later, `agent/graph.py`) resolves tokens → real values only at the moment of calling `bank/api.py`, after re-checking policy and the mandate.
