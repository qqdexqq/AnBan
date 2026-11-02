# Standing-order execution and recovery

The scheduled-transfer tool can create and run a monthly standing order. These
changes do not require retraining, model replacement or a live-database reset.

## What executes

The existing scheduled-transfer tool creates a monthly standing order through
the ordinary policy and approval flow. The runner observes each order once
without paying, then evaluates later monthly occurrences. A customer/order/month
primary key and deterministic transfer idempotency key prevent duplicate payments.
Earlier months are marked missed rather than charged in bulk. A skipped occurrence
is not automatically retried later that month.

Each due payment rechecks the active order under the write lock, the payee,
verification status, fraud list, scam and security locks, daily outgoing exposure,
elder limit and balance. Open errand budgets issued that day count against the
daily quota; this is quota accounting, not a hold on account funds. Operations
above the unattended daily limits are skipped rather than automatically escalated.
An unavailable fraud check fails closed.

Payments use the standard yellow-tier cooling window, with the customer's chosen
arrival delay as a floor. A customer who selected delayed arrival keeps it.
Stopping a standing order is yellow and requires confirmation. It stops future
occurrences; it does not automatically undo a previously submitted transfer.
Negations and questions do not directly stop an order. Mixed financial requests
and multiple explicitly named payees are left to the planner rather than
truncated.

## Audit and recovery

Both chat and summary API entry points use `agent/standing_order_hook.py`.
The durable `standing_order_runs` journal is scanned for missing audit entries,
including rows persisted before a crash or transient logging failure. Appending
one entry unsuccessfully does not hide successful payments or discard the other
rows. The next hook invocation retries the pending journal without re-debiting.
If a previous audit backlog remains unavailable, it does not start a new batch.
The hook serializes its work within this application process. A partial unique
audit index also prevents independent processes from appending duplicate run
events for the same customer/order/month. Existing duplicate history is never
deleted automatically: if index creation fails, new batches remain blocked for
manual review. Payment uniqueness is enforced separately by SQLite and transfer
idempotency.

Historical paid markers missing their transfer ID are linked to an existing
customer-scoped idempotent ledger entry with the same amount. Recovery never
creates a new payment. The unattended hook refuses to run inside a caller's
unfinished transaction, so settings setup or nested bank calls cannot commit
unrelated work.

## Runtime boundary

There is no independent wall-clock scheduler. The current implementation checks
due orders on a chat turn or authenticated `/me/summary` request. Opening the
dashboard can therefore submit an already-authorised due payment. If there is no
activity, execution can be late; it is not a guarantee of execution at a
particular clock time. A continuously running timer or worker is separate
deployment work and should be tested before making an on-time execution claim.

The database run status `paid` means a transfer was submitted into the bank's
cooling queue; it is not proof of final settlement. Review the transfer's state
and release time for that distinction. Audit explanations distinguish submitted,
skipped and missed runs instead of claiming every journal row executed. Returning
an API summary also runs the existing due-payment hook, so use a disposable demo
database for tests and demos.

## Limitations

- `security/audit.py` changed for truthful standing-order outcome labels and
  still needs adversarial review together with the bank payment and audit-hook
  changes.
- The change does not modify the live API, database, chats, model weights or GPU
  jobs.
- The frozen model prompts are not changed.

## Verification

Final validation: **2,129 Python tests passed**, with no failures or skips.
Regression cases cover audit outage and replay, concurrent audit coordinators,
legacy duplicate-history preservation, cancellation after prefetch, fraud-list
changes, unfinished caller transactions, crash-link repair, stop confirmation
and rejection, Chinese and English negation and mixed requests.
