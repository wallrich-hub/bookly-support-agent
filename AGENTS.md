# Bookly: rules for agents

Thesis: the LLM decides what to SAY. It never decides what is TRUE or PERMITTED.

- Every consequential decision (eligibility, amounts, human approval) lives in

  app/policy/[engine.py](http://engine.py) as plain Python returning a Decision. Never in a prompt.

- Tools are intent-shaped. Never add a tool parameter that lets the model name

  an amount of money or a customer id. Identity comes from session.verified.

- Every order-touching tool re-checks order ownership against

  session.verified["customer_id"] and re-derives the policy decision at write time.

- Money goes through app/backend/[money.py](http://money.py) (Decimal). Never float arithmetic.

- Refunds go through bookly_api.issue_refund with an idempotency key. Never

  auto-retry a refund POST.

- Side-effecting tools set side_effecting=True and write to AUDIT_LOG.

- UI: buttons and actions come from an `actions` list the server computes from

  policy. The front end never decides eligibility and never calls a side-effecting

  endpoint directly: a click sends a chat message that goes through the agent.

  Escape every server value with esc() before rendering.

- Every new behaviour needs a scenario test in tests/test_[scenarios.py](http://scenarios.py) that

  passes with the mock planner (LLM_PROVIDER=mock). Run: pytest -q

- New customer-facing policy needs a KB passage in app/backend/[kb.py](http://kb.py) with an id.

- Open pull requests against the `stage` branch, never `main`.