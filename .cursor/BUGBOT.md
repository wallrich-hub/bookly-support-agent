# Bookly review rules

- Flag any tool parameter or UI control that lets a model or client name an

  amount of money.

- Flag any side-effecting endpoint the front end can call without going

  through the agent and the tool registry.

- Flag order-touching code without an ownership check against

  session.verified.

- Flag float arithmetic on money; require app/backend/[money.py](http://money.py).

- Flag refund calls without an idempotency key, or with an automatic retry.

- Flag innerHTML built from server data without esc().