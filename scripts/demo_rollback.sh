#!/usr/bin/env bash
# Put the Bookly demo back to "before". Never touches main or the pre-baked PR.
set -euo pipefail
REPO=wallrich-hub/bookly-support-agent
KEEP=feat/order-cards-prebaked

# 1. Close open demo PRs into stage and delete their branches
gh pr list -R "$REPO" --base stage --state open --json number,headRefName \
  -q '.[] | "\(.number) \(.headRefName)"' | while read -r n br; do
  [ "$br" = "$KEEP" ] && continue
  gh pr close -R "$REPO" "$n" --delete-branch --comment "Demo rollback"
done

# 2. Delete branches left behind by merged demo PRs
gh pr list -R "$REPO" --base stage --state merged --json headRefName \
  -q '.[].headRefName' | while read -r br; do
  git push origin --delete "$br" 2>/dev/null || true
done

# 3. Reset stage to the baseline tag
git fetch origin --tags --quiet
git push --force origin demo-baseline:refs/heads/stage

# 4. Reset the running "before" app
curl -fsS -X POST http://127.0.0.1:8000/api/admin/reset >/dev/null \
  && echo "App reset" || echo "App not running on 8000, restart uvicorn"

echo "Done. Archive agents in Cursor, cancel the Linear issue, post run N+1."