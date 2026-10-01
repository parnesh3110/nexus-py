#!/usr/bin/env bash
# Commit everything and push to GitHub (creates the nexus-py repo on first run).
#   bash scripts/commit.sh "message"
set -euo pipefail
cd "$(dirname "$0")/.."
msg="${1:-Update}"

[ -d .git ] || git init -b main
git add -A
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -m "$msg"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  if command -v gh >/dev/null 2>&1; then
    gh repo create nexus-py --public --source=. --push
    exit 0
  fi
  echo "No GitHub remote yet. Create an empty repo called nexus-py on github.com, then run:"
  echo "  git remote add origin https://github.com/<you>/nexus-py.git && git push -u origin main"
  exit 1
fi
git push -u origin main
