#!/bin/bash
# Started by Nexus.app (which is only a small stub, so it never needs rebuilding).
# Keeps the app up to date:
#   - installs new dependencies whenever requirements.txt changes
#   - restarts Nexus when its code changes (the app exits with code 75 to ask for that)
#   - shows a dialog if Nexus stops right after starting, instead of vanishing silently
APP="${1:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export NEXUS_APP="$APP"
cd "$ROOT" || exit 1
mkdir -p logs
OUT="logs/app-output.log"

say_log() { echo "$(date '+%Y-%m-%d %H:%M:%S') launcher: $1" >> "$OUT"; }
notify() { osascript -e "display notification \"$1\" with title \"Nexus\"" >/dev/null 2>&1; }

if [ ! -x .venv/bin/python ]; then
  osascript -e 'display dialog "Nexus is not set up yet. In Terminal, go to the nexus-py folder and run: bash scripts/setup.sh" buttons {"OK"} default button 1 with title "Nexus"' >/dev/null 2>&1
  exit 1
fi

# Already running: bring its window forward instead of starting a second copy
# (two would fight over the camera and mic)
if pgrep -f "python -m nexus --app" >/dev/null; then
  pkill -USR1 -f "python -m nexus --app"
  exit 0
fi

sync_dependencies() {
  local want have
  want="$(shasum requirements.txt | cut -d' ' -f1)"
  have="$(cat .venv/.nexus-requirements 2>/dev/null)"
  [ "$want" = "$have" ] && return 0
  notify "Installing an update. Nexus opens in a moment."
  say_log "requirements.txt changed, installing"
  if VIRTUAL_ENV="$ROOT/.venv" uv pip install -r requirements.txt >> "$OUT" 2>&1; then
    echo "$want" > .venv/.nexus-requirements
    say_log "dependencies up to date"
  else
    say_log "dependency install failed"
    notify "An update could not be installed. Details are in logs/app-output.log."
  fi
}

# The local LLM for complex requests (harmless if Ollama is already running)
if ! curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
  brew services start ollama >/dev/null 2>&1 || true
fi

while true; do
  sync_dependencies
  started=$(date +%s)
  say_log "starting Nexus"
  .venv/bin/python -m nexus --app >> "$OUT" 2>&1
  code=$?
  if [ "$code" -eq 75 ]; then
    say_log "restarting for an update"
    continue
  fi
  say_log "Nexus exited with code $code"
  if [ "$code" -ne 0 ] && [ $(( $(date +%s) - started )) -lt 20 ]; then
    osascript -e 'display dialog "Nexus stopped right after starting. The details are in logs/app-output.log in the nexus-py folder." buttons {"OK"} default button 1 with title "Nexus"' >/dev/null 2>&1
  fi
  break
done
