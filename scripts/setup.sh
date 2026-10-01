#!/usr/bin/env bash
# One-time setup: Python 3.11 environment, dependencies, and every local model:
# MediaPipe hand model, openWakeWord "hey jarvis", Whisper, MiniLM (intents), Qwen 2.5 via Ollama.
#   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv (fast Python package manager)"
  brew install uv
fi

if [ ! -d .venv ]; then
  echo "==> Python 3.11 environment in .venv (MediaPipe doesn't support the newest Python yet)"
  uv venv --python 3.11 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r requirements.txt

mkdir -p models
if [ ! -f models/hand_landmarker.task ]; then
  echo "==> Downloading MediaPipe hand model"
  curl -fsSL -o models/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
fi

echo "==> Downloading wake word, Whisper and intent models (one time)"
python - <<'EOF'
import openwakeword.utils
openwakeword.utils.download_models(["hey_jarvis"])
from faster_whisper import WhisperModel
WhisperModel("base.en", device="cpu", compute_type="int8")
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("models ready")
EOF

echo "==> Local LLM (Ollama + Qwen 2.5 3B, about 2 GB)"
if ! command -v ollama >/dev/null 2>&1; then
  brew install ollama
fi
if ! curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
  brew services start ollama >/dev/null 2>&1 || (ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 1 20); do curl -fs http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 0.5; done
fi
ollama pull qwen2.5:3b

echo "==> Running tests"
python -m pytest -q

echo ""
echo "Done. Start it with:"
echo "  source .venv/bin/activate && python -m nexus --preview"
