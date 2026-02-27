#!/usr/bin/env bash
# Download Qwen3-Next-80B-A3B-Instruct Q8_0 GGUF from HuggingFace
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
MODEL_DIR="$PROJECT_DIR/models"

REPO="Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF"
FILE="Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf"

# Check if model already exists
if [[ -f "$MODEL_DIR/$FILE" ]]; then
    echo "Model already exists at $MODEL_DIR/$FILE"
    ls -lh "$MODEL_DIR/$FILE"
    exit 0
fi

# Set up venv if needed
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python venv..."
    python3 -m venv "$VENV_DIR"
fi

echo "Installing huggingface-hub..."
"$VENV_DIR/bin/pip" install --quiet huggingface-hub

echo "Downloading $FILE (~84.8 GB)..."
echo "This will take a while. Progress shown below."
"$VENV_DIR/bin/huggingface-cli" download "$REPO" "$FILE" --local-dir "$MODEL_DIR"

echo ""
echo "Download complete!"
ls -lh "$MODEL_DIR/$FILE"
