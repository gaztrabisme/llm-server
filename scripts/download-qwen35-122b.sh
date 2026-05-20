#!/usr/bin/env bash
# Download Qwen3.5-122B-A10B Q4_K_M GGUF from HuggingFace (Unsloth)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
MODEL_DIR="$PROJECT_DIR/models"

REPO="unsloth/Qwen3.5-122B-A10B-GGUF"
FILE="Qwen3.5-122B-A10B-Q4_K_M.gguf"

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

echo "Downloading $FILE (~76.5 GB)..."
echo "This will take a while. Progress shown below."
"$VENV_DIR/bin/huggingface-cli" download "$REPO" "$FILE" --local-dir "$MODEL_DIR"

echo ""
echo "Download complete!"
ls -lh "$MODEL_DIR/$FILE"
