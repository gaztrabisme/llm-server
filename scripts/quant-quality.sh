#!/usr/bin/env bash
# Quantization quality benchmark: Perplexity + KL Divergence
# Usage: ./quant-quality.sh [reference_quant]
#
# Runs llama-perplexity on WikiText-2 for each Qwen3.5-35B-A3B quant.
# Step 1: Generate logits for reference model (default: Q8_0)
# Step 2: Run perplexity + KLD for each comparison quant
#
# Results saved to benchmarks/perplexity/
#
# Examples:
#   ./quant-quality.sh              # Q8_0 as reference, compare Q4_K_M + UD-Q4_K_XL
#   ./quant-quality.sh Q4_K_M       # Use Q4_K_M as reference instead

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/models"
BENCH_DIR="$PROJECT_DIR/benchmarks/perplexity"
WIKI_FILE="$BENCH_DIR/wikitext-2-raw/wiki.test.raw"
IMAGE="llm-server/llama-cpp:latest"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

# Models to test
REFERENCE="${1:-Q8_0}"
QUANTS=("Q8_0" "Q4_K_M" "UD-Q4_K_XL")

# Map quant name to GGUF filename
quant_to_file() {
    local quant="$1"
    echo "Qwen3.5-35B-A3B-${quant}.gguf"
}

# Verify prerequisites
if [[ ! -f "$WIKI_FILE" ]]; then
    echo "WikiText-2 not found at $WIKI_FILE"
    echo "Run: curl -sL https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip | unzip -d benchmarks/perplexity/"
    exit 1
fi

for q in "${QUANTS[@]}"; do
    f="$MODEL_DIR/$(quant_to_file "$q")"
    if [[ ! -f "$f" ]]; then
        echo "Model not found: $f"
        echo "Download it first."
        exit 1
    fi
done

echo "=========================================="
echo " Quant Quality Benchmark"
echo " Reference: $REFERENCE"
echo " Compare: ${QUANTS[*]}"
echo " Timestamp: $TIMESTAMP"
echo "=========================================="

# Step 1: Run perplexity on reference model and save logits
REF_FILE="$(quant_to_file "$REFERENCE")"
REF_LOGITS="$BENCH_DIR/logits-${REFERENCE}-${TIMESTAMP}.bin"
REF_PPL_LOG="$BENCH_DIR/ppl-${REFERENCE}-${TIMESTAMP}.log"

echo ""
echo "=== Step 1: Reference model ($REFERENCE) — perplexity + save logits ==="
echo "Model: $REF_FILE"
echo "This will take a while..."
echo ""

docker run --rm \
    --gpus all \
    --ipc host \
    -v "$MODEL_DIR:/models:ro" \
    -v "$BENCH_DIR:/bench" \
    -v "$WIKI_FILE:/data/wiki.test.raw:ro" \
    --entrypoint llama-perplexity \
    "$IMAGE" \
        -m "/models/$REF_FILE" \
        -f /data/wiki.test.raw \
        -ngl 999 \
        -ot "exps=CPU" \
        -fa on \
        -t 20 \
        --no-mmap \
        --save-all-logits "/bench/logits-${REFERENCE}-${TIMESTAMP}.bin" \
    2>&1 | tee "$REF_PPL_LOG"

echo ""
echo "Reference logits saved: $REF_LOGITS"
echo "Reference PPL log: $REF_PPL_LOG"

# Extract final PPL from log
REF_PPL=$(grep -oP 'Final estimate: PPL = \K[0-9.]+' "$REF_PPL_LOG" 2>/dev/null || echo "N/A")
echo "Reference PPL: $REF_PPL"

# Step 2: Run KLD for each comparison quant
RESULTS_FILE="$BENCH_DIR/quant-quality-${TIMESTAMP}.json"
cat > "$RESULTS_FILE" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "reference": "$REFERENCE",
  "reference_ppl": "$REF_PPL",
  "wiki_corpus": "wikitext-2-raw/wiki.test.raw",
  "model_family": "Qwen3.5-35B-A3B",
  "results": {}
}
EOF

for q in "${QUANTS[@]}"; do
    QFILE="$(quant_to_file "$q")"
    KLD_LOG="$BENCH_DIR/kld-${q}-vs-${REFERENCE}-${TIMESTAMP}.log"

    echo ""
    echo "=== Step 2: $q — perplexity + KLD vs $REFERENCE ==="

    if [[ "$q" == "$REFERENCE" ]]; then
        echo "Skipping self-comparison (reference model)"
        # Save reference PPL result
        python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
data['results']['$q'] = {
    'ppl': '$REF_PPL',
    'kld_mean': 0,
    'kld_max': 0,
    'delta_p_mean': 0,
    'delta_p_rms': 0,
    'is_reference': True,
    'model_file': '$QFILE'
}
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
        continue
    fi

    docker run --rm \
        --gpus all \
        --ipc host \
        -v "$MODEL_DIR:/models:ro" \
        -v "$BENCH_DIR:/bench" \
        -v "$WIKI_FILE:/data/wiki.test.raw:ro" \
        "$IMAGE" \
        llama-perplexity \
            -m "/models/$QFILE" \
            -f /data/wiki.test.raw \
            -ngl 999 \
            -ot "exps=CPU" \
            -fa on \
            -t 20 \
            --no-mmap \
            --kl-divergence-base "/bench/logits-${REFERENCE}-${TIMESTAMP}.bin" \
        2>&1 | tee "$KLD_LOG"

    # Parse results from log
    PPL=$(grep -oP 'Final estimate: PPL = \K[0-9.]+' "$KLD_LOG" 2>/dev/null || echo "N/A")
    KLD_MEAN=$(grep -oP 'KLD mean: \K[0-9.]+' "$KLD_LOG" 2>/dev/null || echo "N/A")
    KLD_MAX=$(grep -oP 'KLD max: \K[0-9.]+' "$KLD_LOG" 2>/dev/null || echo "N/A")
    DP_MEAN=$(grep -oP 'delta-p mean: \K[-0-9.]+' "$KLD_LOG" 2>/dev/null || echo "N/A")
    DP_RMS=$(grep -oP 'delta-p RMS: \K[0-9.]+' "$KLD_LOG" 2>/dev/null || echo "N/A")

    echo ""
    echo "$q results: PPL=$PPL KLD=$KLD_MEAN delta-p=$DP_MEAN"

    # Save to JSON
    python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
data['results']['$q'] = {
    'ppl': '$PPL',
    'kld_mean': '$KLD_MEAN',
    'kld_max': '$KLD_MAX',
    'delta_p_mean': '$DP_MEAN',
    'delta_p_rms': '$DP_RMS',
    'is_reference': False,
    'model_file': '$QFILE'
}
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
done

# Summary
echo ""
echo "=========================================="
echo " Results Summary"
echo "=========================================="
python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
print(f\"Reference: {data['reference']} (PPL: {data['reference_ppl']})\")
print()
print(f\"{'Quant':<16} {'PPL':>10} {'KLD mean':>12} {'Δp mean':>12} {'Δp RMS':>10}\")
print('-' * 62)
for q, r in data['results'].items():
    ref = ' (ref)' if r.get('is_reference') else ''
    print(f\"{q + ref:<16} {r['ppl']:>10} {str(r['kld_mean']):>12} {str(r['delta_p_mean']):>12} {str(r['delta_p_rms']):>10}\")
"
echo ""
echo "Full results: $RESULTS_FILE"
echo "=========================================="
