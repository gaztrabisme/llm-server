#!/bin/bash
# S016.5: PP speed + VRAM breakdown for all 3 configs
# Collects: prompt processing speed at 512/2k/8k/16k tokens + VRAM allocation from startup logs
set -euo pipefail

IMAGE="llm-server/llama-cpp:b9204"
CONTAINER="s016-pp-bench"
PORT=8081
OUTDIR="benchmarks/s016-pp"
mkdir -p "$OUTDIR"

# Generate prompts of known token lengths (approximate: 1 token ≈ 4 chars for English)
generate_prompt() {
    local target_tokens=$1
    local chars=$((target_tokens * 4))
    python3 -c "
import json
# Repeat a realistic text block to hit target size
block = 'The quick brown fox jumps over the lazy dog. In computer science, artificial intelligence is the study of intelligent agents. Machine learning algorithms build models based on sample data to make predictions. Neural networks are computing systems inspired by biological neural networks. '
text = (block * ($chars // len(block) + 1))[:$chars]
print(json.dumps(text))
"
}

# Configs: label, model, fit_target, extra_flags
declare -a CONFIGS=(
    "A-27b-iq3|Qwen3.6-27B-MTP-UD-IQ3_XXS.gguf|0|"
    "B-35b-q4kxl|Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf|1536|"
    "C-35b-q8|Qwen3.6-35B-A3B-MTP-Q8_0.gguf|1536|"
)

for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r LABEL MODEL FIT_TARGET EXTRA <<< "$CONFIG"
    echo ""
    echo "=========================================="
    echo "Config: $LABEL"
    echo "=========================================="

    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true

    docker run --gpus all --ipc host -d --name "$CONTAINER" \
        -p ${PORT}:8080 \
        -v /home/bppc/projects/llm-server/models:/models:ro \
        "$IMAGE" \
        -m "/models/$MODEL" \
        -c 32768 \
        -np 1 \
        --host 0.0.0.0 \
        --fit on \
        --fit-target "$FIT_TARGET" \
        -fa on \
        -t 20 \
        --no-mmap \
        --jinja \
        -ctk q8_0 \
        -ctv q8_0 \
        --spec-type draft-mtp \
        --spec-draft-n-max 2 \
        $EXTRA \
        2>&1 >/dev/null

    # Wait for ready
    READY=0
    for i in $(seq 1 120); do
        curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1 && READY=1 && break
        sleep 2
    done

    if [ "$READY" -eq 0 ]; then
        echo "  FAIL: server didn't start"
        docker logs "$CONTAINER" 2>&1 | tail -10
        docker stop "$CONTAINER" 2>/dev/null; docker rm "$CONTAINER" 2>/dev/null
        continue
    fi

    echo "  Server ready."

    # === VRAM breakdown from logs ===
    echo ""
    echo "--- VRAM Breakdown ---"
    docker logs "$CONTAINER" 2>&1 | grep -iE "layer|offload|VRAM|vram|device (CUDA|CPU)|kv cache|compute buffer|MiB|GiB" | grep -v "^$" | tail -30 > "$OUTDIR/vram-${LABEL}.log"

    # Extract key metrics
    GPU_LAYERS=$(docker logs "$CONTAINER" 2>&1 | grep -c "assigned to device CUDA" || echo "0")
    CPU_LAYERS=$(docker logs "$CONTAINER" 2>&1 | grep -c "assigned to device CPU" || echo "0")
    echo "  GPU layers: $GPU_LAYERS, CPU layers: $CPU_LAYERS"

    # VRAM summary from nvidia-smi
    VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "?")
    VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo "?")
    echo "  VRAM used: ${VRAM_USED} MiB, free: ${VRAM_FREE} MiB"
    echo "$LABEL: GPU=$GPU_LAYERS CPU=$CPU_LAYERS VRAM_used=${VRAM_USED}MiB VRAM_free=${VRAM_FREE}MiB" >> "$OUTDIR/vram-summary.txt"

    # Extract compute buffer and KV cache info
    docker logs "$CONTAINER" 2>&1 | grep -iE "compute buffer|kv (buf|cache|self)" | head -5

    # === PP benchmark ===
    echo ""
    echo "--- PP Speed Benchmark ---"

    for PP_TOKENS in 512 2048 8192 16384; do
        PROMPT=$(generate_prompt $PP_TOKENS)

        # Send completion request with max_tokens=1 to measure PP only
        RESULT=$(curl -sf http://127.0.0.1:${PORT}/v1/completions \
            -H "Content-Type: application/json" \
            -d "{
                \"prompt\": $PROMPT,
                \"max_tokens\": 1,
                \"temperature\": 0.0
            }" 2>/dev/null || echo '{"error": true}')

        # Extract timings
        PP_SPEED=$(echo "$RESULT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    if 'error' in d and d['error'] == True:
        print('ERROR')
    elif 'usage' in d:
        u = d['usage']
        prompt_tokens = u.get('prompt_tokens', 0)
        # Try to get timings from the response
        print(f'tokens={prompt_tokens}')
    else:
        print('NO_USAGE')
except Exception as e:
    print(f'PARSE_ERR: {e}')
" 2>/dev/null)

        # Use /v1/completions with timings from the server's slot endpoint
        TIMINGS=$(curl -sf http://127.0.0.1:${PORT}/slots 2>/dev/null | python3 -c "
import json, sys
try:
    slots = json.load(sys.stdin)
    if isinstance(slots, list) and slots:
        s = slots[0]
        pp = s.get('t_prompt_processing', 0)
        pp_n = s.get('n_prompt_tokens_processed', 0)
        tps = pp_n / (pp/1000) if pp > 0 else 0
        print(f'{tps:.1f}')
    else:
        print('0')
except:
    print('0')
" 2>/dev/null || echo "0")

        echo "  PP-${PP_TOKENS}: ${TIMINGS} tok/s  (${PP_SPEED})"
        echo "${LABEL},${PP_TOKENS},${TIMINGS}" >> "$OUTDIR/pp-results.csv"
    done

    docker stop "$CONTAINER" 2>/dev/null; docker rm "$CONTAINER" 2>/dev/null
    echo ""
    echo "  Config $LABEL complete."
done

echo ""
echo "=========================================="
echo "All configs complete. Results in $OUTDIR/"
echo "=========================================="
