#!/bin/bash
# S016 T3: Ubatch PP sweep for partial-offload configs
# Tests coder543's discovery: increasing -ub from 512→8192 gives 5.5x PP speedup
# Only meaningful for partial offload — no effect when model fits entirely on GPU
#
# Usage: ./scripts/s016-ubatch-sweep.sh <config_name> <model_file> [fit_target]
set -euo pipefail

CONFIG="$1"
MODEL="$2"
FIT_TARGET="${3:-1536}"

IMAGE="llm-server/llama-cpp:b9204"
CONTAINER="s016-ub-${CONFIG}"
PORT=8081

UB_VALUES=(512 1024 2048 4096 8192)

echo "=== T3: Ubatch PP sweep (coder543 trick) ==="
echo "Config: $CONFIG, Model: $MODEL, fit-target: $FIT_TARGET"
echo ""

for UB in "${UB_VALUES[@]}"; do
    echo -n "  -ub $UB: "
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true

    docker run --gpus all --ipc host -d --name "$CONTAINER" \
        -p ${PORT}:8080 \
        -v /home/bppc/projects/llm-server/models:/models:ro \
        "$IMAGE" \
        -m "/models/$MODEL" \
        -c 8192 \
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
        -ub "$UB" \
        2>&1 >/dev/null

    # Wait for ready
    READY=0
    for i in $(seq 1 60); do
        curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1 && READY=1 && break
        sleep 2
    done

    if [ "$READY" -eq 0 ]; then
        echo "FAIL (server didn't start)"
        docker logs "$CONTAINER" 2>&1 | grep -iE "error|oom" | tail -2
        continue
    fi

    # Get layers info
    OVERFLOW=$(docker logs "$CONTAINER" 2>&1 | grep -c "assigned to device CPU" 2>/dev/null || echo "0")

    # Test PP: send a long prompt (512 tokens of padding) and measure PP speed
    # Use /completion endpoint with a long prompt
    PP_RESULT=$(curl -sf "http://127.0.0.1:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\":\"$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 60)")Answer with just 'ok'.\",\"n_predict\":8,\"temperature\":0.0,\"stream\":false}" \
        2>&1)

    PP_TPS=$(echo "$PP_RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
t = d.get('timings', {})
pp = t.get('prompt_per_second', 0)
tg = t.get('predicted_per_second', 0)
pp_n = t.get('prompt_n', 0)
print(f'PP={pp:.0f} tok/s ({pp_n} tokens), TG={tg:.1f} tok/s, overflow={\"$OVERFLOW\"}')" 2>/dev/null || echo "parse_fail")

    echo "$PP_TPS"

    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true
done

echo ""
echo "=== Ubatch sweep complete ==="
