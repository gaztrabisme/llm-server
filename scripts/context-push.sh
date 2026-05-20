#!/bin/bash
# Context push to OOM test — incrementally tests context sizes
# Usage: ./scripts/context-push.sh <model_path> <image> [extra_args...]
set -euo pipefail

MODEL="$1"
IMAGE="$2"
shift 2
EXTRA_ARGS="$@"
CONTAINER="s016-ctx-push"
PORT=8081

CONTEXTS=(32768 49152 56000 65536 98304 131072)

for CTX in "${CONTEXTS[@]}"; do
    echo "=== Testing context=$CTX ==="

    # Clean up any existing container
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true

    # Start server
    docker run --gpus all --ipc host -d --name "$CONTAINER" \
        -p ${PORT}:8080 \
        -v /home/bppc/projects/llm-server/models:/models:ro \
        "$IMAGE" \
        -m "/models/$MODEL" \
        -c "$CTX" \
        -np 1 \
        --host 0.0.0.0 \
        --fit on \
        --fit-target 0 \
        -fa on \
        -t 20 \
        --no-mmap \
        --jinja \
        -ctk q8_0 \
        -ctv q8_0 \
        --spec-type draft-mtp \
        --spec-draft-n-max 2 \
        $EXTRA_ARGS 2>&1

    # Wait for server to be ready (up to 60s)
    READY=0
    for i in $(seq 1 30); do
        if curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1; then
            READY=1
            break
        fi
        sleep 2
    done

    if [ "$READY" -eq 0 ]; then
        echo "  FAILED: Server didn't start at ctx=$CTX"
        docker logs "$CONTAINER" 2>&1 | grep -iE "error|oom|killed|fail|crash" | tail -5
        docker stop "$CONTAINER" 2>/dev/null || true
        docker rm "$CONTAINER" 2>/dev/null || true
        echo "  RESULT: ctx=$CTX = OOM/FAIL"
        continue
    fi

    # Get layer info from logs
    LAYERS=$(docker logs "$CONTAINER" 2>&1 | grep -oP 'offloaded \K\d+/\d+ layers' 2>/dev/null || echo "n/a")
    OVERFLOW=$(docker logs "$CONTAINER" 2>&1 | grep -c "assigned to device CPU" 2>/dev/null || echo "0")

    # Send a test prompt to verify generation works
    RESULT=$(curl -sf http://127.0.0.1:${PORT}/completion \
        -H "Content-Type: application/json" \
        -d '{"prompt":"Write hello world in Python:","n_predict":64,"temperature":0.0,"seed":42,"stream":false}' \
        2>&1) || RESULT="REQUEST_FAILED"

    if echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('timings',{}); print(f'  tok/s={t.get(\"predicted_per_second\",0):.1f}, tokens={t.get(\"predicted_n\",0)}, draft_acc={t.get(\"draft_n_accepted\",0)}/{t.get(\"draft_n\",0)}')" 2>/dev/null; then
        echo "  RESULT: ctx=$CTX = OK (layers=$LAYERS, cpu_overflow=$OVERFLOW)"
    else
        echo "  RESULT: ctx=$CTX = GENERATION_FAILED"
        echo "  Response: $(echo "$RESULT" | head -c 200)"
    fi

    # Clean up
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true
    echo ""
done

echo "=== Context push complete ==="
