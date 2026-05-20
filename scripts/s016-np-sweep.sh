#!/bin/bash
# S016 I5: Sweep -np 1/2/4 on a model with 10 GSM8K cases
# Usage: ./scripts/s016-np-sweep.sh <label> <model_file> <fit_target>
set -euo pipefail

LABEL="$1"
MODEL="$2"
FIT_TARGET="$3"

IMAGE="llm-server/llama-cpp:b9204"
CONTAINER="s016-np-sweep"
PORT=8081

echo "=== -np sweep: $LABEL (fit-target=$FIT_TARGET) ==="

for NP in 1 2 4; do
    echo ""
    echo "--- -np $NP ---"
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true

    docker run --gpus all --ipc host -d --name "$CONTAINER" \
        -p ${PORT}:8080 \
        -v /home/bppc/projects/llm-server/models:/models:ro \
        "$IMAGE" \
        -m "/models/$MODEL" \
        -c 32768 \
        -np "$NP" \
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
        2>&1 >/dev/null

    # Wait for ready (up to 3 min for Q8_0)
    READY=0
    for i in $(seq 1 90); do
        curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1 && READY=1 && break
        sleep 2
    done

    if [ "$READY" -eq 0 ]; then
        echo "  FAIL: server didn't start"
        docker logs "$CONTAINER" 2>&1 | grep -iE "error|oom|fail" | tail -3
        docker stop "$CONTAINER" 2>/dev/null; docker rm "$CONTAINER" 2>/dev/null
        continue
    fi

    # Check if layers pushed to CPU
    CPU_LAYERS=$(docker logs "$CONTAINER" 2>&1 | grep -c "assigned to device CPU" || echo "0")
    echo "  CPU overflow layers: $CPU_LAYERS"

    # Run 10 GSM8K cases with matching concurrency
    OUTPUT="benchmarks/s016-np-sweep-${LABEL}-np${NP}.json"
    START=$(date +%s)

    source .venv/bin/activate
    python3 scripts/llama-eval.py \
        --dataset gsm8k \
        --n_cases 10 \
        --server "http://127.0.0.1:${PORT}" \
        --output "$OUTPUT" \
        --n_predict 32768 \
        --temperature 1.0 \
        --top-k 20 \
        --top-p 0.95 \
        --grader-type regex \
        --concurrency "$NP" \
        2>&1 | tail -3

    END=$(date +%s)
    WALL=$((END - START))

    # Parse results
    python3 -c "
import json
d = json.load(open('$OUTPUT'))
cases = d['task_states']['cases']
done = {k:v for k,v in cases.items() if v.get('status') not in ('pending', None, '')}
correct = {k:v for k,v in done.items() if v.get('correct')}
toks = [v.get('tokens', 0) for v in done.values() if v.get('tokens')]
tps = [v.get('tps_gen') for v in done.values() if v.get('tps_gen')]
avg_tps = sum(tps)/len(tps) if tps else 0
print(f'  Results: {len(correct)}/{len(done)} correct, avg {avg_tps:.1f} tok/s/request')
print(f'  Wall time: ${WALL}s, throughput: {len(done)*60/${WALL}:.1f} cases/min')
" 2>/dev/null || echo "  Parse error"

    docker stop "$CONTAINER" 2>/dev/null; docker rm "$CONTAINER" 2>/dev/null
done

echo ""
echo "=== -np sweep complete ==="
