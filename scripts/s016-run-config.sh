#!/bin/bash
# S016 Phase 5: Run full test suite for a model config
# Usage: ./scripts/s016-run-config.sh <config_name> <model_file> [extra_server_args...]
#
# Runs: T1 (mtp-bench), T2 (context push), T3 (ubatch PP sweep, if partial offload),
#       T4 (CodeNeedle), T5 (GSM8K 100)
set -euo pipefail

CONFIG="$1"
MODEL="$2"
shift 2
EXTRA="${@:-}"

IMAGE="llm-server/llama-cpp:b9204"
CONTAINER="s016-${CONFIG}"
PORT=8081
RESULTS_DIR="benchmarks"
SCRIPTS_DIR="scripts"
CN_DIR="benchmarks/codeneedle"

mkdir -p "$RESULTS_DIR"

stop_server() {
    docker stop "$CONTAINER" 2>/dev/null || true
    docker rm "$CONTAINER" 2>/dev/null || true
}

start_server() {
    local ctx="$1"
    local extra_args="${2:-}"
    stop_server
    docker run --gpus all --ipc host -d --name "$CONTAINER" \
        -p ${PORT}:8080 \
        -v /home/bppc/projects/llm-server/models:/models:ro \
        "$IMAGE" \
        -m "/models/$MODEL" \
        -c "$ctx" \
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
        $extra_args 2>&1

    # Wait for ready
    for i in $(seq 1 60); do
        curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1 && return 0
        sleep 2
    done
    echo "ERROR: Server failed to start"
    docker logs "$CONTAINER" 2>&1 | grep -iE "error|oom|fail" | tail -5
    return 1
}

echo "================================================"
echo "S016 Config: $CONFIG"
echo "Model: $MODEL"
echo "Extra args: $EXTRA"
echo "================================================"

# === T1: mtp-bench ===
echo ""
echo "=== T1: mtp-bench ==="
start_server 8192 "$EXTRA"
source .venv/bin/activate
python3 "$SCRIPTS_DIR/mtp-bench.py" \
    --url "http://127.0.0.1:${PORT}" \
    --out "$RESULTS_DIR/s016-${CONFIG}-t1-mtp-bench.json"

# === T2: Context push ===
echo ""
echo "=== T2: Context push (q8_0 KV) ==="
for CTX in 32768 49152 56000 65536 98304; do
    echo -n "  ctx=$CTX: "
    if start_server "$CTX" "$EXTRA" 2>/dev/null; then
        RESULT=$(curl -sf "http://127.0.0.1:${PORT}/completion" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"Hello","n_predict":32,"temperature":0.0,"stream":false}' 2>&1)
        TPS=$(echo "$RESULT" | python3 -c "import sys,json; t=json.load(sys.stdin).get('timings',{}); print(f'{t.get(\"predicted_per_second\",0):.1f}')" 2>/dev/null || echo "gen_fail")
        echo "OK (tok/s=$TPS)"
    else
        echo "FAIL (OOM)"
        break
    fi
done

echo ""
echo "=== T2: Context push (q4_0 KV) ==="
for CTX in 65536 98304 110592 131072; do
    echo -n "  ctx=$CTX: "
    stop_server
    docker run --gpus all --ipc host -d --name "$CONTAINER" \
        -p ${PORT}:8080 \
        -v /home/bppc/projects/llm-server/models:/models:ro \
        "$IMAGE" \
        -m "/models/$MODEL" \
        -c "$CTX" -np 1 --host 0.0.0.0 --fit on --fit-target 0 -fa on -t 20 --no-mmap --jinja \
        -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 2 \
        $EXTRA 2>&1 >/dev/null

    READY=0
    for i in $(seq 1 60); do
        curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && READY=1 && break
        sleep 2
    done

    if [ "$READY" -eq 1 ]; then
        RESULT=$(curl -sf "http://127.0.0.1:${PORT}/completion" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"Hello","n_predict":32,"temperature":0.0,"stream":false}' 2>&1)
        TPS=$(echo "$RESULT" | python3 -c "import sys,json; t=json.load(sys.stdin).get('timings',{}); print(f'{t.get(\"predicted_per_second\",0):.1f}')" 2>/dev/null || echo "gen_fail")
        echo "OK (tok/s=$TPS)"
    else
        echo "FAIL (OOM)"
        break
    fi
done

# === T4: CodeNeedle ===
echo ""
echo "=== T4: CodeNeedle ==="
start_server 32768 "$EXTRA"

# Create CodeNeedle model config
cat > "$CN_DIR/configs/models/s016-${CONFIG}.toml" << EOF
name              = "s016-${CONFIG}"
base_url          = "http://localhost:${PORT}"
temperature       = 0.0
max_tokens        = 32768
timeout           = 600.0
prefill_no_think  = false
suppress_thinking = true
EOF

cd "$CN_DIR"
python3 bench.py run --model "s016-${CONFIG}" --corpus http_server
cd /home/bppc/projects/llm-server

# === T5: GSM8K 100 ===
echo ""
echo "=== T5: GSM8K 100 ==="
start_server 32768 "$EXTRA"
python3 "$SCRIPTS_DIR/llama-eval.py" \
    --dataset gsm8k \
    --n_cases 100 \
    --server "http://127.0.0.1:${PORT}" \
    --output "$RESULTS_DIR/s016-${CONFIG}-t5-gsm8k-100.json" \
    --n_predict 32768 \
    --temperature 1.0 \
    --top-k 20 \
    --top-p 0.95 \
    --grader-type regex

stop_server

echo ""
echo "================================================"
echo "Config $CONFIG complete!"
echo "================================================"
