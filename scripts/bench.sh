#!/usr/bin/env bash
# Unified benchmark runner for llama.cpp and ik_llama.cpp
# Usage: ./bench.sh <engine> <config> [thread_count]
#   engine: llama-cpp | ik-llama
#   config: baseline | optimized
#   thread_count: optional override (default: from config)
#
# Examples:
#   ./bench.sh llama-cpp baseline
#   ./bench.sh ik-llama optimized 24
#   ./bench.sh llama-cpp optimized 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_DIR/benchmarks"
MODEL_DIR="$PROJECT_DIR/models"

ENGINE="${1:?Usage: bench.sh <engine> <config> [thread_count]}"
CONFIG="${2:?Usage: bench.sh <engine> <config> [thread_count]}"
THREAD_OVERRIDE="${3:-}"
IMAGE_OVERRIDE="${4:-}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RESULT_FILE="$BENCHMARK_DIR/${ENGINE}-${CONFIG}-${TIMESTAMP}.json"

# Map engine to docker compose profile and image
case "$ENGINE" in
    llama-cpp)
        PROFILE="llama-cpp"
        IMAGE="llm-server/llama-cpp:latest"
        ;;
    ik-llama)
        PROFILE="ik-llama"
        IMAGE="llm-server/ik-llama-cpp:latest"
        ;;
    *)
        echo "Unknown engine: $ENGINE (use: llama-cpp | ik-llama)"
        exit 1
        ;;
esac

# Allow image override via 4th argument
if [[ -n "$IMAGE_OVERRIDE" ]]; then
    IMAGE="$IMAGE_OVERRIDE"
    echo "Image override: $IMAGE"
fi

# Map config to env file
ENV_FILE="$PROJECT_DIR/configs/${ENGINE}-${CONFIG}.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Config not found: $ENV_FILE"
    exit 1
fi

# Model check moved to after env file parsing (dynamic path)

echo "=========================================="
echo " Benchmark: $ENGINE / $CONFIG"
echo " Timestamp: $TIMESTAMP"
echo "=========================================="

# Parse extra args from env file (for ik-llama optimized)
EXTRA_ARGS=""
if grep -q "^IK_EXTRA_ARGS=" "$ENV_FILE" 2>/dev/null; then
    EXTRA_ARGS=$(grep "^IK_EXTRA_ARGS=" "$ENV_FILE" | cut -d= -f2-)
fi

# Build server args from env file
SERVER_ARGS=""
DOCKER_ENV_ARGS=""
MODEL_PATH=""
while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" =~ ^# || "$key" == "IK_EXTRA_ARGS" ]] && continue
    case "$key" in
        LLAMA_ARG_MODEL)          SERVER_ARGS+=" -m $value"; MODEL_PATH="$value" ;;
        LLAMA_ARG_CTX_SIZE)       SERVER_ARGS+=" -c $value" ;;
        LLAMA_ARG_N_GPU_LAYERS)   SERVER_ARGS+=" -ngl $value" ;;
        LLAMA_ARG_OVERRIDE_TENSOR) SERVER_ARGS+=" -ot $value" ;;
        LLAMA_ARG_CPU_MOE)        [[ "$value" == "true" ]] && SERVER_ARGS+=" -cmoe" ;;
        LLAMA_ARG_FLASH_ATTN)     [[ "$value" == "true" ]] && SERVER_ARGS+=" -fa on" ;;
        LLAMA_ARG_THREADS)        SERVER_ARGS+=" -t ${THREAD_OVERRIDE:-$value}" ;;
        LLAMA_ARG_BATCH_SIZE)     SERVER_ARGS+=" -b $value" ;;
        LLAMA_ARG_UBATCH_SIZE)    SERVER_ARGS+=" -ub $value" ;;
        LLAMA_ARG_NO_MMAP)        [[ "$value" == "true" ]] && SERVER_ARGS+=" --no-mmap" ;;
        LLAMA_ARG_MLOCK)          [[ "$value" == "true" ]] && SERVER_ARGS+=" --mlock" ;;
        LLAMA_ARG_JINJA)          [[ "$value" == "true" ]] && SERVER_ARGS+=" --jinja" ;;
        LLAMA_ARG_HOST)           SERVER_ARGS+=" --host $value" ;;
        LLAMA_ARG_PORT)           SERVER_ARGS+=" --port $value" ;;
        LLAMA_ARG_CACHE_TYPE_K)   SERVER_ARGS+=" -ctk $value" ;;
        LLAMA_ARG_CACHE_TYPE_V)   SERVER_ARGS+=" -ctv $value" ;;
        LLAMA_ARG_SWA_FULL)       [[ "$value" == "true" ]] && SERVER_ARGS+=" --swa-full" ;;
        LLAMA_ARG_N_CPU_MOE)      SERVER_ARGS+=" --n-cpu-moe $value" ;;
        LLAMA_ARG_FIT)            [[ "$value" == "true" ]] && SERVER_ARGS+=" --fit on" ;;
        LLAMA_ARG_FIT_TARGET)     SERVER_ARGS+=" --fit-target $value" ;;
        LLAMA_ARG_MMPROJ)         SERVER_ARGS+=" --mmproj $value" ;;
        LLAMA_ARG_SPEC_TYPE)      SERVER_ARGS+=" --spec-type $value" ;;
        LLAMA_ARG_SPEC_NGRAM_N)   SERVER_ARGS+=" --spec-ngram-size-n $value" ;;
        LLAMA_ARG_SPEC_NGRAM_M)   SERVER_ARGS+=" --spec-ngram-size-m $value" ;;
        LLAMA_ARG_DRAFT_MAX)      SERVER_ARGS+=" --draft-max $value" ;;
        LLAMA_ARG_DRAFT_MIN)      SERVER_ARGS+=" --draft-min $value" ;;
        GGML_CUDA_GRAPH_OPT)     DOCKER_ENV_ARGS+=" -e GGML_CUDA_GRAPH_OPT=$value" ;;
    esac
done < "$ENV_FILE"

SERVER_ARGS+=" $EXTRA_ARGS"
THREAD_COUNT="${THREAD_OVERRIDE:-20}"

# Validate model file exists on host (map container path /models/... to host MODEL_DIR/...)
if [[ -n "$MODEL_PATH" ]]; then
    HOST_MODEL_PATH="${MODEL_DIR}/${MODEL_PATH#/models/}"
    if [[ ! -f "$HOST_MODEL_PATH" ]]; then
        echo "Model not found: $HOST_MODEL_PATH"
        echo "Check LLAMA_ARG_MODEL in $ENV_FILE"
        exit 1
    fi
else
    echo "No LLAMA_ARG_MODEL in $ENV_FILE"
    exit 1
fi

CONTAINER_NAME="llm-bench-${ENGINE}-${CONFIG}-$$"

echo ""
echo "Starting server container: $CONTAINER_NAME"
echo "Args: $SERVER_ARGS"
echo ""

# Start server in background
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc host \
    -v "$MODEL_DIR:/models:ro" \
    -p 8080:8080 \
    $DOCKER_ENV_ARGS \
    "$IMAGE" \
    $SERVER_ARGS

# Wait for server health
echo "Waiting for server to be ready..."
MAX_WAIT=600  # 10 minutes (model loading takes a while for 84.8 GB)
WAITED=0
while ! curl -sf http://localhost:8080/health > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "Server did not become healthy within ${MAX_WAIT}s"
        docker logs "$CONTAINER_NAME" 2>&1 | tail -20
        docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1
        exit 1
    fi
    printf "\r  Waiting... %ds / %ds" "$WAITED" "$MAX_WAIT"
done
echo ""
echo "Server healthy!"

# Capture VRAM usage
echo ""
echo "--- GPU Stats ---"
VRAM_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
echo "VRAM: $VRAM_INFO"

# Initialize result JSON
cat > "$RESULT_FILE" <<EOF
{
  "engine": "$ENGINE",
  "config": "$CONFIG",
  "timestamp": "$TIMESTAMP",
  "thread_count": $THREAD_COUNT,
  "model": "$MODEL_PATH",
  "gpu_stats": {
    "vram_info": "$VRAM_INFO"
  },
  "workloads": {}
}
EOF

# Run benchmark workloads via API
run_workload() {
    local name="$1"
    local prompt_tokens="$2"
    local max_tokens="$3"
    local runs="${4:-5}"

    echo ""
    echo "--- Workload: $name (prompt ~${prompt_tokens} tok, gen ${max_tokens} tok, ${runs} runs) ---"

    # Generate a prompt of approximate length
    local prompt
    case "$prompt_tokens" in
        128)  prompt="Write a short story about a robot learning to paint. Be creative and detailed." ;;
        1024) prompt="Write a comprehensive guide about the history of artificial intelligence, covering the following topics in detail: the early pioneers like Alan Turing and John McCarthy, the AI winters and what caused them, the rise of machine learning, deep learning breakthroughs with neural networks, the development of large language models, and future predictions. Include specific dates, names, and technical details. Discuss the ethical implications and societal impact. Cover both narrow AI and the quest for artificial general intelligence. Explain key algorithms like backpropagation, transformers, attention mechanisms, and reinforcement learning. Describe major milestones like IBM's Deep Blue, Google's AlphaGo, and ChatGPT. Discuss the role of compute scaling, data availability, and architectural innovations. Explain the differences between supervised, unsupervised, and self-supervised learning. Cover computer vision, natural language processing, and robotics applications. Discuss AI safety research and alignment problems. Describe the economic impact of AI on various industries." ;;
        4096) prompt="Write an extremely detailed and comprehensive textbook chapter about quantum computing and its relationship with artificial intelligence. Start with the fundamental principles of quantum mechanics including superposition, entanglement, and wave function collapse. Explain how these principles are leveraged in quantum computing through qubits, quantum gates, and quantum circuits. Cover the major quantum computing architectures: superconducting qubits (IBM, Google), trapped ions (IonQ, Honeywell), photonic systems (Xanadu, PsiQuantum), and topological approaches (Microsoft). Discuss quantum error correction, including surface codes, color codes, and the threshold theorem. Explain quantum algorithms in detail: Shor's algorithm for factoring, Grover's search algorithm, quantum approximate optimization algorithm (QAOA), variational quantum eigensolver (VQE), and quantum machine learning algorithms. Cover the quantum advantage debate, including Google's quantum supremacy claim and subsequent discussions. Discuss quantum-classical hybrid approaches and how they relate to near-term quantum computing (NISQ era). Explain how quantum computing could transform AI through quantum neural networks, quantum kernel methods, quantum reinforcement learning, and quantum generative models. Cover the hardware challenges including decoherence times, gate fidelities, connectivity constraints, and scaling issues. Discuss the software stack: quantum programming languages (Qiskit, Cirq, Q#, Pennylane), quantum simulators, and quantum cloud services. Explain quantum cryptography and post-quantum cryptography, including lattice-based, hash-based, and code-based approaches. Cover the economic landscape of quantum computing: major players, startup ecosystem, government investments, and market projections. Discuss quantum sensing and quantum communication as related quantum technologies. Explain the concept of quantum internet and quantum key distribution. Cover recent developments in quantum error correction, including demonstrations by Google, IBM, and others. Discuss the timeline predictions for fault-tolerant quantum computing and what milestones need to be achieved. Explain the relationship between computational complexity theory and quantum computing, including BQP, QMA, and other complexity classes. Cover philosophical implications of quantum computing for our understanding of computation and physics. Include detailed mathematical formulations where appropriate, using Dirac notation and density matrices. Discuss decoherence mechanisms and how different qubit types handle environmental noise. Explain the DiVincenzo criteria for quantum computing. Cover adiabatic quantum computing and quantum annealing (D-Wave). Discuss topological quantum computing and Majorana fermions. Explain quantum walks and their applications. Cover quantum simulation of chemical and material systems. Discuss the role of classical optimization in variational quantum algorithms. Explain barren plateaus and trainability issues in parameterized quantum circuits. Cover quantum tomography and quantum benchmarking. Now also discuss how all of these topics connect to modern AI: could quantum computers train better language models? What about quantum sampling for generative models? How might quantum sensing improve data collection for AI systems? What are the implications of quantum computing for AI safety and alignment?" ;;
    esac

    local results=()
    for i in $(seq 1 "$runs"); do
        local start_time
        start_time=$(date +%s%3N)

        local response
        response=$(curl -sf http://localhost:8080/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"qwen3-next\",
                \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
                \"max_tokens\": $max_tokens,
                \"temperature\": 0.7,
                \"stream\": false
            }" 2>/dev/null) || { echo "  Run $i: FAILED"; continue; }

        local end_time
        end_time=$(date +%s%3N)
        local wall_ms=$((end_time - start_time))

        # Extract usage stats from response
        local prompt_tok completion_tok
        prompt_tok=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('prompt_tokens',0))" 2>/dev/null || echo "0")
        completion_tok=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")

        # Compute tokens/sec
        local tps="0"
        if [[ "$completion_tok" -gt 0 && "$wall_ms" -gt 0 ]]; then
            tps=$(python3 -c "print(round($completion_tok / ($wall_ms / 1000), 2))")
        fi

        echo "  Run $i: ${prompt_tok} pp, ${completion_tok} tg, ${tps} tok/s, ${wall_ms}ms total"
        results+=("{\"run\": $i, \"prompt_tokens\": $prompt_tok, \"completion_tokens\": $completion_tok, \"tokens_per_sec\": $tps, \"wall_ms\": $wall_ms}")
    done

    # Add results to JSON
    local runs_json
    runs_json=$(printf '%s\n' "${results[@]}" | python3 -c "
import sys, json
runs = [json.loads(line) for line in sys.stdin if line.strip()]
# Skip first run (warmup), compute stats on rest
if len(runs) > 1:
    data_runs = runs[1:]
else:
    data_runs = runs
tps_values = [r['tokens_per_sec'] for r in data_runs if r['tokens_per_sec'] > 0]
if tps_values:
    import statistics
    mean_tps = round(statistics.mean(tps_values), 2)
    stddev_tps = round(statistics.stdev(tps_values), 2) if len(tps_values) > 1 else 0
else:
    mean_tps = 0
    stddev_tps = 0
print(json.dumps({
    'runs': runs,
    'mean_tps': mean_tps,
    'stddev_tps': stddev_tps,
    'num_measured': len(data_runs)
}))
")

    # Merge workload results into the result file
    python3 << PYEOF
import json
with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)
data['workloads']['$name'] = json.loads('''$runs_json''')
with open('$RESULT_FILE', 'w') as f:
    json.dump(data, f, indent=2)
PYEOF

    echo "  Results saved."
}

# Run workloads
run_workload "short_prompt"  128  256  5
run_workload "medium_prompt" 1024 256  5
run_workload "long_prompt"   4096 256  5

# Multi-turn workload — uses temp files to avoid shell quoting issues with LLM output
echo ""
echo "--- Workload: multi_turn (5 turns, ~256 tok each) ---"
MULTI_RESULTS=()
MESSAGES_FILE=$(mktemp /tmp/bench-messages-XXXXXX.json)
RESPONSE_FILE=$(mktemp /tmp/bench-response-XXXXXX.json)
trap "rm -f $MESSAGES_FILE $RESPONSE_FILE" EXIT

for run in $(seq 1 5); do
    echo '[{"role":"user","content":"Tell me about the history of computing. Start from the beginning."}]' > "$MESSAGES_FILE"
    TOTAL_TOKENS=0
    TOTAL_MS=0

    for turn in $(seq 1 5); do
        START_MS=$(date +%s%3N)

        # Build request body from messages file
        python3 -c "
import json
with open('$MESSAGES_FILE') as f:
    msgs = json.load(f)
print(json.dumps({
    'model': 'qwen3-next',
    'messages': msgs,
    'max_tokens': 256,
    'temperature': 0.7,
    'stream': False
}))
" | curl -sf http://localhost:8080/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d @- > "$RESPONSE_FILE" 2>/dev/null || { echo "  Run $run Turn $turn: FAILED"; break; }

        END_MS=$(date +%s%3N)
        WALL=$((END_MS - START_MS))
        TOTAL_MS=$((TOTAL_MS + WALL))

        COMP_TOKENS=$(python3 -c "import json; d=json.load(open('$RESPONSE_FILE')); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")
        TOTAL_TOKENS=$((TOTAL_TOKENS + COMP_TOKENS))

        # Append assistant reply + next user message via Python (safe JSON handling)
        python3 -c "
import json
with open('$RESPONSE_FILE') as f:
    resp = json.load(f)
with open('$MESSAGES_FILE') as f:
    msgs = json.load(f)
content = resp['choices'][0]['message']['content'][:200]
msgs.append({'role': 'assistant', 'content': content})
msgs.append({'role': 'user', 'content': 'Continue. Tell me more about what happened next.'})
with open('$MESSAGES_FILE', 'w') as f:
    json.dump(msgs, f)
" 2>/dev/null
    done

    TPS="0"
    if [[ "$TOTAL_TOKENS" -gt 0 && "$TOTAL_MS" -gt 0 ]]; then
        TPS=$(python3 -c "print(round($TOTAL_TOKENS / ($TOTAL_MS / 1000), 2))")
    fi
    echo "  Run $run: ${TOTAL_TOKENS} total tokens, ${TPS} avg tok/s, ${TOTAL_MS}ms"
    MULTI_RESULTS+=("{\"run\": $run, \"total_tokens\": $TOTAL_TOKENS, \"tokens_per_sec\": $TPS, \"wall_ms\": $TOTAL_MS}")
done

# Save multi-turn results
MULTI_JSON=$(printf '%s\n' "${MULTI_RESULTS[@]}" | python3 -c "
import sys, json, statistics
runs = [json.loads(line) for line in sys.stdin if line.strip()]
data_runs = runs[1:] if len(runs) > 1 else runs
tps = [r['tokens_per_sec'] for r in data_runs if r['tokens_per_sec'] > 0]
mean_tps = round(statistics.mean(tps), 2) if tps else 0
stddev_tps = round(statistics.stdev(tps), 2) if len(tps) > 1 else 0
print(json.dumps({'runs': runs, 'mean_tps': mean_tps, 'stddev_tps': stddev_tps, 'num_measured': len(data_runs)}))
")
python3 << PYEOF
import json
with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)
data['workloads']['multi_turn'] = json.loads('''$MULTI_JSON''')
with open('$RESULT_FILE', 'w') as f:
    json.dump(data, f, indent=2)
PYEOF

# Capture final GPU stats
FINAL_VRAM=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
python3 -c "
import json
with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)
data['gpu_stats']['vram_after_bench'] = '$FINAL_VRAM'
with open('$RESULT_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"

echo ""
echo "=========================================="
echo " Benchmark complete!"
echo " Results: $RESULT_FILE"
echo "=========================================="

# Cleanup
echo "Stopping container..."
docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1
echo "Done."
