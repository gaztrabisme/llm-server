#!/bin/bash
# DEPRECATED: Use `llm-bench eval` instead (Session 009).
#   pip install -e . && llm-bench eval --help
#
# Run lm-eval-harness benchmarks for all 4 quants sequentially
# Each quant: start server → wait for health → run evals → stop server
#
# IMPORTANT: Uses generation-based (generate_until) benchmarks ONLY.
# llama.cpp does NOT support echo=true for prompt logprobs, so
# loglikelihood-based tasks (arc_challenge, hellaswag, winogrande) produce
# invalid results (all scores identical at random chance). See S008 notes.
#
# Usage: ./scripts/run-eval-suite.sh [quant]
#   No args: run all 4 quants
#   quant: run only specified quant (Q8_0, UD-Q4_K_XL, Q4_K_M, AesSedai)
#
# Prerequisites:
# - .venv/ with lm-eval[api], transformers installed
# - Docker image llm-server/llama-cpp:latest-fit available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv"
MODELS_DIR="/home/bppc/projects/llm-server/models"
DOCKER_IMAGE="llm-server/llama-cpp:latest-fit"
CONTAINER_NAME="llm-eval-server"
PORT=8080

# Activate venv
source "$VENV/bin/activate"
export HF_ALLOW_CODE_EVAL=1

# Common model args for lm-eval
MODEL_ARGS_BASE="model=Qwen/Qwen3.5-35B-A3B,base_url=http://localhost:${PORT}/v1/completions,tokenized_requests=False,num_concurrent=1"

# Common server args
COMMON_ARGS="-c 65536 --fit on -fa on -t 20 --no-mmap --jinja --host 0.0.0.0 --port 8080 -ctk q8_0 -ctv q8_0"

# Quant configs
declare -A QUANT_MODELS=(
    ["Q8_0"]="/models/Qwen3.5-35B-A3B-Q8_0.gguf"
    ["UD-Q4_K_XL"]="/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    ["Q4_K_M"]="/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
    ["AesSedai"]="/models/aessedai/Q4_K_M/Qwen3.5-35B-A3B-Q4_K_M-00001-of-00002.gguf"
)

# Generation-only benchmarks (llama.cpp doesn't support loglikelihood via API)
# gpqa_main_generative_n_shot: graduate-level QA, generation (flexible-extract works)
# gsm8k: math reasoning, generation (flexible-extract works)
#
# NOTE: arc_challenge_chat/arc_challenge_llama don't work with local-completions:
#   - arc_challenge_chat: stop sequence "." fires immediately
#   - arc_challenge_llama: model generates <think> tags, never reaches answer in 100 tokens
# Loglikelihood tasks (arc_challenge, hellaswag, winogrande) broken: llama.cpp
# doesn't support echo=true for prompt logprobs, all scores identical at random chance.
GEN_TASKS="gpqa_main_generative_n_shot,gsm8k"
GEN_LIMIT=200

start_server() {
    local label="$1"
    local model_path="$2"

    echo "--- Starting server for $label ---"

    # Stop any existing server
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    sleep 2

    # Build volume mounts
    local volumes="-v $MODELS_DIR:/models:ro"

    # Start server
    docker run -d --gpus all --ipc host \
        --name "$CONTAINER_NAME" \
        -p ${PORT}:8080 \
        $volumes \
        "$DOCKER_IMAGE" \
        -m "$model_path" $COMMON_ARGS

    # Wait for health
    echo "Waiting for server..."
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "Server healthy after ${i}s"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Server failed to start"
    return 1
}

run_evals() {
    local label="$1"
    local output_dir="$PROJECT_DIR/benchmarks/evals/gen-v2/$label"
    mkdir -p "$output_dir"

    echo ""
    echo "=== Generation evals for $label ==="
    echo "Tasks: $GEN_TASKS"
    echo "Limit: $GEN_LIMIT"
    echo "Started: $(date '+%H:%M:%S')"

    lm_eval --tasks "$GEN_TASKS" \
        --model local-completions \
        --model_args "$MODEL_ARGS_BASE" \
        --num_fewshot 0 \
        --limit "$GEN_LIMIT" \
        --output_path "$output_dir" \
        2>&1 | tee "$output_dir/eval.log" | tail -20

    echo ""
    echo "=== $label complete at $(date '+%H:%M:%S') ==="
}

# Determine which quants to run
if [[ $# -gt 0 ]]; then
    QUANTS=("$1")
else
    QUANTS=("Q8_0" "UD-Q4_K_XL" "Q4_K_M" "AesSedai")
fi

echo "=========================================="
echo " lm-eval-harness Generation Suite v2"
echo " Quants: ${QUANTS[*]}"
echo " Tasks: $GEN_TASKS"
echo " Limit: $GEN_LIMIT per task"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

for quant in "${QUANTS[@]}"; do
    model_path="${QUANT_MODELS[$quant]}"

    echo ""
    echo ">>>>>>>>>> $quant <<<<<<<<<<"
    start_server "$quant" "$model_path"
    run_evals "$quant"
done

echo ""
echo "=========================================="
echo " ALL COMPLETE"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
