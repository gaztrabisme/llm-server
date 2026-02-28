#!/bin/bash
# Run lm-eval-harness benchmarks against a local llama.cpp server
# Usage: ./scripts/run-lm-eval.sh <quant-label> [--limit N]
#
# Prerequisites:
# - Server must be running on localhost:8080
# - .venv/ must have lm-eval[api] and transformers installed
# - Patched openai_completions.py for llama.cpp logprobs format
#
# Quant labels: Q8_0, UD-Q4_K_XL, Q4_K_M-bartowski, Q4_K_M-AesSedai

set -euo pipefail

QUANT_LABEL="${1:?Usage: $0 <quant-label> [--limit N]}"
LIMIT="${2:-}"  # Optional --limit flag value

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv"
OUTPUT_DIR="$PROJECT_DIR/benchmarks/evals/$QUANT_LABEL"

# Activate venv
source "$VENV/bin/activate"
export HF_ALLOW_CODE_EVAL=1

# Common model args
MODEL_ARGS="model=Qwen/Qwen3.5-35B-A3B,base_url=http://localhost:8080/v1/completions,tokenized_requests=False,num_concurrent=4"

# Benchmarks to run
# Group 1: Loglikelihood tasks (multiple choice)
LL_TASKS="arc_challenge,hellaswag,mmlu_pro"
# Group 2: Generation tasks (code + math)
GEN_TASKS="gsm8k,humaneval,mbpp"

# Build limit args
LIMIT_ARG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

echo "=========================================="
echo " lm-eval-harness benchmark suite"
echo " Quant: $QUANT_LABEL"
echo " Output: $OUTPUT_DIR"
echo " Limit: ${LIMIT:-full dataset}"
echo " Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Check server health
if ! curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "ERROR: Server not healthy at localhost:8080"
    exit 1
fi

echo ""
echo "--- Phase 1: Loglikelihood tasks ($LL_TASKS) ---"
echo "Started: $(date '+%H:%M:%S')"

lm_eval --tasks "$LL_TASKS" \
    --model local-completions \
    --model_args "$MODEL_ARGS" \
    --num_fewshot 0 \
    $LIMIT_ARG \
    --output_path "$OUTPUT_DIR/loglikelihood" \
    2>&1 | tee "$OUTPUT_DIR/loglikelihood.log"

echo ""
echo "--- Phase 2: Generation tasks ($GEN_TASKS) ---"
echo "Started: $(date '+%H:%M:%S')"

lm_eval --tasks "$GEN_TASKS" \
    --model local-completions \
    --model_args "$MODEL_ARGS" \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    $LIMIT_ARG \
    --output_path "$OUTPUT_DIR/generation" \
    2>&1 | tee "$OUTPUT_DIR/generation.log"

echo ""
echo "=========================================="
echo " COMPLETE: $QUANT_LABEL"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Results: $OUTPUT_DIR/"
echo "=========================================="
