#!/usr/bin/env bash
# Run llama-eval across multiple configs and datasets.
# Each config starts a Docker container, runs all datasets, then stops.
set -euo pipefail

MODELS_DIR="/home/bppc/projects/llm-server/models"
RESULTS_DIR="benchmarks/llama-eval"
EVAL_SCRIPT="scripts/llama-eval.py"
PYTHON=".venv/bin/python3"
PORT=8082
CONTAINER="llama-eval"

N_CASES_GSM8K=100
N_CASES_AIME2025=0   # 0 = all (~30)
N_CASES_GPQA=0       # 0 = all (~198)

DATASETS=(gsm8k aime2025 gpqa)
SAMPLING="--temperature 1.0 --top-k 20 --top-p 0.95"

mkdir -p "$RESULTS_DIR"

declare -A CONFIG_MODEL CONFIG_IMAGE CONFIG_FLAGS CONFIG_LABEL

CONFIG_LABEL[27b-iq3-mtp]="27B IQ3+MTP (dream)"
CONFIG_MODEL[27b-iq3-mtp]="Qwen3.6-27B-MTP-UD-IQ3_XXS.gguf"
CONFIG_IMAGE[27b-iq3-mtp]="llm-server/llama-cpp:s015-mtp"
CONFIG_FLAGS[27b-iq3-mtp]="-c 32768 --fit on -ctk q8_0 -ctv q8_0 --spec-type mtp -ctkd q8_0 -ctvd q8_0 -fa on -t 20 --no-mmap --jinja --host 0.0.0.0 -np 1"

CONFIG_LABEL[27b-iq3]="27B IQ3 (no MTP)"
CONFIG_MODEL[27b-iq3]="Qwen3.6-27B-UD-IQ3_XXS.gguf"
CONFIG_IMAGE[27b-iq3]="llm-server/llama-cpp:b8943"
CONFIG_FLAGS[27b-iq3]="-c 32768 --fit on -ctk q8_0 -ctv q8_0 -fa on -t 20 --no-mmap --jinja --host 0.0.0.0 -np 1"

CONFIG_LABEL[27b-q2-mtp]="27B Q2_K_XL+MTP"
CONFIG_MODEL[27b-q2-mtp]="Qwen3.6-27B-MTP-UD-Q2_K_XL.gguf"
CONFIG_IMAGE[27b-q2-mtp]="llm-server/llama-cpp:s015-mtp"
CONFIG_FLAGS[27b-q2-mtp]="-c 32768 --fit on -ctk q8_0 -ctv q8_0 --spec-type mtp -ctkd q8_0 -ctvd q8_0 -fa on -t 20 --no-mmap --jinja --host 0.0.0.0 -np 1"

CONFIG_LABEL[35b-q4-prod]="35B MoE UD-Q4_K_XL"
CONFIG_MODEL[35b-q4-prod]="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
CONFIG_IMAGE[35b-q4-prod]="llm-server/llama-cpp:b8943"
CONFIG_FLAGS[35b-q4-prod]="-c 32768 --fit on --fit-target 0 -ctk q8_0 -ctv q8_0 -fa on -t 20 --no-mmap --jinja --host 0.0.0.0 -np 1"

CONFIG_LABEL[35b-q8-ref]="35B MoE Q8_0 (ref)"
CONFIG_MODEL[35b-q8-ref]="Qwen3.6-35B-A3B-Q8_0.gguf"
CONFIG_IMAGE[35b-q8-ref]="llm-server/llama-cpp:b8943"
CONFIG_FLAGS[35b-q8-ref]="-c 32768 --fit on -ctk q8_0 -ctv q8_0 -fa on -t 20 --no-mmap --jinja --host 0.0.0.0 -np 1"

# Order: dream config first (fastest), then others
CONFIGS=(27b-iq3-mtp 27b-iq3 27b-q2-mtp 35b-q4-prod 35b-q8-ref)

start_server() {
  local config="$1"
  local model="${CONFIG_MODEL[$config]}"
  local image="${CONFIG_IMAGE[$config]}"
  local flags="${CONFIG_FLAGS[$config]}"

  echo ""
  echo "========================================"
  echo "  Starting: ${CONFIG_LABEL[$config]}"
  echo "  Model: $model"
  echo "  Image: $image"
  echo "========================================"

  docker stop "$CONTAINER" 2>/dev/null || true
  docker rm "$CONTAINER" 2>/dev/null || true
  sleep 2

  # shellcheck disable=SC2086
  docker run --gpus all -d --name "$CONTAINER" \
    -v "$MODELS_DIR:/models" \
    -p "$PORT:8080" \
    "$image" \
    -m "/models/$model" \
    $flags

  echo "Waiting for server..."
  local attempts=0
  until curl -s "http://localhost:$PORT/health" | grep -q ok; do
    sleep 3
    attempts=$((attempts + 1))
    if [ "$attempts" -gt 40 ]; then
      echo "TIMEOUT waiting for server"
      docker logs "$CONTAINER" 2>&1 | tail -5
      return 1
    fi
  done
  echo "Server ready"
}

run_dataset() {
  local config="$1"
  local dataset="$2"
  local outfile="$RESULTS_DIR/s016-${config}-${dataset}.json"

  if [ -f "$outfile" ]; then
    local done_count
    done_count=$(python3 -c "
import json
d = json.load(open('$outfile'))
ts = d['task_states']
done = sum(1 for c in ts['cases'].values() if c.get('status') and c['status'] != 'pending')
total = len(ts['cases'])
print(f'{done}/{total}')
" 2>/dev/null || echo "0/0")
    echo "  $dataset: existing results ($done_count), using --resume"
    RESUME_FLAG="--resume"
  else
    RESUME_FLAG=""
  fi

  local n_cases_flag=""
  case "$dataset" in
    gsm8k) [ "$N_CASES_GSM8K" -gt 0 ] && n_cases_flag="--n_cases $N_CASES_GSM8K" ;;
    aime2025) [ "$N_CASES_AIME2025" -gt 0 ] && n_cases_flag="--n_cases $N_CASES_AIME2025" ;;
    gpqa) [ "$N_CASES_GPQA" -gt 0 ] && n_cases_flag="--n_cases $N_CASES_GPQA" ;;
  esac

  echo ""
  echo "  Running $dataset on $config..."
  # shellcheck disable=SC2086
  $PYTHON "$EVAL_SCRIPT" \
    --server "http://localhost:$PORT" \
    --dataset "$dataset" \
    $n_cases_flag \
    --n_predict 32768 \
    $SAMPLING \
    --grader-type regex \
    --threads 1 \
    --model "$config" \
    --output "$outfile" \
    $RESUME_FLAG 2>&1 | tail -5

  echo "  $dataset done: $outfile"
}

# Main loop
for config in "${CONFIGS[@]}"; do
  start_server "$config" || { echo "FAILED to start $config, skipping"; continue; }

  for dataset in "${DATASETS[@]}"; do
    run_dataset "$config" "$dataset"
  done

  docker stop "$CONTAINER" 2>/dev/null || true
  docker rm "$CONTAINER" 2>/dev/null || true
done

echo ""
echo "========================================"
echo "  ALL EVALS COMPLETE"
echo "========================================"
echo ""
echo "Results in $RESULTS_DIR/"
ls -la "$RESULTS_DIR"/s016-*.json
