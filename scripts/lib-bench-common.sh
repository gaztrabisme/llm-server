#!/usr/bin/env bash
# =============================================================================
# lib-bench-common.sh — Shared functions for benchmark scripts
# =============================================================================
# Source this from bench.sh and bench-matrix.sh:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib-bench-common.sh"
#
# Provides:
#   parse_env_file <file> [thread_override]  → sets SERVER_ARGS, MODEL_PATH, DOCKER_ENV_ARGS, CFG_*
#   start_server <container> <image> <model_dir> [extra_docker_env] → starts Docker container
#   wait_for_health <url> [max_wait]         → polls /health endpoint
#   capture_vram                              → prints "used_mb,total_mb,util_pct"
#   capture_vram_json                         → prints JSON: {"used_mb":N,"total_mb":N,"utilization_pct":N}
#   validate_model <model_path> <model_dir> <env_file>  → exits if model not found
# =============================================================================

# Parse a LLAMA_ARG_* env file into SERVER_ARGS and related variables.
# Usage: parse_env_file <env_file> [thread_override]
# Sets: SERVER_ARGS, DOCKER_ENV_ARGS, MODEL_PATH, CFG_BATCH_SIZE, CFG_UBATCH_SIZE,
#       CFG_CTK, CFG_CTV, CFG_CTX_SIZE
parse_env_file() {
    local env_file="$1"
    local thread_override="${2:-}"

    SERVER_ARGS=""
    DOCKER_ENV_ARGS=""
    MODEL_PATH=""
    CFG_BATCH_SIZE=""
    CFG_UBATCH_SIZE=""
    CFG_CTK=""
    CFG_CTV=""
    CFG_CTX_SIZE=""

    # Parse extra args (ik-llama compat)
    local extra_args=""
    if grep -q "^IK_EXTRA_ARGS=" "$env_file" 2>/dev/null; then
        extra_args=$(grep "^IK_EXTRA_ARGS=" "$env_file" | cut -d= -f2-)
    fi

    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" =~ ^# || "$key" == "IK_EXTRA_ARGS" ]] && continue
        case "$key" in
            LLAMA_ARG_MODEL)          SERVER_ARGS+=" -m $value"; MODEL_PATH="$value" ;;
            LLAMA_ARG_CTX_SIZE)       SERVER_ARGS+=" -c $value"; CFG_CTX_SIZE="$value" ;;
            LLAMA_ARG_N_GPU_LAYERS)   SERVER_ARGS+=" -ngl $value" ;;
            LLAMA_ARG_OVERRIDE_TENSOR) SERVER_ARGS+=" -ot $value" ;;
            LLAMA_ARG_CPU_MOE)        [[ "$value" == "true" ]] && SERVER_ARGS+=" -cmoe" ;;
            LLAMA_ARG_FLASH_ATTN)     [[ "$value" == "true" ]] && SERVER_ARGS+=" -fa on" ;;
            LLAMA_ARG_THREADS)
                if [[ -n "$thread_override" ]]; then
                    SERVER_ARGS+=" -t $thread_override"
                else
                    SERVER_ARGS+=" -t $value"
                fi
                ;;
            LLAMA_ARG_BATCH_SIZE)     SERVER_ARGS+=" -b $value"; CFG_BATCH_SIZE="$value" ;;
            LLAMA_ARG_UBATCH_SIZE)    SERVER_ARGS+=" -ub $value"; CFG_UBATCH_SIZE="$value" ;;
            LLAMA_ARG_NO_MMAP)        [[ "$value" == "true" ]] && SERVER_ARGS+=" --no-mmap" ;;
            LLAMA_ARG_MLOCK)          [[ "$value" == "true" ]] && SERVER_ARGS+=" --mlock" ;;
            LLAMA_ARG_JINJA)          [[ "$value" == "true" ]] && SERVER_ARGS+=" --jinja" ;;
            LLAMA_ARG_HOST)           SERVER_ARGS+=" --host $value" ;;
            LLAMA_ARG_PORT)           SERVER_ARGS+=" --port $value" ;;
            LLAMA_ARG_CACHE_TYPE_K)   SERVER_ARGS+=" -ctk $value"; CFG_CTK="$value" ;;
            LLAMA_ARG_CACHE_TYPE_V)   SERVER_ARGS+=" -ctv $value"; CFG_CTV="$value" ;;
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
    done < "$env_file"

    [[ -n "$extra_args" ]] && SERVER_ARGS+=" $extra_args" || true
}

# Start a Docker container for llama-server.
# Usage: start_server <container_name> <image> <model_dir> [docker_env_args]
# Uses SERVER_ARGS from parse_env_file.
start_server() {
    local container="$1"
    local image="$2"
    local model_dir="$3"
    local docker_env_args="${4:-}"

    docker run -d \
        --name "$container" \
        --gpus all \
        --ipc host \
        -v "$model_dir:/models:ro" \
        -p 8080:8080 \
        $docker_env_args \
        "$image" \
        $SERVER_ARGS
}

# Wait for /health endpoint to return OK.
# Usage: wait_for_health <base_url> [max_wait_seconds]
wait_for_health() {
    local url="$1"
    local max_wait="${2:-600}"
    local waited=0

    echo "Waiting for server to be ready..."
    while ! curl -sf "${url}/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [[ $waited -ge $max_wait ]]; then
            echo "Server did not become healthy within ${max_wait}s"
            return 1
        fi
        printf "\r  Waiting... %ds / %ds" "$waited" "$max_wait"
    done
    echo ""
    echo "Server healthy!"
}

# Capture VRAM as CSV: "used_mb,total_mb,util_pct"
capture_vram() {
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || echo "N/A"
}

# Capture VRAM as JSON object.
capture_vram_json() {
    local raw
    raw=$(capture_vram)
    if [[ "$raw" == "N/A" ]]; then
        echo '{"used_mb": 0, "total_mb": 0, "utilization_pct": 0}'
    else
        python3 -c "
parts = '$raw'.split(',')
print('{\"used_mb\": %s, \"total_mb\": %s, \"utilization_pct\": %s}' % (parts[0].strip(), parts[1].strip(), parts[2].strip()))
"
    fi
}

# Validate model file exists on host.
# Usage: validate_model <model_path> <model_dir> <env_file>
# Exits with error if model not found.
validate_model() {
    local model_path="$1"
    local model_dir="$2"
    local env_file="$3"

    if [[ -z "$model_path" ]]; then
        echo "No LLAMA_ARG_MODEL in $env_file"
        exit 1
    fi

    local host_path="${model_dir}/${model_path#/models/}"
    if [[ ! -f "$host_path" ]]; then
        echo "Model not found: $host_path"
        echo "Check LLAMA_ARG_MODEL in $env_file"
        exit 1
    fi
}
