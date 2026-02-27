#!/usr/bin/env bash
# =============================================================================
# bench-matrix.sh — Universal matrix benchmark runner for llama.cpp
# =============================================================================
#
# Usage:
#   ./bench-matrix.sh --config <env-file> [options]
#   ./bench-matrix.sh --quant Q4_K_M --kv q8_0 --batch none --ctx 32768 [options]
#
# Options:
#   --config <file>    Path to .env config file (mutually exclusive with axis flags)
#   --quant <name>     Quantization: Q4_K_M | MXFP4_MOE | UD-Q4_K_XL | Q8_0
#   --kv <type>        KV cache type: f16 | q8_0 | q4_0 | asym
#   --batch <type>     Batch config: none | large | asym
#   --ctx <size>       Context length: 4096 | 8192 | 16384 | 32768 | 65536
#   --image <name>     Docker image override (default: llm-server/llama-cpp:latest-fit)
#   --threads <n>      Thread count override (default: 20)
#   --quick            Quick mode: skip long PP lengths, 3 runs instead of 5
#   --tg-only          Skip PP measurement, only measure TG
#   --pp-only          Skip TG measurement, only measure PP
#   --dry-run          Print what would run without executing
#   --help             Show this help
#
# Axis flag mode:
#   When using --quant/--kv/--batch/--ctx, the script constructs the server args
#   automatically by looking up the config from configs/llama-cpp-s007-*.env
#   matching the requested axis values. If no matching config exists, it builds
#   the args from a template.
#
# Config file mode:
#   When using --config, the script reads the env file directly (same format as
#   bench.sh). This provides backward compatibility with existing configs.
#
# Output:
#   Results saved to: benchmarks/matrix/{quant}-{kv}-{batch}-{ctx}-{timestamp}.json
#   One compact summary line printed to stdout per measurement.
#
# Examples:
#   # Run full benchmark with existing config
#   ./bench-matrix.sh --config configs/llama-cpp-s007-e2-batch-none.env
#
#   # Run matrix point by axis values
#   ./bench-matrix.sh --quant Q4_K_M --kv q8_0 --batch none --ctx 32768
#
#   # Quick mode for community reproduction
#   ./bench-matrix.sh --config configs/llama-cpp-s007-e1-kvq8-ctx32k.env --quick
#
#   # PP-only measurement
#   ./bench-matrix.sh --config configs/llama-cpp-s007-e2-batch-large.env --pp-only
#
# =============================================================================
#
# SPEC — Implementation notes for the developer:
#
# 1. ARGUMENT PARSING
#    - Parse all flags into variables: CONFIG_FILE, QUANT, KV, BATCH, CTX,
#      IMAGE, THREADS, QUICK, TG_ONLY, PP_ONLY, DRY_RUN
#    - Validate: either --config or all four axis flags must be provided
#    - If axis flags used, resolve to a config file path:
#      configs/llama-cpp-s007-e{N}-{variant}.env
#      If no matching file, error out (don't auto-generate configs)
#
# 2. CONFIG PARSING
#    - Reuse the same env file format as bench.sh (LLAMA_ARG_* keys)
#    - Parse into SERVER_ARGS string for docker run
#    - New keys to support:
#      LLAMA_ARG_FIT_TARGET — maps to --fit-target <value>
#      LLAMA_ARG_MMPROJ — maps to --mmproj <path>
#    - Extract MODEL_PATH for host-side validation
#
# 3. SERVER LIFECYCLE
#    - Start server in Docker container (same as bench.sh)
#    - Wait for /health endpoint (up to 600s for large models)
#    - Run benchmarks
#    - Capture VRAM via nvidia-smi
#    - Stop and remove container
#    - Trap EXIT to ensure cleanup on failure
#
# 4. PP MEASUREMENT
#    - Use /completion endpoint (NOT /v1/chat/completions)
#    - Send raw text prompt of target length with n_predict: 1
#    - Extract timings.prompt_per_second from response
#    - Prompt lengths: 512, 1024, 4096, 16384 (quick mode: 512, 1024 only)
#    - Use a fixed text corpus for consistent prompts:
#      Read first N tokens from benchmarks/perplexity/wikitext-2-raw/wiki.test.raw
#    - 3 runs per length (quick mode: 2 runs), first run is warmup
#    - Record: prompt_tokens, prompt_per_second per run
#
# 5. TG MEASUREMENT
#    - Use /v1/chat/completions endpoint (same as bench.sh)
#    - 4 workloads: short_prompt (128 tok), medium_prompt (1024 tok),
#      long_prompt (4096 tok), multi_turn (5 turns)
#    - 5 runs per workload (quick mode: 3 runs), first run is warmup
#    - Extract timings.predicted_per_second from response when available,
#      fall back to wall-clock calculation
#    - Record: prompt_tokens, completion_tokens, tokens_per_sec, wall_ms per run
#
# 6. VRAM MEASUREMENT
#    - Capture via: nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu
#      --format=csv,noheader,nounits
#    - Capture twice: after server startup (model loaded) and after benchmark complete
#    - Parse into structured JSON: {used_mb, total_mb, utilization_pct}
#
# 7. OUTPUT FORMAT
#    - JSON file at benchmarks/matrix/{label}-{timestamp}.json where label is
#      derived from config file name (strip llama-cpp- prefix and .env suffix)
#    - Structure:
#      {
#        "label": "s007-e1-kvq8-ctx32k",
#        "matrix": {"quant": "Q4_K_M", "kv": "q8_0", "batch": "none", "ctx": 32768},
#        "config_file": "configs/llama-cpp-s007-e1-kvq8-ctx32k.env",
#        "timestamp": "20260301-143000",
#        "server_args": "-m /models/... -c 32768 ...",
#        "docker_image": "llm-server/llama-cpp:latest-fit",
#        "thread_count": 20,
#        "gpu_stats": {
#          "after_load": {"used_mb": 14559, "total_mb": 16384, "utilization_pct": 0},
#          "after_bench": {"used_mb": 14600, "total_mb": 16384, "utilization_pct": 95}
#        },
#        "pp_benchmarks": {
#          "512": {"runs": [...], "mean_tps": 1200.5, "stddev_tps": 15.2},
#          "1024": {"runs": [...], "mean_tps": 1100.3, "stddev_tps": 12.1},
#          "4096": {"runs": [...], "mean_tps": 800.0, "stddev_tps": 20.0},
#          "16384": {"runs": [...], "mean_tps": 500.0, "stddev_tps": 10.0}
#        },
#        "tg_benchmarks": {
#          "short_prompt": {"runs": [...], "mean_tps": 74.7, "stddev_tps": 1.2},
#          "medium_prompt": {"runs": [...], "mean_tps": 72.9, "stddev_tps": 0.8},
#          "long_prompt": {"runs": [...], "mean_tps": 73.7, "stddev_tps": 1.0},
#          "multi_turn": {"runs": [...], "mean_tps": 76.1, "stddev_tps": 0.5}
#        }
#      }
#
# 8. STDOUT SUMMARY
#    - After each measurement, print one compact line:
#      [PP  512] 1200.5 tok/s (±15.2)  VRAM: 14559 MB
#      [PP 1024] 1100.3 tok/s (±12.1)
#      [TG short]  74.7 tok/s (±1.2)
#      [TG medium] 72.9 tok/s (±0.8)
#      ...
#    - At the end, print a one-line summary:
#      RESULT: PP=1200/1100/800/500 TG=74.7/72.9/73.7/76.1 VRAM=14559 MB
#
# 9. BACKWARD COMPATIBILITY
#    - All existing S006 .env files work with --config flag
#    - bench.sh remains untouched (this is a new script, not a replacement)
#    - Output goes to benchmarks/matrix/ (not benchmarks/) to avoid conflicts
#
# 10. QUICK MODE
#    - PP: measure only at 512 and 1024 prompt lengths
#    - TG: 3 runs instead of 5 (1 warmup + 2 measured)
#    - Skip multi_turn workload
#    - Total time: ~10-15 min per config (vs ~25-30 min full)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_DIR/benchmarks/matrix"
MODEL_DIR="$PROJECT_DIR/models"
CONFIG_DIR="$PROJECT_DIR/configs"
WIKI_FILE="$PROJECT_DIR/benchmarks/perplexity/wikitext-2-raw/wiki.test.raw"

# Source shared library
source "$SCRIPT_DIR/lib-bench-common.sh"

# ============================================================================
# 1. ARGUMENT PARSING
# ============================================================================

CONFIG_FILE=""
QUANT=""
KV=""
BATCH=""
CTX=""
IMAGE="llm-server/llama-cpp:latest-fit"
THREADS="20"
QUICK=false
TG_ONLY=false
PP_ONLY=false
DRY_RUN=false

show_help() {
    # Print the usage block (lines 2-51, between the first two separator lines)
    sed -n '2,51p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG_FILE="$2"; shift 2 ;;
        --quant)    QUANT="$2"; shift 2 ;;
        --kv)       KV="$2"; shift 2 ;;
        --batch)    BATCH="$2"; shift 2 ;;
        --ctx)      CTX="$2"; shift 2 ;;
        --image)    IMAGE="$2"; shift 2 ;;
        --threads)  THREADS="$2"; shift 2 ;;
        --quick)    QUICK=true; shift ;;
        --tg-only)  TG_ONLY=true; shift ;;
        --pp-only)  PP_ONLY=true; shift ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --help|-h)  show_help ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

# Validate: either --config or all four axis flags
if [[ -n "$CONFIG_FILE" ]]; then
    if [[ -n "$QUANT" || -n "$KV" || -n "$BATCH" || -n "$CTX" ]]; then
        echo "Error: --config is mutually exclusive with --quant/--kv/--batch/--ctx"
        exit 1
    fi
    # Resolve relative path to absolute
    if [[ ! "$CONFIG_FILE" = /* ]]; then
        CONFIG_FILE="$PROJECT_DIR/$CONFIG_FILE"
    fi
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Config not found: $CONFIG_FILE"
        exit 1
    fi
elif [[ -n "$QUANT" && -n "$KV" && -n "$BATCH" && -n "$CTX" ]]; then
    # Axis flag mode: find matching config file by extracting matrix values inline
    MATCHED_CONFIG=""
    for candidate in "$CONFIG_DIR"/llama-cpp-s007-*.env; do
        [[ -f "$candidate" ]] || continue

        # Extract quant from model filename
        c_model=$(grep "^LLAMA_ARG_MODEL=" "$candidate" 2>/dev/null | cut -d= -f2 || true)
        c_quant=""
        case "$c_model" in
            *Q4_K_M*)     c_quant="Q4_K_M" ;;
            *Q4_K_XL*)    c_quant="UD-Q4_K_XL" ;;
            *MXFP4_MOE*)  c_quant="MXFP4_MOE" ;;
            *Q8_0*)       c_quant="Q8_0" ;;
            *Q4_K_L*)     c_quant="Q4_K_L" ;;
        esac

        # Extract KV cache type
        c_ctk=$(grep "^LLAMA_ARG_CACHE_TYPE_K=" "$candidate" 2>/dev/null | cut -d= -f2 || true)
        c_ctv=$(grep "^LLAMA_ARG_CACHE_TYPE_V=" "$candidate" 2>/dev/null | cut -d= -f2 || true)
        if [[ -z "$c_ctk" && -z "$c_ctv" ]]; then
            c_kv="f16"
        elif [[ "$c_ctk" == "$c_ctv" ]]; then
            c_kv="$c_ctk"
        else
            c_kv="asym"
        fi

        # Extract batch config
        c_bs=$(grep "^LLAMA_ARG_BATCH_SIZE=" "$candidate" 2>/dev/null | cut -d= -f2 || true)
        c_ubs=$(grep "^LLAMA_ARG_UBATCH_SIZE=" "$candidate" 2>/dev/null | cut -d= -f2 || true)
        if [[ -z "$c_bs" ]]; then
            c_batch="none"
        elif [[ -n "$c_ubs" && "$c_bs" -ge 4096 && "$c_bs" == "$c_ubs" ]]; then
            c_batch="large"
        elif [[ -n "$c_ubs" && "$c_bs" -ge 4096 && "$c_ubs" -lt "$c_bs" ]]; then
            c_batch="asym"
        else
            c_batch="custom"
        fi

        # Extract ctx
        c_ctx=$(grep "^LLAMA_ARG_CTX_SIZE=" "$candidate" 2>/dev/null | cut -d= -f2 || true)

        if [[ "$c_quant" == "$QUANT" && "$c_kv" == "$KV" && "$c_batch" == "$BATCH" && "$c_ctx" == "$CTX" ]]; then
            MATCHED_CONFIG="$candidate"
            break
        fi
    done

    if [[ -z "$MATCHED_CONFIG" ]]; then
        echo "Error: No matching config found for --quant $QUANT --kv $KV --batch $BATCH --ctx $CTX"
        echo "Available configs in $CONFIG_DIR/llama-cpp-s007-*.env:"
        for f in "$CONFIG_DIR"/llama-cpp-s007-*.env; do
            [[ -f "$f" ]] && echo "  $(basename "$f")"
        done
        exit 1
    fi
    CONFIG_FILE="$MATCHED_CONFIG"
else
    echo "Error: Must provide either --config <file> or all four axis flags (--quant --kv --batch --ctx)"
    echo "Use --help for usage."
    exit 1
fi

if [[ "$TG_ONLY" == true && "$PP_ONLY" == true ]]; then
    echo "Error: --tg-only and --pp-only are mutually exclusive"
    exit 1
fi

# ============================================================================
# 2. CONFIG PARSING
# ============================================================================

# Parse env file using shared library (sets SERVER_ARGS, MODEL_PATH, DOCKER_ENV_ARGS, CFG_*)
parse_env_file "$CONFIG_FILE" "$THREADS"

# ============================================================================
# 3. EXTRACT MATRIX VALUES FROM CONFIG
# ============================================================================

# Quant: from model filename
MATRIX_QUANT=""
case "$MODEL_PATH" in
    *UD-Q4_K_XL* | *UD_Q4_K_XL*) MATRIX_QUANT="UD-Q4_K_XL" ;;
    *Q4_K_M*)                      MATRIX_QUANT="Q4_K_M" ;;
    *Q4_K_L*)                      MATRIX_QUANT="Q4_K_L" ;;
    *MXFP4_MOE*)                   MATRIX_QUANT="MXFP4_MOE" ;;
    *Q8_0*)                        MATRIX_QUANT="Q8_0" ;;
    *)                             MATRIX_QUANT="unknown" ;;
esac

# KV cache type
if [[ -z "$CFG_CTK" && -z "$CFG_CTV" ]]; then
    MATRIX_KV="f16"
elif [[ "$CFG_CTK" == "$CFG_CTV" ]]; then
    MATRIX_KV="$CFG_CTK"
else
    MATRIX_KV="asym"
fi

# Batch config
if [[ -z "$CFG_BATCH_SIZE" ]]; then
    MATRIX_BATCH="none"
elif [[ "$CFG_BATCH_SIZE" -ge 4096 && "$CFG_BATCH_SIZE" == "$CFG_UBATCH_SIZE" ]]; then
    MATRIX_BATCH="large"
elif [[ "$CFG_BATCH_SIZE" -ge 4096 && -n "$CFG_UBATCH_SIZE" && "$CFG_UBATCH_SIZE" -lt "$CFG_BATCH_SIZE" ]]; then
    MATRIX_BATCH="asym"
else
    MATRIX_BATCH="custom"
fi

# Context size
MATRIX_CTX="${CFG_CTX_SIZE:-32768}"

# Validate model file exists on host (skip in dry-run mode)
if [[ "$DRY_RUN" != true ]]; then
    validate_model "$MODEL_PATH" "$MODEL_DIR" "$CONFIG_FILE"
fi

# Validate wiki test file for PP measurement (skip in dry-run mode)
if [[ "$DRY_RUN" != true && "$TG_ONLY" != true && ! -f "$WIKI_FILE" ]]; then
    echo "Wiki test file not found: $WIKI_FILE"
    echo "Needed for PP measurement. Use --tg-only to skip PP."
    exit 1
fi

# ============================================================================
# 4. DERIVE LABEL AND OUTPUT PATH
# ============================================================================

# Label from config filename: strip path, "llama-cpp-" prefix, and ".env" suffix
CONFIG_BASENAME="$(basename "$CONFIG_FILE" .env)"
LABEL="${CONFIG_BASENAME#llama-cpp-}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BENCHMARK_DIR"
RESULT_FILE="$BENCHMARK_DIR/${LABEL}-${TIMESTAMP}.json"

# Relative config path for JSON output
RELATIVE_CONFIG="${CONFIG_FILE#$PROJECT_DIR/}"

CONTAINER_NAME="llm-bench-matrix-${LABEL}-$$"

# ============================================================================
# 5. DRY RUN
# ============================================================================

if [[ "$DRY_RUN" == true ]]; then
    echo "=== DRY RUN ==="
    echo "Config:     $CONFIG_FILE"
    echo "Label:      $LABEL"
    echo "Matrix:     quant=$MATRIX_QUANT kv=$MATRIX_KV batch=$MATRIX_BATCH ctx=$MATRIX_CTX"
    echo "Image:      $IMAGE"
    echo "Threads:    $THREADS"
    echo "Quick:      $QUICK"
    echo "PP only:    $PP_ONLY"
    echo "TG only:    $TG_ONLY"
    echo "Container:  $CONTAINER_NAME"
    echo "Output:     $RESULT_FILE"
    echo ""
    echo "Docker command:"
    echo "  docker run -d --gpus all --ipc host \\"
    echo "    -v $MODEL_DIR:/models:ro -p 8080:8080 \\"
    [[ -n "$DOCKER_ENV_ARGS" ]] && echo "    $DOCKER_ENV_ARGS \\"
    echo "    $IMAGE$SERVER_ARGS"
    echo ""
    if [[ "$TG_ONLY" != true ]]; then
        if [[ "$QUICK" == true ]]; then
            echo "PP workloads: 512, 1024 (2 runs each, 1 warmup)"
        else
            echo "PP workloads: 512, 1024, 4096, 16384 (3 runs each, 1 warmup)"
        fi
    fi
    if [[ "$PP_ONLY" != true ]]; then
        if [[ "$QUICK" == true ]]; then
            echo "TG workloads: short_prompt, medium_prompt, long_prompt (3 runs each, 1 warmup)"
        else
            echo "TG workloads: short_prompt, medium_prompt, long_prompt, multi_turn (5 runs each, 1 warmup)"
        fi
    fi
    exit 0
fi

# ============================================================================
# 6. TEMP FILES AND CLEANUP TRAP
# ============================================================================

MESSAGES_FILE=$(mktemp /tmp/bench-matrix-messages-XXXXXX.json)
RESPONSE_FILE=$(mktemp /tmp/bench-matrix-response-XXXXXX.json)
PP_RESPONSE_FILE=$(mktemp /tmp/bench-matrix-pp-response-XXXXXX.json)

cleanup() {
    rm -f "$MESSAGES_FILE" "$RESPONSE_FILE" "$PP_RESPONSE_FILE"
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$" 2>/dev/null; then
        echo "Cleaning up container: $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# ============================================================================
# 7. HELPER FUNCTIONS
# ============================================================================

# capture_vram_json is provided by lib-bench-common.sh

compute_stats() {
    # Reads JSON run objects from stdin (one per line), computes mean/stddev
    # skipping the first run (warmup). Outputs JSON with runs, mean_tps, stddev_tps.
    python3 -c "
import sys, json, statistics
runs = [json.loads(line) for line in sys.stdin if line.strip()]
if len(runs) > 1:
    data_runs = runs[1:]
else:
    data_runs = runs
tps_key = 'prompt_per_second' if 'prompt_per_second' in (runs[0] if runs else {}) else 'tokens_per_sec'
tps_values = [r[tps_key] for r in data_runs if r.get(tps_key, 0) > 0]
if tps_values:
    mean_tps = round(statistics.mean(tps_values), 1)
    stddev_tps = round(statistics.stdev(tps_values), 1) if len(tps_values) > 1 else 0.0
else:
    mean_tps = 0.0
    stddev_tps = 0.0
print(json.dumps({
    'runs': runs,
    'mean_tps': mean_tps,
    'stddev_tps': stddev_tps
}))
"
}

save_to_result() {
    # Usage: save_to_result <section> <key> <json_value>
    # section: pp_benchmarks or tg_benchmarks
    # key: e.g. "512" or "short_prompt"
    local section="$1"
    local key="$2"
    local json_value="$3"
    python3 << PYEOF
import json
with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)
if '$section' not in data:
    data['$section'] = {}
data['$section']['$key'] = json.loads('''$json_value''')
with open('$RESULT_FILE', 'w') as f:
    json.dump(data, f, indent=2)
PYEOF
}

# ============================================================================
# 8. START SERVER
# ============================================================================

echo "=========================================="
echo " bench-matrix.sh"
echo " Config: $(basename "$CONFIG_FILE")"
echo " Matrix: quant=$MATRIX_QUANT kv=$MATRIX_KV batch=$MATRIX_BATCH ctx=$MATRIX_CTX"
echo " Image:  $IMAGE"
echo " Mode:   $(if [[ "$PP_ONLY" == true ]]; then echo "PP only"; elif [[ "$TG_ONLY" == true ]]; then echo "TG only"; else echo "PP + TG"; fi)$(if [[ "$QUICK" == true ]]; then echo " (quick)"; fi)"
echo " Output: $RESULT_FILE"
echo "=========================================="
echo ""
echo "Starting server container: $CONTAINER_NAME"
echo "Args:$SERVER_ARGS"
echo ""

start_server "$CONTAINER_NAME" "$IMAGE" "$MODEL_DIR" "$DOCKER_ENV_ARGS"

if ! wait_for_health "http://localhost:8080"; then
    docker logs "$CONTAINER_NAME" 2>&1 | tail -20
    exit 1
fi
echo ""

# Capture VRAM after model load
VRAM_AFTER_LOAD=$(capture_vram_json)
VRAM_USED_MB=$(echo "$VRAM_AFTER_LOAD" | python3 -c "import sys,json; print(json.load(sys.stdin)['used_mb'])")
echo "VRAM after load: ${VRAM_USED_MB} MB"
echo ""

# ============================================================================
# 9. INITIALIZE RESULT JSON
# ============================================================================

python3 << PYEOF
import json
data = {
    "label": "$LABEL",
    "matrix": {
        "quant": "$MATRIX_QUANT",
        "kv": "$MATRIX_KV",
        "batch": "$MATRIX_BATCH",
        "ctx": $MATRIX_CTX
    },
    "config_file": "$RELATIVE_CONFIG",
    "timestamp": "$TIMESTAMP",
    "server_args": "${SERVER_ARGS# }",
    "docker_image": "$IMAGE",
    "thread_count": $THREADS,
    "gpu_stats": {
        "after_load": json.loads('$VRAM_AFTER_LOAD')
    },
    "pp_benchmarks": {},
    "tg_benchmarks": {}
}
with open('$RESULT_FILE', 'w') as f:
    json.dump(data, f, indent=2)
PYEOF

# ============================================================================
# 10. PP MEASUREMENT
# ============================================================================

# Collect summary values for final RESULT line
PP_SUMMARY=()
TG_SUMMARY=()

if [[ "$TG_ONLY" != true ]]; then
    echo "--- PP Benchmarks ---"

    if [[ "$QUICK" == true ]]; then
        PP_LENGTHS=(512 1024)
        PP_RUNS=2
    else
        PP_LENGTHS=(512 1024 4096 16384)
        PP_RUNS=3
    fi

    for pp_len in "${PP_LENGTHS[@]}"; do
        # Read approximate number of characters for target token count
        # ~5 chars per token is a rough approximation for English text
        char_count=$((pp_len * 5))

        # Build PP request JSON via Python for safe escaping
        pp_request_file=$(mktemp /tmp/bench-matrix-pp-request-XXXXXX.json)
        python3 -c "
import json
with open('$WIKI_FILE', 'r') as f:
    text = f.read($char_count)
req = {
    'prompt': text,
    'n_predict': 1,
    'temperature': 0.0,
    'stream': False
}
with open('$pp_request_file', 'w') as f:
    json.dump(req, f)
"

        run_results=()
        for i in $(seq 1 "$PP_RUNS"); do
            curl -sf http://localhost:8080/completion \
                -H "Content-Type: application/json" \
                -d @"$pp_request_file" > "$PP_RESPONSE_FILE" 2>/dev/null || {
                echo "  [PP $pp_len] Run $i: FAILED"
                continue
            }

            # Extract timings from response
            run_data=$(python3 -c "
import json, sys
with open('$PP_RESPONSE_FILE') as f:
    resp = json.load(f)
timings = resp.get('timings', {})
pp_sec = timings.get('prompt_per_second', 0)
prompt_n = timings.get('prompt_n', 0)
print(json.dumps({
    'run': $i,
    'prompt_tokens': prompt_n,
    'prompt_per_second': round(pp_sec, 1)
}))
" 2>/dev/null) || {
                echo "  [PP $pp_len] Run $i: parse error"
                continue
            }

            run_tps=$(echo "$run_data" | python3 -c "import sys,json; print(json.load(sys.stdin)['prompt_per_second'])")
            run_ptok=$(echo "$run_data" | python3 -c "import sys,json; print(json.load(sys.stdin)['prompt_tokens'])")

            if [[ "$i" -eq 1 ]]; then
                echo "  [PP $pp_len] Run $i (warmup): ${run_ptok} tokens, ${run_tps} tok/s"
            else
                echo "  [PP $pp_len] Run $i: ${run_ptok} tokens, ${run_tps} tok/s"
            fi
            run_results+=("$run_data")
        done

        rm -f "$pp_request_file"

        # Compute stats and save
        if [[ ${#run_results[@]} -gt 0 ]]; then
            stats_json=$(printf '%s\n' "${run_results[@]}" | compute_stats)
            save_to_result "pp_benchmarks" "$pp_len" "$stats_json"

            mean_tps=$(echo "$stats_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['mean_tps'])")
            stddev_tps=$(echo "$stats_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['stddev_tps'])")
            echo "[PP $pp_len] ${mean_tps} tok/s (+/-${stddev_tps})"
            PP_SUMMARY+=("$mean_tps")
        else
            echo "[PP $pp_len] NO DATA"
            PP_SUMMARY+=("0")
        fi
        echo ""
    done
fi

# ============================================================================
# 11. TG MEASUREMENT
# ============================================================================

if [[ "$PP_ONLY" != true ]]; then
    echo "--- TG Benchmarks ---"

    if [[ "$QUICK" == true ]]; then
        TG_RUNS=3
    else
        TG_RUNS=5
    fi

    # TG workload function
    run_tg_workload() {
        local name="$1"
        local prompt_tokens="$2"
        local max_tokens="$3"
        local runs="$4"

        # Generate a prompt of approximate length
        local prompt
        case "$prompt_tokens" in
            128)  prompt="Write a short story about a robot learning to paint. Be creative and detailed." ;;
            1024) prompt="Write a comprehensive guide about the history of artificial intelligence, covering the following topics in detail: the early pioneers like Alan Turing and John McCarthy, the AI winters and what caused them, the rise of machine learning, deep learning breakthroughs with neural networks, the development of large language models, and future predictions. Include specific dates, names, and technical details. Discuss the ethical implications and societal impact. Cover both narrow AI and the quest for artificial general intelligence. Explain key algorithms like backpropagation, transformers, attention mechanisms, and reinforcement learning. Describe major milestones like IBM's Deep Blue, Google's AlphaGo, and ChatGPT. Discuss the role of compute scaling, data availability, and architectural innovations. Explain the differences between supervised, unsupervised, and self-supervised learning. Cover computer vision, natural language processing, and robotics applications. Discuss AI safety research and alignment problems. Describe the economic impact of AI on various industries." ;;
            4096) prompt="Write an extremely detailed and comprehensive textbook chapter about quantum computing and its relationship with artificial intelligence. Start with the fundamental principles of quantum mechanics including superposition, entanglement, and wave function collapse. Explain how these principles are leveraged in quantum computing through qubits, quantum gates, and quantum circuits. Cover the major quantum computing architectures: superconducting qubits (IBM, Google), trapped ions (IonQ, Honeywell), photonic systems (Xanadu, PsiQuantum), and topological approaches (Microsoft). Discuss quantum error correction, including surface codes, color codes, and the threshold theorem. Explain quantum algorithms in detail: Shor's algorithm for factoring, Grover's search algorithm, quantum approximate optimization algorithm (QAOA), variational quantum eigensolver (VQE), and quantum machine learning algorithms. Cover the quantum advantage debate, including Google's quantum supremacy claim and subsequent discussions. Discuss quantum-classical hybrid approaches and how they relate to near-term quantum computing (NISQ era). Explain how quantum computing could transform AI through quantum neural networks, quantum kernel methods, quantum reinforcement learning, and quantum generative models. Cover the hardware challenges including decoherence times, gate fidelities, connectivity constraints, and scaling issues. Discuss the software stack: quantum programming languages (Qiskit, Cirq, Q#, Pennylane), quantum simulators, and quantum cloud services. Explain quantum cryptography and post-quantum cryptography, including lattice-based, hash-based, and code-based approaches. Cover the economic landscape of quantum computing: major players, startup ecosystem, government investments, and market projections. Discuss quantum sensing and quantum communication as related quantum technologies. Explain the concept of quantum internet and quantum key distribution. Cover recent developments in quantum error correction, including demonstrations by Google, IBM, and others. Discuss the timeline predictions for fault-tolerant quantum computing and what milestones need to be achieved. Explain the relationship between computational complexity theory and quantum computing, including BQP, QMA, and other complexity classes. Cover philosophical implications of quantum computing for our understanding of computation and physics. Include detailed mathematical formulations where appropriate, using Dirac notation and density matrices. Discuss decoherence mechanisms and how different qubit types handle environmental noise. Explain the DiVincenzo criteria for quantum computing. Cover adiabatic quantum computing and quantum annealing (D-Wave). Discuss topological quantum computing and Majorana fermions. Explain quantum walks and their applications. Cover quantum simulation of chemical and material systems. Discuss the role of classical optimization in variational quantum algorithms. Explain barren plateaus and trainability issues in parameterized quantum circuits. Cover quantum tomography and quantum benchmarking. Now also discuss how all of these topics connect to modern AI: could quantum computers train better language models? What about quantum sampling for generative models? How might quantum sensing improve data collection for AI systems? What are the implications of quantum computing for AI safety and alignment?" ;;
        esac

        # Build request JSON via Python for safe escaping of prompt text
        local request_file
        request_file=$(mktemp /tmp/bench-matrix-tg-request-XXXXXX.json)
        python3 -c "
import json
prompt = '''$prompt'''
req = {
    'model': 'bench',
    'messages': [{'role': 'user', 'content': prompt}],
    'max_tokens': $max_tokens,
    'temperature': 0.7,
    'stream': False
}
with open('$request_file', 'w') as f:
    json.dump(req, f)
" || { echo "  [TG $name] Failed to build request JSON"; rm -f "$request_file"; return; }

        local run_results=()
        for i in $(seq 1 "$runs"); do
            local start_time end_time wall_ms
            start_time=$(date +%s%3N)

            curl -sf http://localhost:8080/v1/chat/completions \
                -H "Content-Type: application/json" \
                -d @"$request_file" > "$RESPONSE_FILE" 2>/dev/null || {
                echo "  [TG $name] Run $i: FAILED"
                continue
            }

            end_time=$(date +%s%3N)
            wall_ms=$((end_time - start_time))

            # Extract stats — prefer timings.predicted_per_second, fall back to wall-clock
            run_data=$(python3 -c "
import json
with open('$RESPONSE_FILE') as f:
    resp = json.load(f)
usage = resp.get('usage', {})
prompt_tok = usage.get('prompt_tokens', 0)
completion_tok = usage.get('completion_tokens', 0)

# Prefer server-side timings if available
timings = resp.get('timings', {})
predicted_ps = timings.get('predicted_per_second', 0)

if predicted_ps and predicted_ps > 0:
    tps = round(predicted_ps, 1)
elif completion_tok > 0 and $wall_ms > 0:
    tps = round(completion_tok / ($wall_ms / 1000), 1)
else:
    tps = 0.0

print(json.dumps({
    'run': $i,
    'prompt_tokens': prompt_tok,
    'completion_tokens': completion_tok,
    'tokens_per_sec': tps,
    'wall_ms': $wall_ms
}))
" 2>/dev/null) || {
                echo "  [TG $name] Run $i: parse error"
                continue
            }

            local run_tps run_ptok run_ctok
            run_tps=$(echo "$run_data" | python3 -c "import sys,json; print(json.load(sys.stdin)['tokens_per_sec'])")
            run_ptok=$(echo "$run_data" | python3 -c "import sys,json; print(json.load(sys.stdin)['prompt_tokens'])")
            run_ctok=$(echo "$run_data" | python3 -c "import sys,json; print(json.load(sys.stdin)['completion_tokens'])")

            if [[ "$i" -eq 1 ]]; then
                echo "  [TG $name] Run $i (warmup): ${run_ptok}pp ${run_ctok}tg ${run_tps} tok/s ${wall_ms}ms"
            else
                echo "  [TG $name] Run $i: ${run_ptok}pp ${run_ctok}tg ${run_tps} tok/s ${wall_ms}ms"
            fi
            run_results+=("$run_data")
        done

        # Compute stats and save
        if [[ ${#run_results[@]} -gt 0 ]]; then
            local stats_json mean_tps stddev_tps
            stats_json=$(printf '%s\n' "${run_results[@]}" | compute_stats)
            save_to_result "tg_benchmarks" "$name" "$stats_json"

            mean_tps=$(echo "$stats_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['mean_tps'])")
            stddev_tps=$(echo "$stats_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['stddev_tps'])")
            echo "[TG $name] ${mean_tps} tok/s (+/-${stddev_tps})"
            TG_SUMMARY+=("$mean_tps")
        else
            echo "[TG $name] NO DATA"
            TG_SUMMARY+=("0")
        fi
        rm -f "$request_file"
        echo ""
    }

    # Run standard TG workloads
    run_tg_workload "short_prompt"  128  256 "$TG_RUNS"
    run_tg_workload "medium_prompt" 1024 256 "$TG_RUNS"
    run_tg_workload "long_prompt"   4096 256 "$TG_RUNS"

    # Multi-turn workload (skip in quick mode)
    if [[ "$QUICK" != true ]]; then
        echo "  Multi-turn workload (5 turns, $TG_RUNS runs)"
        MULTI_RESULTS=()

        for run in $(seq 1 "$TG_RUNS"); do
            echo '[{"role":"user","content":"Tell me about the history of computing. Start from the beginning."}]' > "$MESSAGES_FILE"
            TOTAL_TOKENS=0
            TOTAL_MS=0
            RUN_FAILED=false

            for turn in $(seq 1 5); do
                START_MS=$(date +%s%3N)

                # Build request body from messages file
                python3 -c "
import json
with open('$MESSAGES_FILE') as f:
    msgs = json.load(f)
print(json.dumps({
    'model': 'bench',
    'messages': msgs,
    'max_tokens': 256,
    'temperature': 0.7,
    'stream': False
}))
" | curl -sf http://localhost:8080/v1/chat/completions \
                    -H "Content-Type: application/json" \
                    -d @- > "$RESPONSE_FILE" 2>/dev/null || {
                    echo "  [TG multi_turn] Run $run Turn $turn: FAILED"
                    RUN_FAILED=true
                    break
                }

                END_MS=$(date +%s%3N)
                WALL=$((END_MS - START_MS))
                TOTAL_MS=$((TOTAL_MS + WALL))

                # Extract completion tokens — prefer timings, fall back to usage
                TURN_DATA=$(python3 -c "
import json
with open('$RESPONSE_FILE') as f:
    resp = json.load(f)
comp_tok = resp.get('usage', {}).get('completion_tokens', 0)
print(comp_tok)
" 2>/dev/null || echo "0")
                TOTAL_TOKENS=$((TOTAL_TOKENS + TURN_DATA))

                # Append assistant reply + next user message
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

            if [[ "$RUN_FAILED" == true ]]; then
                continue
            fi

            # Compute TPS for this multi-turn run
            TPS="0"
            if [[ "$TOTAL_TOKENS" -gt 0 && "$TOTAL_MS" -gt 0 ]]; then
                TPS=$(python3 -c "print(round($TOTAL_TOKENS / ($TOTAL_MS / 1000), 1))")
            fi

            if [[ "$run" -eq 1 ]]; then
                echo "  [TG multi_turn] Run $run (warmup): ${TOTAL_TOKENS} tokens, ${TPS} tok/s, ${TOTAL_MS}ms"
            else
                echo "  [TG multi_turn] Run $run: ${TOTAL_TOKENS} tokens, ${TPS} tok/s, ${TOTAL_MS}ms"
            fi
            MULTI_RESULTS+=("{\"run\": $run, \"total_tokens\": $TOTAL_TOKENS, \"tokens_per_sec\": $TPS, \"wall_ms\": $TOTAL_MS}")
        done

        # Compute stats and save multi-turn
        if [[ ${#MULTI_RESULTS[@]} -gt 0 ]]; then
            MULTI_STATS=$(printf '%s\n' "${MULTI_RESULTS[@]}" | compute_stats)
            save_to_result "tg_benchmarks" "multi_turn" "$MULTI_STATS"

            MT_MEAN=$(echo "$MULTI_STATS" | python3 -c "import sys,json; print(json.load(sys.stdin)['mean_tps'])")
            MT_STDDEV=$(echo "$MULTI_STATS" | python3 -c "import sys,json; print(json.load(sys.stdin)['stddev_tps'])")
            echo "[TG multi_turn] ${MT_MEAN} tok/s (+/-${MT_STDDEV})"
            TG_SUMMARY+=("$MT_MEAN")
        else
            echo "[TG multi_turn] NO DATA"
            TG_SUMMARY+=("0")
        fi
        echo ""
    fi
fi

# ============================================================================
# 12. CAPTURE FINAL VRAM AND SAVE GPU STATS
# ============================================================================

VRAM_AFTER_BENCH=$(capture_vram_json)

python3 << PYEOF
import json
with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)
data['gpu_stats']['after_bench'] = json.loads('$VRAM_AFTER_BENCH')
with open('$RESULT_FILE', 'w') as f:
    json.dump(data, f, indent=2)
PYEOF

# ============================================================================
# 13. FINAL SUMMARY
# ============================================================================

echo "=========================================="

# Build summary line
SUMMARY="RESULT:"

if [[ "$TG_ONLY" != true && ${#PP_SUMMARY[@]} -gt 0 ]]; then
    PP_STR=$(IFS=/; echo "${PP_SUMMARY[*]}")
    SUMMARY+=" PP=${PP_STR}"
fi

if [[ "$PP_ONLY" != true && ${#TG_SUMMARY[@]} -gt 0 ]]; then
    TG_STR=$(IFS=/; echo "${TG_SUMMARY[*]}")
    SUMMARY+=" TG=${TG_STR}"
fi

SUMMARY+=" VRAM=${VRAM_USED_MB} MB"

echo "$SUMMARY"
echo "Output: $RESULT_FILE"
echo "=========================================="
