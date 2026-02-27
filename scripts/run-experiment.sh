#!/usr/bin/env bash
# Session 006: Experiment runner
# Usage: ./run-experiment.sh <experiment> [--dry-run]
#   experiment: e1 | e2 | e4
#
# E1: KV Cache Quality — runs PPL for 6 configs (Q8_0/Q4_K_M × KV f16/q8_0/q4_0)
# E2: KL Divergence — runs KLD base logits then compares Q4_K_M and UD-Q4_K_XL
# E4: fit-target tuning — runs speed benchmarks for 3 fit configs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/models"
BENCHMARK_DIR="$PROJECT_DIR/benchmarks"
PPL_DIR="$BENCHMARK_DIR/perplexity"
KLD_DIR="$BENCHMARK_DIR/kl-divergence"
WIKITEXT="$PPL_DIR/wikitext-2-raw/wiki.test.raw"
IMAGE="llm-server/llama-cpp:latest"
IMAGE_FIT="llm-server/llama-cpp:latest-fit"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
DRY_RUN=false

EXPERIMENT="${1:?Usage: run-experiment.sh <e1|e2|e4> [--dry-run]}"
[[ "${2:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "$PPL_DIR" "$KLD_DIR"

# Helper: parse env file into llama-perplexity args
build_ppl_args() {
    local env_file="$1"
    local extra_args="${2:-}"
    local args=""
    local model_path=""

    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" =~ ^# ]] && continue
        case "$key" in
            LLAMA_ARG_MODEL)          args+=" -m $value"; model_path="$value" ;;
            LLAMA_ARG_CTX_SIZE)       args+=" -c $value" ;;
            LLAMA_ARG_N_GPU_LAYERS)   args+=" -ngl $value" ;;
            LLAMA_ARG_OVERRIDE_TENSOR) args+=" -ot $value" ;;
            LLAMA_ARG_FLASH_ATTN)     [[ "$value" == "true" ]] && args+=" -fa on" ;;
            LLAMA_ARG_THREADS)        args+=" -t $value" ;;
            LLAMA_ARG_BATCH_SIZE)     args+=" -b $value" ;;
            LLAMA_ARG_UBATCH_SIZE)    args+=" -ub $value" ;;
            LLAMA_ARG_NO_MMAP)        [[ "$value" == "true" ]] && args+=" --no-mmap" ;;
            LLAMA_ARG_CACHE_TYPE_K)   args+=" -ctk $value" ;;
            LLAMA_ARG_CACHE_TYPE_V)   args+=" -ctv $value" ;;
            LLAMA_ARG_N_CPU_MOE)      args+=" --n-cpu-moe $value" ;;
        esac
    done < "$env_file"

    echo "$args $extra_args"
}

# Helper: run perplexity in Docker
run_ppl() {
    local name="$1"
    local env_file="$2"
    local extra_args="${3:-}"
    local log_file="$PPL_DIR/ppl-${name}-${TIMESTAMP}.log"

    local args
    args=$(build_ppl_args "$env_file" "$extra_args")

    echo ""
    echo "=========================================="
    echo " PPL: $name"
    echo " Config: $(basename "$env_file")"
    echo " Log: $log_file"
    echo " Args: llama-perplexity $args -f /data/wiki.test.raw"
    echo "=========================================="

    if $DRY_RUN; then
        echo "[DRY RUN] Would run: docker run --rm --gpus all --ipc host ..."
        return 0
    fi

    docker run --rm \
        --gpus all \
        --ipc host \
        -v "$MODEL_DIR:/models:ro" \
        -v "$PPL_DIR/wikitext-2-raw:/data:ro" \
        --entrypoint llama-perplexity \
        "$IMAGE" \
        $args \
        -f /data/wiki.test.raw \
        2>&1 | tee "$log_file"

    # Extract final PPL
    local ppl
    ppl=$(grep "Final estimate:" "$log_file" | grep -oP 'PPL = \K[\d.]+' || echo "N/A")
    echo ""
    echo ">>> $name PPL = $ppl"
    echo ""
}

# Helper: run KLD base logits generation
run_kld_base() {
    local name="$1"
    local env_file="$2"
    local chunks="$3"
    local kld_file="$KLD_DIR/base-${name}.kld"

    local args
    args=$(build_ppl_args "$env_file")

    echo ""
    echo "=========================================="
    echo " KLD Base Logits: $name"
    echo " Chunks: $chunks"
    echo " Output: $kld_file"
    echo "=========================================="

    if $DRY_RUN; then
        echo "[DRY RUN] Would generate base logits"
        return 0
    fi

    docker run --rm \
        --gpus all \
        --ipc host \
        -v "$MODEL_DIR:/models:ro" \
        -v "$PPL_DIR/wikitext-2-raw:/data:ro" \
        -v "$KLD_DIR:/kld" \
        --entrypoint llama-perplexity \
        "$IMAGE" \
        $args \
        -f /data/wiki.test.raw \
        --chunks "$chunks" \
        --kl-divergence-base "/kld/base-${name}.kld" \
        2>&1 | tee "$KLD_DIR/kld-base-${name}-${TIMESTAMP}.log"
}

# Helper: run KLD comparison
run_kld_compare() {
    local name="$1"
    local env_file="$2"
    local base_name="$3"
    local kld_base="$KLD_DIR/base-${base_name}.kld"

    local args
    args=$(build_ppl_args "$env_file")

    echo ""
    echo "=========================================="
    echo " KLD Compare: $name vs $base_name"
    echo "=========================================="

    if $DRY_RUN; then
        echo "[DRY RUN] Would compare KLD against $kld_base"
        return 0
    fi

    if [[ ! -f "$kld_base" ]]; then
        echo "ERROR: Base KLD file not found: $kld_base"
        echo "Run KLD base generation first."
        return 1
    fi

    docker run --rm \
        --gpus all \
        --ipc host \
        -v "$MODEL_DIR:/models:ro" \
        -v "$KLD_DIR:/kld" \
        --entrypoint llama-perplexity \
        "$IMAGE" \
        $args \
        --kl-divergence-base "/kld/base-${base_name}.kld" \
        --kl-divergence \
        2>&1 | tee "$KLD_DIR/kld-${name}-vs-${base_name}-${TIMESTAMP}.log"
}

case "$EXPERIMENT" in
    e1)
        echo "============================================================"
        echo " Experiment 1: KV Cache Quality Impact"
        echo " 6 PPL runs: Q8_0/Q4_K_M × KV f16/q8_0/q4_0"
        echo " Estimated time: ~25 minutes"
        echo "============================================================"

        CONFIGS_DIR="$PROJECT_DIR/configs"

        # Q8_0 model × 3 KV types
        run_ppl "s006-e1-q8-kvf16" "$CONFIGS_DIR/llama-cpp-s006-e1-q8-kvf16.env"
        run_ppl "s006-e1-q8-kvq8"  "$CONFIGS_DIR/llama-cpp-s006-e1-q8-kvq8.env"
        run_ppl "s006-e1-q8-kvq4"  "$CONFIGS_DIR/llama-cpp-s006-e1-q8-kvq4.env"

        # Q4_K_M model × 3 KV types
        run_ppl "s006-e1-q4km-kvf16" "$CONFIGS_DIR/llama-cpp-s006-e1-q4km-kvf16.env"
        run_ppl "s006-e1-q4km-kvq8"  "$CONFIGS_DIR/llama-cpp-s006-e1-q4km-kvq8.env"
        run_ppl "s006-e1-q4km-kvq4"  "$CONFIGS_DIR/llama-cpp-s006-e1-q4km-kvq4.env"

        echo ""
        echo "============================================================"
        echo " E1 Complete — extracting results"
        echo "============================================================"

        echo ""
        echo "| Model Quant | KV f16 | KV q8_0 | KV q4_0 |"
        echo "|-------------|--------|---------|---------|"
        for model in q8 q4km; do
            row="| $([ "$model" = "q8" ] && echo "Q8_0" || echo "Q4_K_M") |"
            for kv in kvf16 kvq8 kvq4; do
                log="$PPL_DIR/ppl-s006-e1-${model}-${kv}-${TIMESTAMP}.log"
                if [[ -f "$log" ]]; then
                    ppl=$(grep "Final estimate:" "$log" | grep -oP 'PPL = \K[\d.]+' || echo "N/A")
                    row+=" $ppl |"
                else
                    row+=" pending |"
                fi
            done
            echo "$row"
        done
        ;;

    e2)
        echo "============================================================"
        echo " Experiment 2: KL Divergence"
        echo " Step 1: Generate base logits from Q8_0 (25 chunks)"
        echo " Step 2: Compare Q4_K_M and UD-Q4_K_XL"
        echo " Estimated time: ~20 minutes"
        echo "============================================================"

        CONFIGS_DIR="$PROJECT_DIR/configs"

        # Use small-context KLD configs (512 ctx to keep logit file manageable)
        # 248K vocab × 40K tokens × 2 bytes ≈ ~19 GiB logit file at 80 chunks

        # Step 1: Generate base logits from Q8_0
        run_kld_base "Q8_0" "$CONFIGS_DIR/llama-cpp-s006-e2-q8-kld-base.env" 80

        # Step 2: Compare Q4_K_M
        run_kld_compare "Q4_K_M" "$CONFIGS_DIR/llama-cpp-s006-e2-q4km-kld.env" "Q8_0"

        # Step 3: Compare UD-Q4_K_XL
        run_kld_compare "UD-Q4_K_XL" "$CONFIGS_DIR/llama-cpp-s006-e2-ud-q4kxl-kld.env" "Q8_0"

        echo ""
        echo "============================================================"
        echo " E2 Complete — check logs in $KLD_DIR/"
        echo "============================================================"
        ;;

    e4)
        echo "============================================================"
        echo " Experiment 4: --fit-target Tuning"
        echo " 3 speed benchmarks: fit default vs fit-target 256 vs fit no-batch"
        echo " Estimated time: ~45 minutes"
        echo " NOTE: Uses latest-fit image (b8149, supports --fit)"
        echo "============================================================"

        CONFIGS_DIR="$PROJECT_DIR/configs"

        # These need the latest-fit image and bench.sh
        if $DRY_RUN; then
            echo "[DRY RUN] Would run: bench.sh llama-cpp s006-e4-fit-default (image: $IMAGE_FIT)"
            echo "[DRY RUN] Would run: bench.sh llama-cpp s006-e4-fit-256 (image: $IMAGE_FIT)"
            echo "[DRY RUN] Would run: bench.sh llama-cpp s006-e4-fit-nobatch (image: $IMAGE_FIT)"
        else
            bash "$SCRIPT_DIR/bench.sh" llama-cpp "s006-e4-fit-default" "" "$IMAGE_FIT"
            bash "$SCRIPT_DIR/bench.sh" llama-cpp "s006-e4-fit-256" "" "$IMAGE_FIT"
            bash "$SCRIPT_DIR/bench.sh" llama-cpp "s006-e4-fit-nobatch" "" "$IMAGE_FIT"
        fi

        echo ""
        echo "============================================================"
        echo " E4 Complete — compare with:"
        echo "   python3 scripts/compare-results.py benchmarks/ --filter s006-e4"
        echo "============================================================"
        ;;

    e5)
        echo "============================================================"
        echo " Experiment 5: Self-Speculative Decoding (ngram methods)"
        echo " No draft model needed — no compatible small Qwen3.5 exists"
        echo " 3 speed benchmarks: ngram-simple, ngram-mod, ngram-simple-short"
        echo " Estimated time: ~45 minutes"
        echo " NOTE: Uses latest image (HEAD, has --spec-type + --fit)"
        echo "============================================================"

        CONFIGS_DIR="$PROJECT_DIR/configs"

        if $DRY_RUN; then
            echo "[DRY RUN] Would run: bench.sh llama-cpp s006-e5-ngram-simple (image: $IMAGE)"
            echo "[DRY RUN] Would run: bench.sh llama-cpp s006-e5-ngram-mod (image: $IMAGE)"
            echo "[DRY RUN] Would run: bench.sh llama-cpp s006-e5-ngram-simple-short (image: $IMAGE)"
        else
            bash "$SCRIPT_DIR/bench.sh" llama-cpp "s006-e5-ngram-simple" "" "$IMAGE"
            bash "$SCRIPT_DIR/bench.sh" llama-cpp "s006-e5-ngram-mod" "" "$IMAGE"
            bash "$SCRIPT_DIR/bench.sh" llama-cpp "s006-e5-ngram-simple-short" "" "$IMAGE"
        fi

        echo ""
        echo "============================================================"
        echo " E5 Complete — compare with:"
        echo "   python3 scripts/compare-results.py benchmarks/ --filter s006-e5"
        echo " Baseline (fit-nobatch): ~74 tok/s from E4"
        echo "============================================================"
        ;;

    e7)
        echo "============================================================"
        echo " Experiment 7: MXFP4_MOE Quality + Speed"
        echo " PPL, KLD, and speed benchmark for MXFP4_MOE quant"
        echo " Recommended by danielhanchen (Unsloth creator)"
        echo " Estimated time: ~30 minutes"
        echo "============================================================"

        CONFIGS_DIR="$PROJECT_DIR/configs"
        MXFP4_MODEL="$MODEL_DIR/Qwen3.5-35B-A3B-MXFP4_MOE.gguf"

        if [[ ! -f "$MXFP4_MODEL" ]]; then
            echo "ERROR: MXFP4_MOE model not found at $MXFP4_MODEL"
            echo "Download first: huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-MXFP4_MOE.gguf --local-dir models/"
            exit 1
        fi

        if $DRY_RUN; then
            echo "[DRY RUN] Would run PPL for MXFP4_MOE"
            echo "[DRY RUN] Would run KLD compare for MXFP4_MOE vs Q8_0"
            echo "[DRY RUN] Would run speed benchmark for MXFP4_MOE"
        else
            # Step 1: PPL
            run_ppl "s006-e7-mxfp4-ppl" "$CONFIGS_DIR/llama-cpp-s006-e7-mxfp4-ppl.env"

            # Step 2: KLD compare (uses existing Q8_0 base logits from E2)
            run_kld_compare "MXFP4_MOE" "$CONFIGS_DIR/llama-cpp-s006-e7-mxfp4-kld.env" "Q8_0"

            # Step 3: Speed benchmark
            bash "$SCRIPT_DIR/bench.sh" llama-cpp "s006-e7-mxfp4-speed"
        fi

        echo ""
        echo "============================================================"
        echo " E7 Complete — compare with:"
        echo "   Q4_K_M PPL: 6.0 | KLD: 0.028 | Speed: ~74 tok/s"
        echo "   UD-Q4_K_XL PPL: 7.17 | KLD: 0.109"
        echo "============================================================"
        ;;

    e3)
        echo "============================================================"
        echo " Experiment 3: Bartowski Q4_K_L Quality + Speed"
        echo " PPL, KLD, and speed benchmark for bartowski Q4_K_L quant"
        echo " Compare vs our Q4_K_M"
        echo " Estimated time: ~30 minutes"
        echo "============================================================"

        CONFIGS_DIR="$PROJECT_DIR/configs"
        Q4KL_MODEL="$MODEL_DIR/Qwen_Qwen3.5-35B-A3B-Q4_K_L.gguf"

        if [[ ! -f "$Q4KL_MODEL" ]]; then
            echo "ERROR: Q4_K_L model not found at $Q4KL_MODEL"
            echo "Download first from bartowski/Qwen_Qwen3.5-35B-A3B-GGUF"
            exit 1
        fi

        if $DRY_RUN; then
            echo "[DRY RUN] Would run PPL for Q4_K_L"
            echo "[DRY RUN] Would run KLD compare for Q4_K_L vs Q8_0"
            echo "[DRY RUN] Would run speed benchmark for Q4_K_L"
        else
            # Step 1: PPL
            run_ppl "s006-e3-q4kl-ppl" "$CONFIGS_DIR/llama-cpp-s006-e3-q4kl-ppl.env"

            # Step 2: KLD compare (uses existing Q8_0 base logits from E2)
            run_kld_compare "Q4_K_L" "$CONFIGS_DIR/llama-cpp-s006-e3-q4kl-kld.env" "Q8_0"

            # Step 3: Speed benchmark
            bash "$SCRIPT_DIR/bench.sh" llama-cpp "s006-e3-q4kl-speed" "" "$IMAGE_FIT"
        fi

        echo ""
        echo "============================================================"
        echo " E3 Complete — compare with:"
        echo "   Q4_K_M PPL: 6.0 | KLD: 0.028 | Speed: ~74 tok/s"
        echo "============================================================"
        ;;

    e6)
        echo "============================================================"
        echo " Experiment 6: Qwen3.5-27B Dense Model Comparison"
        echo " PPL and speed benchmark for dense 27B vs MoE 35B-A3B"
        echo " Estimated time: ~20 minutes"
        echo "============================================================"

        CONFIGS_DIR="$PROJECT_DIR/configs"
        MODEL_27B="$MODEL_DIR/Qwen3.5-27B-Q4_K_M.gguf"

        if [[ ! -f "$MODEL_27B" ]]; then
            echo "ERROR: Qwen3.5-27B model not found at $MODEL_27B"
            echo "Download first from unsloth/Qwen3.5-27B-GGUF"
            exit 1
        fi

        if $DRY_RUN; then
            echo "[DRY RUN] Would run PPL for Qwen3.5-27B Q4_K_M"
            echo "[DRY RUN] Would run speed benchmark for Qwen3.5-27B Q4_K_M"
        else
            # Step 1: PPL
            run_ppl "s006-e6-27b-ppl" "$CONFIGS_DIR/llama-cpp-s006-e6-27b-ppl.env"

            # Step 2: Speed benchmark
            bash "$SCRIPT_DIR/bench.sh" llama-cpp "s006-e6-27b-speed" "" "$IMAGE_FIT"
        fi

        echo ""
        echo "============================================================"
        echo " E6 Complete — compare with:"
        echo "   35B-A3B Q4_K_M PPL: 6.67 | Speed: ~74 tok/s"
        echo "============================================================"
        ;;

    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Available: e1, e2, e3, e4, e5, e6, e7"
        exit 1
        ;;
esac
