#!/usr/bin/env bash
# Phase 1: Flag sweep for IQ3+MTP dream config
# Tests each flag in isolation vs baseline, records mtp-bench results.
set -euo pipefail

MODEL="/models/Qwen3.6-27B-MTP-UD-IQ3_XXS.gguf"
IMAGE="llm-server/llama-cpp:s015-mtp"
PORT=8082
CONTAINER="flag-sweep"
RESULTS_DIR="benchmarks/flag-sweep-mtp"
BENCH_SCRIPT="scripts/mtp-bench.py"

mkdir -p "$RESULTS_DIR"

BASE_FLAGS=(
  -m "$MODEL"
  -c 8192
  --fit on
  --host 0.0.0.0
  -fa on
  -t 20
  --no-mmap
  --jinja
  -ctk q8_0
  -ctv q8_0
  --spec-type mtp
)

declare -A CONFIGS
CONFIGS[baseline]=""
CONFIGS[mlock]="--mlock"
CONFIGS[fit-target-0]="-fitt 0"
CONFIGS[ctx-checkpoints-2]="-ctxcp 2"
CONFIGS[ctx-checkpoints-4]="-ctxcp 4"
CONFIGS[no-repack]="--no-repack"
CONFIGS[mlock-fitt0]="--mlock -fitt 0"

run_test() {
  local name="$1"
  local extra_flags="$2"
  local outfile="$RESULTS_DIR/s016-$name.json"

  echo ""
  echo "========================================="
  echo "  TEST: $name"
  echo "  Flags: ${extra_flags:-'(none)'}"
  echo "========================================="

  # Stop any existing container
  docker stop "$CONTAINER" 2>/dev/null || true
  docker rm "$CONTAINER" 2>/dev/null || true
  sleep 2

  # Start container
  echo "Starting server..."
  # shellcheck disable=SC2086
  docker run --gpus all --rm -d --name "$CONTAINER" \
    -v /home/bppc/projects/llm-server/models:/models \
    -p "$PORT:8080" \
    "$IMAGE" \
    "${BASE_FLAGS[@]}" $extra_flags

  # Wait for health
  echo "Waiting for server..."
  local attempts=0
  until curl -s "http://localhost:$PORT/health" | grep -q ok; do
    sleep 2
    attempts=$((attempts + 1))
    if [ "$attempts" -gt 30 ]; then
      echo "TIMEOUT waiting for server"
      docker logs "$CONTAINER" 2>&1 | tail -10
      return 1
    fi
  done
  echo "Server ready"

  # Grab VRAM and layer info
  echo "Server logs:"
  docker logs "$CONTAINER" 2>&1 | grep -iE "offloaded|projected|free|fit_target|mlock" || true

  # Run mtp-bench
  echo "Running mtp-bench..."
  .venv/bin/python3 "$BENCH_SCRIPT" --url "http://localhost:$PORT" --out "$outfile" 2>&1

  echo "Results saved to $outfile"

  # Stop
  docker stop "$CONTAINER" 2>/dev/null || true
}

# Run all configs
ORDER=(baseline mlock fit-target-0 ctx-checkpoints-2 ctx-checkpoints-4 no-repack mlock-fitt0)

for name in "${ORDER[@]}"; do
  run_test "$name" "${CONFIGS[$name]}"
done

echo ""
echo "========================================="
echo "  ALL TESTS COMPLETE"
echo "========================================="
echo ""

# Compare all results vs baseline
BASELINE="$RESULTS_DIR/s016-baseline.json"
if [ -f "$BASELINE" ]; then
  for name in "${ORDER[@]}"; do
    if [ "$name" = "baseline" ]; then continue; fi
    result="$RESULTS_DIR/s016-$name.json"
    if [ -f "$result" ]; then
      echo ""
      echo "--- $name vs baseline ---"
      .venv/bin/python3 "$BENCH_SCRIPT" --diff "$BASELINE" "$result"
    fi
  done
fi
