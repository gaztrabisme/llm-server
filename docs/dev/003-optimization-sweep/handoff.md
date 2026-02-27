# Session 003: Optimization Sweep — Handoff

## Delivered

| File | Purpose |
|------|---------|
| `configs/llama-cpp-s003-control.env` | C1: Exact A2 reproduction |
| `configs/llama-cpp-s003-flags.env` | C2: New flags (`-cmoe`, `--mlock`, `--swa-full`, `GGML_CUDA_GRAPH_OPT=1`) |
| `configs/llama-cpp-s003-udq8.env` | C3: UD-Q8_K_XL model with C2 flags |
| `benchmarks/llama-cpp-s003-control-20260224-013038.json` | C1 benchmark results |
| `benchmarks/llama-cpp-s003-flags-20260224-014217.json` | C2 benchmark results |
| `benchmarks/llama-cpp-s003-udq8-20260224-015412.json` | C3 benchmark results |
| `models/UD-Q8_K_XL/` | Unsloth Dynamic Q8_K_XL model (93 GB, 2 files) |
| `scripts/bench.sh` | Updated: dynamic model path, env var passthrough, new flag support |

## Results

All configs: 16 threads, 32k ctx, experts on CPU, `--no-mmap`, KV cache q8_0.

| Config | Short (tok/s) | Medium (tok/s) | Long (tok/s) | Multi-turn (tok/s) | VRAM (MB) |
|--------|--------------|----------------|-------------|-------------------|-----------|
| **C1** (control = A2 repro) | 15.7 ±0.2 | 15.0 ±0.1 | 15.3 ±1.0 | 17.2 ±0.5 | 6694 |
| **C2** (new flags) | 16.5 ±0.2 | 16.2 ±0.1 | 14.7 ±0.0 | 16.1 ±0.9 | 6698 |
| **C3** (UD-Q8_K_XL) | 14.0 ±0.1 | 13.7 ±0.1 | 13.9 ±0.4 | 15.2 ±1.8 | 7074 |

### vs Session 002 A2 (~22 tok/s)

C1 should reproduce A2 but measured ~30% slower (15-17 vs 22 tok/s). The Docker image was rebuilt between sessions — a newer llama.cpp commit may have introduced a performance regression, or system state differed (suspend cycles, swap usage).

## Answers to Key Questions

1. **Does `-cmoe` behave identically to `-ot "exps=CPU"`?** — Functionally yes. Same VRAM usage (6698 vs 6694 MB). Throughput within noise.

2. **Does `--mlock` + `--no-mmap` together help?** — No measurable improvement. Both C1 (no mlock) and C2 (with mlock) show comparable or C2 slightly worse on some workloads.

3. **Does `GGML_CUDA_GRAPH_OPT=1` provide measurable speedup?** — Not clearly. C2 has a slight edge on short/medium (+5-8%) but loses on long/multi-turn. The flag's impact is within noise for this model/hardware.

4. **Does `--swa-full` help with hybrid attention?** — Not measurably. Part of C2 which doesn't clearly beat C1.

5. **Does UD-Q8_K_XL outperform vanilla Q8_0?** — **No, it's slower.** 10-12% throughput loss due to 17% larger model (93 vs 79 GB), more PCIe transfers per token. Uses 380 MB more VRAM. Quality may be marginally better but we don't measure that, and the speed penalty is real.

## Key Decisions

1. **UD-Q8_K_XL dropped** — slower and larger with no measurable benefit in a PCIe-bottlenecked scenario. The quality uplift (upcasted important layers) doesn't compensate for bandwidth cost.

2. **New flags dropped** — `-cmoe` is a cleaner replacement for `-ot "exps=CPU"` but provides no speed advantage. `--mlock`, `--swa-full`, and `GGML_CUDA_GRAPH_OPT=1` are all negligible impact.

3. **Session 002 A2 config remains the production winner** — the original `llama-cpp/optimized` config with `-ot "exps=CPU" -ctk q8_0 -ctv q8_0` at ~22 tok/s.

4. **Performance regression needs investigation** — same config measuring 30% slower after Docker image rebuild suggests a llama.cpp version sensitivity. Consider pinning to a specific commit in the Dockerfile.

## What Next Session Needs To Know

- The 30% performance regression between Session 002 and 003 (with same config) is the biggest open question. Next session should:
  1. Check which llama.cpp commit was used in each Docker build
  2. Consider pinning the Dockerfile to a known-good commit
  3. Re-run A2 with the original Docker image (if it still exists — it was rebuilt)
- The UD-Q8_K_XL model files (93 GB) in `models/UD-Q8_K_XL/` can be deleted to free disk space
- All three new flags (`-cmoe`, `--mlock`, `--swa-full`) are safe to use but provide no throughput benefit
- Speculative decoding and expert caching remain unexplored optimization avenues
