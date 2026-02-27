# Session 003: Optimization Sweep — Success Criteria

## Goal

Test newly discovered llama.cpp flags and Unsloth Dynamic Q8_K_XL quant against our Session 002 winner (A2: llama.cpp + KV q8_0, ~22 tok/s).

## Test Matrix

| Config | ID | Description | Model |
|--------|----|-------------|-------|
| Control | C1 | Reproduce A2 baseline | Vanilla Q8_0 (79 GB) |
| New Flags | C2 | `-cmoe` + `--mlock` + `GGML_CUDA_GRAPH_OPT=1` + `--swa-full` | Vanilla Q8_0 (79 GB) |
| UD Q8 | C3 | C2 flags + Unsloth Dynamic Q8_K_XL model | UD-Q8_K_XL (93 GB) |

### Flag Changes in C2 vs C1

| What | C1 (Control) | C2 (New Flags) |
|------|-------------|----------------|
| MoE offload | `-ot "exps=CPU"` | `-cmoe` |
| Memory lock | (none) | `--mlock` |
| CUDA graphs | (none) | `GGML_CUDA_GRAPH_OPT=1` |
| SWA cache | (default: sized to window) | `--swa-full` |

## Success Criteria

- [x] C1 reproduces A2 baseline within ±2 tok/s (20-24 tok/s) — **FAILED**: measured 15-17 tok/s (30% regression, Docker image rebuilt with newer llama.cpp)
- [x] C2 benchmarked — shows whether new flags improve throughput — **marginal**: +5-8% short/medium, -4-6% long/multi-turn
- [x] C3 benchmarked — shows UD-Q8_K_XL impact on throughput and VRAM — **slower**: -10-12%, +380 MB VRAM
- [x] All results saved to `benchmarks/` as JSON
- [x] Comparison table produced via `compare-results.py`
- [x] Winner documented with rationale — **no new winner**, Session 002 A2 remains best
- [ ] CLAUDE.md updated if new winner found — N/A, no new winner

## Key Questions to Answer

1. Does `-cmoe` behave identically to `-ot "exps=CPU"`?
2. Does `--mlock` + `--no-mmap` together help vs `--no-mmap` alone?
3. Does `GGML_CUDA_GRAPH_OPT=1` provide measurable speedup?
4. Does `--swa-full` help with Qwen3-Next's hybrid attention (36 DeltaNet + 12 GQA)?
5. Does Unsloth Dynamic Q8_K_XL (93 GB, per-layer optimized) outperform vanilla Q8_0 (79 GB)?

## Execution

```bash
# Run in order:
bash scripts/bench.sh llama-cpp s003-control
bash scripts/bench.sh llama-cpp s003-flags
bash scripts/bench.sh llama-cpp s003-udq8      # after UD model download completes

# Compare:
python3 scripts/compare-results.py benchmarks/
```
