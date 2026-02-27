# Session 002: Infrastructure Setup + A/B Benchmark — Handoff

## What Was Done

1. **Model download**: Qwen3-Next-80B-A3B-Instruct Q8_0 GGUF (79 GB) downloaded via HuggingFace Hub to `models/`
2. **Docker images built**: Two multi-stage Docker images with CUDA sm_120 (RTX 5080):
   - `llm-server/llama-cpp:latest` — mainline llama.cpp
   - `llm-server/ik-llama-cpp:latest` — ik_llama.cpp fork
3. **Benchmark framework**: `scripts/bench.sh` + `scripts/compare-results.py` + 4 config files in `configs/`
4. **A/B benchmarks completed**: 4 configs x 4 workloads x 5 runs each = 80 inference runs total

## Benchmark Results

| Config | Engine | Mean tok/s (all workloads) | VRAM (MB) | Stability |
|--------|--------|--------------------------|-----------|-----------|
| **A2 (WINNER)** | llama.cpp + KV q8_0 | **21.9** | **6247** | Excellent (stddev < 1) |
| A1 | llama.cpp baseline | 18.5 | 6823 | Excellent |
| B1 | ik_llama baseline | 17.4 | 7650 | Poor (high variance) |
| B2 | ik_llama + merge-qkv + ger | 17.9 | 4778 | Fair |

**Winner: llama.cpp + KV cache q8_0** — 22 tok/s, lowest VRAM, most stable.

## Key Discoveries

### KV Cache q8_0 = Free Performance
- +20% throughput over FP16 KV cache (22 vs 18 tok/s)
- Lower VRAM (6247 vs 6823 MB) — quantized cache is smaller
- No perceptible quality loss at q8_0 precision

### ik_llama.cpp Is Problematic
- **SIGSEGV** with `-b 4096 -ub 4096` at any ctx >= 8192 (works with defaults)
- Several documented flags don't exist (`-rtr`, `-fmoe`); correct flags: `--merge-qkv`, `-ger`
- High variance on short prompts (10-17 tok/s), suggesting initialization overhead

### PCIe Bandwidth Is the Bottleneck
- All configs converge near the same ceiling (~18-22 tok/s)
- Token generation speed is independent of prompt length (as expected for MoE offloading)
- Expert transfers (~1.5 GB/token) dominate latency

## Files Created/Modified

- `docker/Dockerfile.llama-cpp` — Multi-stage build with libcuda.so.1 symlink fix
- `docker/Dockerfile.ik-llama-cpp` — Same structure for ik fork
- `docker-compose.yml` — Profiles for both engines
- `configs/llama-cpp-baseline.env` (A1)
- `configs/llama-cpp-optimized.env` (A2) — winning config
- `configs/ik-llama-baseline.env` (B1)
- `configs/ik-llama-optimized.env` (B2)
- `scripts/bench.sh` — Unified benchmark runner
- `scripts/compare-results.py` — Results comparison
- `benchmarks/*.json` — Raw benchmark data (4 files)
- `CLAUDE.md` — Updated with full results and winning config

## Build Quirks Discovered

1. **libcuda.so.1 symlink**: Docker builds without GPU need `ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && ldconfig` — the shared lib records `libcuda.so.1` as DT_NEEDED, not `libcuda.so`
2. **LD_LIBRARY_PATH**: Runtime stage needs `ENV LD_LIBRARY_PATH="/opt/llama-cpp/lib:${LD_LIBRARY_PATH}"` for shared libs
3. **`-fa on` not `-fa`**: Newer llama.cpp requires explicit `on|off|auto` value for flash attention

## Recommended Next Steps

1. **Thread sweep on A2**: Run `bench.sh llama-cpp optimized 8` and `bench.sh llama-cpp optimized 24` to find optimal thread count
2. **KV cache q4_0**: Test even more aggressive quantization (`-ctk q4_0 -ctv q4_0`) — may push past 22 tok/s
3. **Production Docker Compose**: Update `docker-compose.yml` to use A2 config by default
4. **API integration**: Build a thin wrapper for application-specific endpoints
