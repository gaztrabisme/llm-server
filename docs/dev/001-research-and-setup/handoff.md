# Session 001: Research & Setup — Handoff

## Delivered

| File | Purpose |
|------|---------|
| `research/qwen3-next-80b-research.md` | Model architecture, GGUF sizes, memory analysis, MoE offloading strategy, benchmarks |
| `research/rtx-5080-specs-and-cuda.md` | GPU specs, CUDA 13.1 details, Docker images, bandwidth analysis |
| `research/llama-cpp-research-feb2026.md` | llama.cpp features, Blackwell support, build options, server mode |
| `research/moe-offloading-research.md` | MoE offloading strategies, expert caching, alternative tools (ktransformers, HOBBIT) |
| `research/local-llm-serving-guide.md` | Serving patterns, API compatibility, llama-swap, Docker Compose, application integration |
| `CLAUDE.md` | Project-level context file for future sessions |

## Key Findings

1. **Qwen3-Next-80B-A3B at Q8_0 (84.8 GB)** fits in 128GB RAM with ~43 GB headroom
2. **Active experts (~3.9 GB at Q8)** fit in 16 GB VRAM with ~10 GB to spare
3. **llama.cpp `-ot "exps=CPU"`** is purpose-built for this exact strategy
4. **PCIe 5.0 x16** provides ~64 GB/s, yielding theoretical ceiling of ~42 t/s (realistic: 15-30 t/s)
5. **Hybrid DeltaNet attention** means only 12/48 layers maintain KV cache — tiny memory footprint even at 128K context
6. **CUDA 13.1** with driver 590.48.01 is fully supported; compile with `sm_120`
7. **cuBLAS Grouped GEMM** in CUDA 13.1 offers up to 4x speedup for MoE models
8. **Docker base image**: `nvidia/cuda:13.1.1-devel-ubuntu24.04` for building

## Key Decisions

1. Target model: Qwen3-Next-80B-A3B-Instruct at Q8_0 quantization
2. Strategy: Full model in RAM, experts offloaded via `-ot "exps=CPU"`, attention on GPU
3. PCIe bandwidth is the primary bottleneck — optimization experiments will focus here
4. Docker-based builds targeting sm_120 (compute capability 12.0)

## Architecture Impact

This is a greenfield project. No existing architecture to modify.

## What Next Session Needs To Know

### Recommended first launch command:
```bash
./llama-server \
  -m ./Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf \
  -c 32768 \
  -ngl 999 \
  -ot "exps=CPU" \
  -fa on \
  -t 16 \
  -b 4096 \
  -ub 4096 \
  --no-mmap \
  --jinja
```

### Build next steps:
1. Set up Docker environment with CUDA 13.1 base image
2. Build llama.cpp from source with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`
3. Download Qwen3-Next-80B-A3B-Instruct Q8_0 GGUF
4. Run baseline benchmarks with the recommended command
5. Experiment with tuning: thread count (`-t`), batch sizes (`-b`, `-ub`), context size (`-c`)
6. Explore alternative tools: ktransformers (reported ~2x faster for MoE), ik_llama.cpp fork

### Optimization areas to explore:
- Expert caching strategies (if llama.cpp adds support)
- Thread count tuning (start with 16, test range 8-24)
- Batch size tuning for different workloads (single vs concurrent requests)
- KV cache quantization (Q8_0 or Q4_0) for longer contexts
- Speculative decoding with a smaller draft model
- Compare Instruct vs Thinking variant for different use cases
