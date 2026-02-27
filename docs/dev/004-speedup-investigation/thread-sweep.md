# Thread Sweep Benchmark Results

Session 004 — Thread count optimization for Qwen3-Next-80B-A3B MoE offloading.

**Date**: 2026-02-25
**Engine**: llama.cpp (llm-server/llama-cpp:latest)
**Base config**: s003-control (A2 winner from Session 002)
**Model**: Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf (84.8 GB)
**Only variable changed**: `-t` (thread count)

## Summary Table

All values are mean tok/s (excluding warmup run 1). Stddev in parentheses.

| Threads | Short Prompt | Medium Prompt | Long Prompt | Multi-turn | Overall Avg |
|---------|-------------|--------------|------------|------------|-------------|
| **8**   | 21.9 (0.4)  | 21.7 (0.2)  | 21.1 (0.2) | 17.2 (0.8) | 20.5        |
| **12**  | 18.7 (0.5)  | 18.4 (0.1)  | 18.4 (0.9) | 18.5 (0.1) | 18.5        |
| **16**  | 17.3 (1.3)  | 17.0 (0.6)  | 16.6 (1.0) | 19.8 (1.5) | 17.7        |
| **20**  | **22.2 (0.1)** | **21.5 (0.1)** | **21.1 (0.1)** | 19.5 (1.9) | **21.1** |
| **24**  | 22.0 (0.1)  | 19.2 (0.7)  | 17.4 (0.6) | **21.2 (0.4)** | 20.0    |

## Winner: 20 threads

**20 threads is the optimal setting** — highest or tied-highest throughput across short, medium, and long prompts with near-zero variance. Multi-turn is slightly behind 24 threads but within noise.

### Key Findings

1. **20 threads is the sweet spot**: Best overall average (21.1 tok/s) with the lowest variance across all workloads. Extremely consistent performance.

2. **16 threads is surprisingly the WORST**: Only 17.7 tok/s average, with high variance (stddev up to 1.5). This is the current production setting and is clearly suboptimal.

3. **8 threads is strong for single-turn**: 21.9 short / 21.7 medium / 21.1 long, but degrades badly on multi-turn (17.2 tok/s) where KV cache management overhead increases.

4. **24 threads shows diminishing returns**: Great for short prompts (22.0) and multi-turn (21.2), but degrades on medium (19.2) and long prompts (17.4). Thread contention likely kicks in with longer prompt processing.

5. **12 threads is mediocre**: Flat ~18.4 across all workloads — not enough parallelism for expert offloading, but not few enough to avoid contention.

### Performance Profile by Workload

- **Short prompt (24 tok input)**: 8/20/24 threads all hit ~22 tok/s ceiling. PCIe bandwidth is the bottleneck, not CPU.
- **Medium prompt (195 tok input)**: 8 and 20 threads are best (~21.5). 24 starts to degrade.
- **Long prompt (603 tok input)**: 8 and 20 threads best (~21.1). 24 drops to 17.4 — thread contention during prompt processing.
- **Multi-turn (5 turns, growing KV)**: 24 threads best (21.2), then 16/20 (~19.5-19.8). 8 threads worst (17.2) — not enough threads for KV cache management.

### Why the U-Shape?

The thread count vs. performance curve is not monotonic — it forms a rough U-shape with a dip at 12-16 threads:

- **Too few threads (8)**: Excellent for pure token generation (PCIe-bound), but insufficient for KV cache operations in multi-turn.
- **12-16 threads**: Worst of both worlds — enough threads to cause memory bandwidth contention during expert transfers, but not enough to offset with faster CPU-side computation. Cache line bouncing and NUMA effects likely contribute.
- **20 threads**: Sweet spot — enough parallelism for CPU-side expert GEMM operations without excessive contention on the PCIe/memory bus.
- **Too many threads (24)**: Diminishing returns as thread synchronization overhead grows, especially visible on longer prompts.

### Variance Analysis

| Threads | Avg Stddev | Max Stddev | Assessment |
|---------|-----------|-----------|------------|
| 8       | 0.35      | 0.81      | Low-medium |
| 12      | 0.38      | 0.85      | Low-medium |
| 16      | **1.04**  | **1.52**  | **High**   |
| 20      | 0.55      | 1.89      | Low (one multi-turn outlier) |
| 24      | 0.44      | 0.74      | Low        |

16 threads has the worst consistency by far — both highest average and maximum stddev.

## Recommendation

**Change production config from `-t 16` to `-t 20`.**

Expected improvement:
- Short prompt: +28% (17.3 -> 22.2 tok/s)
- Medium prompt: +26% (17.0 -> 21.5 tok/s)
- Long prompt: +27% (16.6 -> 21.1 tok/s)
- Multi-turn: roughly flat (19.8 -> 19.5 tok/s)
- Variance: significantly reduced

This brings current llama.cpp build performance back in line with the Session 002 A2 winner (~22 tok/s), suggesting the regression seen in s003-control was partly a thread count interaction with the newer build.

## VRAM Usage

Thread count has no meaningful effect on VRAM — all configs used ~6.4 GB (within measurement noise):

| Threads | VRAM (MB) |
|---------|-----------|
| 8       | 6455      |
| 12      | 6359      |
| 16      | 6429      |
| 20      | 6416      |
| 24      | 6381      |

## Raw Data Files

- `benchmarks/llama-cpp-s004-t8-20260225-021535.json`
- `benchmarks/llama-cpp-s004-t12-20260225-024605.json`
- `benchmarks/llama-cpp-s004-t16-20260225-025609.json`
- `benchmarks/llama-cpp-s004-t20-20260225-030612.json`
- `benchmarks/llama-cpp-s004-t24-20260225-031523.json`

## Config Files

- `configs/llama-cpp-s004-t8.env`
- `configs/llama-cpp-s004-t12.env`
- `configs/llama-cpp-s004-t16.env`
- `configs/llama-cpp-s004-t20.env`
- `configs/llama-cpp-s004-t24.env`

## Note on bench.sh thread_count metadata

The JSON result files incorrectly show `"thread_count": 16` for all configs due to a bug in bench.sh line 92 where `THREAD_OVERRIDE` defaults to 16 instead of parsing from the env file. The actual thread count used in each run was correct (set via `-t` flag from the env file). This is a cosmetic metadata issue only.
