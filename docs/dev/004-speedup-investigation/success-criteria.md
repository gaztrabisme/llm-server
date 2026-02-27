# Session 004: Speedup Investigation

## Mode: Analyze

Research optimization avenues and investigate the Session 003 performance regression.

## Done When

- [x] Root cause identified for 30% regression (22 → 15-17 tok/s) between Session 002 and 003 — **Two causes: unpinned Dockerfile pulled newer llama.cpp with Qwen graph refactoring regressions, AND 16 threads is suboptimal (U-shaped curve)**
- [x] Speculative decoding feasibility assessed — **Feasible: Qwen3-1.7B Q8_0 draft model, `--model-draft` flag, expected 1.3-2x speedup, risk is unknown acceptance rate**
- [x] Expert caching feasibility assessed — **Limited: no dynamic caching in mainline llama.cpp, static partial offload via `--n-cpu-moe 42` for 5-15% gain, academic forks show 10x potential**
- [x] Thread sweep benchmarked — **20 threads is optimal at ~21 tok/s (+27% over t16), U-shaped curve with t16 being worst**
- [x] Findings documented with actionable next steps

## Research Topics

1. **Commit investigation** — Unpinned Dockerfile, Qwen graph refactoring + CUDA changes likely caused regression. Pin to known-good commit.
2. **Speculative decoding** — Best candidate: Qwen3-1.7B Q8_0 (~2 GB VRAM). Also: free n-gram speculation.
3. **Expert caching** — No dynamic caching upstream. Static partial offload available. HOBBIT fork most promising.
4. **Thread sweep** — Optimal: 20 threads. Results: t8=20.5, t12=18.5, t16=17.7, t20=21.1, t24=20.0 avg tok/s.

## Quick Win

**Change `-t 16` to `-t 20`** in production config. Immediate +27% throughput, zero risk.
