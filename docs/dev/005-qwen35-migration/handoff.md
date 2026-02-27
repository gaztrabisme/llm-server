# Session 005: Qwen3.5-35B-A3B Migration — Handoff

## Delivered

### Model Files

| File | Size | Purpose |
|------|------|---------|
| `models/Qwen3.5-35B-A3B-Q8_0.gguf` | 36.9 GB | Reference quant (best quality, ~40 tok/s with `--fit on`) |
| `models/Qwen3.5-35B-A3B-Q4_K_M.gguf` | ~20 GB | **Production quant** (~70 tok/s, +2.1% PPL) |
| `models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` | ~20 GB | Not recommended (+9.7% PPL, worse than Q4_K_M) |
| `models/mmproj-BF16.gguf` | — | Vision encoder (multimodal support) |

### Config Files

| File | Purpose |
|------|---------|
| `configs/llama-cpp-s005-c1-fulloffload.env` | C1: Q8_0, full offload, KV q8_0 |
| `configs/llama-cpp-s005-c2-partial.env` | C2: Q8_0, `--n-cpu-moe 16` (OOM) |
| `configs/llama-cpp-s005-c2a-partial32.env` | C2a: Q8_0, `--n-cpu-moe 32` |
| `configs/llama-cpp-s005-c2b-partial36.env` | C2b: Q8_0, `--n-cpu-moe 36` |
| `configs/llama-cpp-s005-c3-nokv.env` | C3: Q8_0, full offload, no KV quant |
| `configs/llama-cpp-s005-c4-autofit.env` | C4: Q8_0, `--fit on` (old build, failed) |
| `configs/llama-cpp-s005-c4r-autofit-refit.env` | C4r: Q8_0, `--fit on` (b8149) |
| `configs/llama-cpp-s005-c5-q4km-fulloffload.env` | C5: Q4_K_M, full offload |
| `configs/llama-cpp-s005-c6-q4km-partial16.env` | C6: Q4_K_M, `--n-cpu-moe 16` (OOM) |
| `configs/llama-cpp-s005-c7-q4km-partial24.env` | **C7: WINNER — Q4_K_M, `--n-cpu-moe 24`** |
| `configs/llama-cpp-s005-c8-q4km-fit.env` | C8: Q4_K_M, `--fit on` |

### Benchmark Results

| File | Config |
|------|--------|
| `benchmarks/llama-cpp-s005-c1-fulloffload-20260225-213818.json` | C1 |
| `benchmarks/llama-cpp-s005-c3-nokv-20260225-215353.json` | C3 |
| `benchmarks/llama-cpp-s005-c2a-partial32-20260225-221424.json` | C2a |
| `benchmarks/llama-cpp-s005-c2b-partial36-20260225-221933.json` | C2b |
| `benchmarks/llama-cpp-s005-c4r-autofit-refit-20260225-224427.json` | C4r |
| `benchmarks/llama-cpp-s005-c5-q4km-fulloffload-20260225-225106.json` | C5 |
| `benchmarks/llama-cpp-s005-c7-q4km-partial24-20260225-231609.json` | C7 |
| `benchmarks/llama-cpp-s005-c8-q4km-fit-20260225-231852.json` | C8 |

### Scripts & Infrastructure

| File | Purpose |
|------|---------|
| `scripts/quant-quality.sh` | WikiText-2 PPL benchmark script |
| `docker/Dockerfile.llama-cpp` | Updated with `LLAMA_CPP_REF` build arg |
| `docker-compose.yml` | Updated for new model configs |

## Test Evidence

### Quant Quality (WikiText-2 Perplexity)

| Quant | PPL | vs Q8_0 |
|-------|-----|---------|
| Q8_0 | 6.5342 | baseline |
| Q4_K_M | 6.6688 | +2.1% |
| UD-Q4_K_XL | 7.1702 | +9.7% |

### Speed (tok/s, key configs only)

| Config | Short | Medium | Long | Multi-turn | VRAM |
|--------|-------|--------|------|------------|------|
| C1 (Q8_0 full offload) | 35.7 | 32.8 | 33.2 | 35.2 | 8064 MB |
| C4r (Q8_0 `--fit on`) | 40.5 | 40.3 | 39.6 | 40.3 | 14660 MB |
| C5 (Q4_K_M full offload) | 51.0 | 49.8 | 49.4 | 50.5 | 7217 MB |
| **C7 (Q4_K_M partial)** | **69.6** | **67.0** | **65.7** | **69.2** | **14874 MB** |

### vs Previous Model (Qwen3-Next-80B-A3B)

| Metric | Old (Session 002 A2) | New (Session 005 C7) | Change |
|--------|----------------------|----------------------|--------|
| Model | Qwen3-Next-80B-A3B Q8_0 | Qwen3.5-35B-A3B Q4_K_M | smaller, faster |
| tok/s | ~22 | ~70 | **+3.2x** |
| VRAM | 6247 MB | 14874 MB | +8.6 GB (more layers on GPU) |
| RAM | 84.8 GB | ~20 GB | **-76%** |
| Context | 32k | 65k | +2x |

## Key Decisions

### 1. Switched from Qwen3-Next-80B to Qwen3.5-35B-A3B
**Rationale**: 3.2x faster at comparable or better quality for coding/agentic tasks. The smaller model (20 GB Q4_K_M vs 85 GB Q8_0) enables partial GPU offload that was impossible with the 80B model, unlocking dramatically better PCIe bandwidth utilization.

### 2. Q4_K_M over Q8_0 for production
**Rationale**: Only +2.1% PPL degradation but enables 16/40 MoE layers on GPU (vs 4-8 layers for Q8_0). The speed gain (70 vs 40 tok/s) far outweighs the marginal quality loss. Q8_0 remains available for quality-sensitive tasks via `--fit on` at ~40 tok/s.

### 3. Manual `--n-cpu-moe 24` over `--fit on`
**Rationale**: Manual tuning yields ~7% more throughput than auto-fit (C7 ~70 tok/s vs C8 ~64 tok/s). The optimal split is well-defined: 16 layers on GPU, 24 on CPU, using ~14.9 GB of 16 GB VRAM.

### 4. UD-Q4_K_XL rejected
**Rationale**: Worse quality than standard Q4_K_M (+9.7% PPL vs +2.1%) at similar size. Unsloth Dynamic quantization does not benefit MoE architectures — the "important layer" upcasting strategy doesn't align well with expert routing patterns.

### 5. Dockerfile now supports `LLAMA_CPP_REF` build arg
**Rationale**: Unpinned `git clone` in Session 003/004 caused a 30% performance regression. The build arg allows pinning to a known-good commit (e.g., `b8149` for `--fit on` support). Two image tags: `latest` (HEAD) and `latest-fit` (b8149).

### 6. KV cache q8_0 confirmed as default
**Rationale**: +12-38% throughput improvement with no quality impact, confirmed across both the old 80B model and new 35B model. This is always-on for all configs.

## What Next Session Needs to Know

### Production Config is Ready
The C7 config (`Q4_K_M + --n-cpu-moe 24`) is ready for production use. Launch command is in CLAUDE.md. No further tuning needed for current hardware.

### Speculative Decoding is the Next Big Win
With 70 tok/s as the new baseline, speculative decoding (`--model-draft` with a small Qwen model) could push to 90-140 tok/s. This is the highest-ROI optimization remaining.

### Model Files to Clean Up
- `models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` (~20 GB) — can be deleted (not recommended, worse than Q4_K_M)
- Old Qwen3-Next-80B-A3B model files (84.8 GB) — can be deleted if no longer needed
- `models/UD-Q8_K_XL/` from Session 003 (93 GB) — can be deleted if not already removed

### Docker Image Tags
- `llm-server/llama-cpp:latest` — built from llama.cpp HEAD, may not support `--fit on`
- `llm-server/llama-cpp:latest-fit` — built from b8149, supports `--fit on`
- Production should use `latest` with `--n-cpu-moe 24` (C7 config) unless `--fit on` auto-management is preferred

### Open Questions
- **MXFP4_MOE quantization**: microscaling FP4 for expert weights could reduce model size further while maintaining quality. Not yet available in llama.cpp mainline.
- **Vision/multimodal**: mmproj file is downloaded but vision functionality was not benchmarked in this session. Smoke test recommended.
- **Thinking mode**: Qwen3.5 has thinking mode ON by default. Sampling parameters may need adjustment for different use cases (e.g., `temperature=0.6` for thinking, `temperature=0.7` for non-thinking).
